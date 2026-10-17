"""Small, reversible frozen interventions; no replacement CHM equations."""
from contextlib import contextmanager
import math

import numpy as np
import torch


@contextmanager
def parameter_override(osram, *, alpha=None, beta=None):
    """Temporarily set sigmoid-space constants, restoring original logits exactly.

    Diagnostic only: eval-mode forward under no_grad, never optimizer steps.
    Keeping the same Parameter objects preserves the original numerical path.
    """
    edits = {}
    for name, value in (('alpha_logits', alpha), ('beta_logits', beta)):
        if value is not None:
            value = float(value)
            if not math.isfinite(value) or not 0 < value < 1:
                raise ValueError('Frozen alpha/beta constants must lie strictly in (0,1)')
            edits[name] = math.log(value / (1-value))
    saved = {name: getattr(osram, name).detach().clone() for name in edits}
    try:
        with torch.no_grad():
            for name, logit in edits.items():
                getattr(osram, name).fill_(logit)
        yield
    finally:
        with torch.no_grad():
            for name, original in saved.items():
                getattr(osram, name).copy_(original)


def mask_head(flat_input, *, local_dim, context_dim, head, value_dim, kind):
    """Mask one forward read head in Base or all active Gap slots, not memory.

    Captured original Flat input already masks inactive Gap and padding. No bias,
    projection or synthetic evidence is introduced by this readout intervention.
    """
    if (kind not in ('base', 'gap') or context_dim % 2 or value_dim < 1
            or head < 0 or (head+1)*value_dim > context_dim//2
            or flat_input.shape[-1] != local_dim+4*context_dim):
        raise ValueError('Invalid head/readout shape or intervention kind')
    edited = flat_input.clone()
    for slot in ([0] if kind == 'base' else [1, 2, 3]):
        offset = local_dim+slot*context_dim+head*value_dim
        edited[..., offset:offset+value_dim] = 0
    return edited


def spectral_rows(keys, availability, valid, beta, ridge):
    """Active-column spectra at actual joint writes, in float64 for diagnosis.

    Not the padded 3x3 Gram spectrum: missing columns are excluded first. Embedded
    beta makes the equivalent unscaled solve use diag(ridge/beta_active), so we
    save both weighted spectra/ridge ratios and that diagonal. No condition
    number is fabricated when the smallest eigenvalue is numerically zero.
    """
    if keys.ndim != 4 or keys.shape[-1] != 3:
        raise ValueError('keys must be [B,H,K,3]')
    batch, heads = keys.shape[:2]
    if (availability.shape != (batch, 3) or valid.shape != (batch,)
            or beta.shape != (heads, 3) or not math.isfinite(ridge) or ridge <= 0):
        raise ValueError('Invalid spectral diagnostic shapes or ridge')
    k = keys.detach().cpu().double().numpy()
    available = availability.detach().cpu().bool().numpy()
    is_valid = valid.detach().cpu().bool().numpy()
    b = beta.detach().cpu().double().numpy()
    if not np.isfinite(b).all() or np.any(b <= 0):
        raise ValueError('Embedded beta must be finite and positive')
    rows = []
    for index in np.flatnonzero(is_valid):
        active = np.flatnonzero(available[index])
        if not len(active):
            raise ValueError('Valid utterance cannot have no observed modality')
        for head in range(heads):
            columns = k[index, head][:, active]
            if not np.isfinite(columns).all():
                raise ValueError('Nonfinite active keys')
            gram = columns.T @ columns
            scale = np.sqrt(b[head, active])
            weighted = gram*scale[:, None]*scale[None, :]
            eig = np.linalg.eigvalsh(gram).clip(min=0)
            weighted_eig = np.linalg.eigvalsh(weighted).clip(min=0)
            rows.append(dict(batch=int(index), head=int(head),
                active_modalities=active.tolist(),
                gram_eigenvalues=eig.tolist(),
                weighted_gram_eigenvalues=weighted_eig.tolist(),
                effective_ridge_diagonal=(ridge/b[head, active]).tolist(),
                weighted_eigenvalues_below_ridge=int((weighted_eig < ridge).sum()),
                weighted_min_eigenvalue_over_ridge=float(weighted_eig[0]/ridge),
                weighted_max_eigenvalue_over_ridge=float(weighted_eig[-1]/ridge)))
    return rows


@torch.no_grad()
def alpha_forks(osram, snapshots, alpha, distances=(1, 2, 4, 8)):
    """Change retention only *after* each fixed original source post-write state.

    Reuse actual future joint-write K/V and observed masks. Yield the target's
    pre-write state. These are paired source-target cohorts, not full-testset
    predictions or effects of a single isolated source association.
    """
    if not snapshots:
        return
    if not distances or any(distance < 1 for distance in distances):
        raise ValueError('Future distances must be positive')
    alpha = torch.as_tensor(alpha, device=snapshots[0][1].device,
                            dtype=snapshots[0][1].dtype)
    if alpha.numel() not in (1, osram.num_heads) or not bool(torch.isfinite(alpha).all()) or not bool(((alpha > 0) & (alpha < 1)).all()):
        raise ValueError('Require scalar or per-head retention in (0,1)')
    decay = alpha.reshape(1, -1, 1, 1)
    by_time = {row[0]: row for row in snapshots}
    if len(by_time) != len(snapshots):
        raise ValueError('Duplicate snapshot time')
    for source_t, *_, post in snapshots:
        memory = post.clone()
        for distance in range(1, max(distances)+1):
            target_t = source_t+distance
            if target_t not in by_time:
                break
            _, _, keys, values, available, _ = by_time[target_t]
            valid = available.bool().any(-1)
            memory = torch.where(valid[:, None, None, None], memory*decay, memory)
            if distance in distances:
                yield source_t, target_t, memory.clone()
            memory = osram.block_write(memory, keys, values, available)
