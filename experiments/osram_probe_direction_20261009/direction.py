"""Frozen history-probe directions in RAW Memory coordinates; no training or labels."""
from pathlib import Path

import numpy as np
import torch
from torch import nn


def _active(local, memory, availability, indices):
    n = len(local)
    for name, value, shape in (('local', local, (n, 256)), ('memory', memory, (n, 4, 512)),
                               ('availability', availability, (n, 3)), ('indices', indices, (n,))):
        if tuple(value.shape) != shape or value.device != memory.device:
            raise ValueError(f'{name}: invalid shape or device; expected {shape}')
    if not torch.isfinite(local).all():
        raise ValueError('Nonfinite local features')
    if not (((availability == 0) | (availability == 1)).all() and (availability.sum(-1) > 0).all()):
        raise ValueError('Availability must be binary and nonempty')
    if not (torch.isfinite(indices).all() and (indices >= 0).all() and (indices == indices.long()).all()):
        raise ValueError('indices must be nonnegative integral utterance positions')
    history = indices > 0
    return torch.cat((history[:, None], history[:, None] & ~availability.bool()), -1)


class FrozenHistoryProbe(nn.Module):
    """Strictly restore the original B 515→64→1 probe and training-only preprocessing.

    ``forward`` retains autograd only for raw Memory. All parameters, buffers,
    Local and availability are frozen; ``train()`` intentionally remains eval.
    """

    def __init__(self, weights_path, preprocessing_path, device='cpu'):
        super().__init__()
        checkpoint = torch.load(Path(weights_path), map_location='cpu', weights_only=True)
        if checkpoint.get('input_dim') != 515 or checkpoint.get('hidden_dim') != 64:
            raise ValueError('Expected B probe with input_dim=515 and hidden_dim=64')
        self.network = nn.Sequential(nn.Linear(515, 64), nn.GELU(), nn.Linear(64, 1))
        self.network.load_state_dict(checkpoint['state_dict'], strict=True)
        shapes = dict(local_mean=(256,), local_scale=(256,), memory_mean=(4, 512),
                      memory_scale=(4, 512), projection=(512, 64))
        with np.load(preprocessing_path, allow_pickle=False) as source:
            if set(source.files) != set(shapes):
                raise ValueError('Preprocessing keys do not match original B probe')
            for name, shape in shapes.items():
                value = source[name]
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f'Invalid preprocessing: {name}')
                if name.endswith('scale') and not (value > 0).all():
                    raise ValueError(f'Preprocessing scale must be positive: {name}')
                self.register_buffer(name, torch.tensor(value, dtype=torch.float32))
        if any(not torch.isfinite(p).all() for p in self.parameters()):
            raise ValueError('Nonfinite probe weights')
        self.requires_grad_(False)
        self.to(device)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    def forward(self, local, memory, availability, indices):
        active = _active(local, memory, availability, indices)
        # Sanitize BEFORE arithmetic; inactive NaN/Inf must never leak gradients.
        safe = torch.where(active[..., None], memory, self.memory_mean)
        normalized = torch.where(active[..., None],
                                 (safe - self.memory_mean) / self.memory_scale, 0.)
        local_z = (local.detach() - self.local_mean) / self.local_scale
        features = torch.cat((local_z, (normalized @ self.projection).flatten(1),
                              availability.detach()), dim=-1)
        return self.network(features).squeeze(-1)


def probe_direction(probe, local, memory, availability, indices):
    """Per-row d probe / d raw Memory, normalized jointly across active slots.

    No gold history or current label is accepted. Degenerate gradients have
    zero direction and valid_gradient=False; nonfinite derivatives are errors.
    """
    with torch.enable_grad():
        raw = memory.detach().clone().requires_grad_(True)
        prediction = probe(local.detach(), raw, availability.detach(), indices.detach())
        gradient = torch.autograd.grad(prediction.sum(), raw, create_graph=False)[0]
    if not (torch.isfinite(prediction).all() and torch.isfinite(gradient).all()):
        raise ValueError('Nonfinite probe prediction or raw-memory gradient')
    gradient = gradient.detach()
    norm = torch.linalg.vector_norm(gradient.double().flatten(1), dim=1)
    valid = norm > 1e-12
    u = torch.where(valid[:, None, None], gradient.double() / norm.clamp_min(1e-12)[:, None, None], 0.)
    return dict(prediction=prediction.detach(), gradient=gradient, u=u,
                gradient_norm=norm, valid_gradient=valid)


def split_delta(delta, u, valid_gradient):
    """Orthogonal projection in raw coordinates, evaluated/returned in float64.

    The caller casts interventions to the frozen Flat model's dtype. Errors are
    per-row raw Euclidean reconstruction norm and absolute inner product.
    """
    if delta.ndim != 3 or delta.shape[1:] != (4, 512) or u.shape != delta.shape or valid_gradient.shape != delta.shape[:1]:
        raise ValueError('Expected delta/u [N,4,512] and valid_gradient [N]')
    d = delta.detach().double()
    direction = torch.where(valid_gradient[:, None, None], u.detach().double(), 0.)
    if not (torch.isfinite(d).all() and torch.isfinite(direction).all()):
        raise ValueError('Nonfinite decomposition input')
    norms = direction.flatten(1).norm(dim=1)
    if not torch.allclose(norms[valid_gradient], torch.ones_like(norms[valid_gradient]), rtol=1e-6, atol=1e-8):
        raise ValueError('Valid probe direction must have unit norm')
    coefficient = (d * direction).sum((1, 2))
    history = coefficient[:, None, None] * direction
    other = d - history
    return dict(history=history, other=other, projection_coefficient=coefficient,
                reconstruction_error=(d - history - other).flatten(1).norm(dim=1),
                orthogonality_error=(history * other).sum((1, 2)).abs(),
                full_norm=d.flatten(1).norm(dim=1), history_norm=history.flatten(1).norm(dim=1),
                other_norm=other.flatten(1).norm(dim=1))


@torch.no_grad()
def flat_predict(model, local, memory, availability, indices):
    """Replay original causal Flat readout on valid rows only, with zero rear halves."""
    from experiments.osram_history_drift_20261002.diagnostic import _validate

    backbone = _validate(model)
    for name in ('osram_meaningful_block', 'osram_readout_candidate'):
        if getattr(backbone, name, 'none') != 'none':
            raise ValueError(f'Adapted readout forbidden: {name}')
    for name in ('meaningful_input_mode', 'osram_gap_increment_filter', 'osram_relation_block',
                 'osram_relation_dual_readout', 'osram_decision_correction'):
        if getattr(backbone, name, False):
            raise ValueError(f'Adapted readout forbidden: {name}')
    if backbone.context_dim != 1024:
        raise ValueError('Expected four 1024-dimensional original context slots')
    active = _active(local, memory, availability, indices)
    safe = torch.where(active[..., None], memory, 0.)
    if not torch.isfinite(safe).all():
        raise ValueError('Nonfinite active Memory')
    full = torch.cat((safe, torch.zeros_like(safe)), dim=-1)
    fusion = torch.cat((local, full.flatten(1)), dim=-1)
    hidden = backbone.emotion_norm(backbone.local_skip(local) + backbone.emotion_adapter(fusion))
    prediction = model.smax_fc(hidden)
    if prediction.shape != (len(local), 1) or not torch.isfinite(prediction).all():
        raise ValueError('Expected finite scalar regression prediction per valid row')
    return prediction.squeeze(-1)
