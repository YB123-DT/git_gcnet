"""Frozen PAM-E Reader utility cache built with real model.forward calls.

Candidate 0 is always a direct normal Reader forward.  Candidate i>0 substitutes
one historical real Text latent into the current T-missing Completion slot via
``pam_text_override`` and runs the real ``MissingM3GraphModel.forward``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Mapping

import torch

from gcnet_missing_m3 import train_gcnet as tr
from gcnet_missing_m3.train_gcnet import TrainConfig
from experiments.osram_pam_address_text_20260918.address_probe import build_model

READER_ROOT = Path('/data2/yb/remote_experiments/osram_pam_episodic_text_20260917/mosi')
FEATURE_NAMES = ('wav2vec-large-c-UTT', 'deberta-large-4-UTT', 'manet_UTT')
FEATURES = Path('/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset/CMUMOSI/features')
MODALITIES = ('audio', 'text', 'visual')


def reader_path(seed: int, rate: str) -> Path:
    tag = str(rate).replace('.', 'p')
    return READER_ROOT / f'seed_{seed}' / f'best_miss_{tag}.pt'


def build_reader(seed: int, rate: str, device: torch.device | None = None):
    """Load the target PAM-E checkpoint with its own saved TrainConfig."""
    path = reader_path(seed, rate)
    payload = torch.load(path, map_location='cpu', weights_only=False)
    cfg = TrainConfig(**payload['config'])
    if device is not None:
        cfg = replace(cfg, device=str(device))
    shape = tr._resolve_task_contract(cfg.dataset, cfg.mosi_task_mode)
    loaders = tr.get_loaders(
        audio_root=str(FEATURES / FEATURE_NAMES[0]),
        text_root=str(FEATURES / FEATURE_NAMES[1]),
        video_root=str(FEATURES / FEATURE_NAMES[2]),
        num_folder=int(shape['num_folds']),
        dataset=cfg.dataset,
        batch_size=cfg.batch_size,
        num_workers=0,
        seed=cfg.seed,
        validation_fraction=cfg.validation_fraction,
        evaluation_protocol=cfg.evaluation_protocol,
    )
    train_loaders, val_loaders, test_loaders, adim, tdim, vdim = loaders
    dims = (adim, tdim, vdim)
    model = build_model(cfg, dims, shape, torch.device('cpu'))
    model.load_state_dict(payload['model'], strict=True)
    model.to(device or torch.device('cuda'))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, cfg, dims, shape, (train_loaders, val_loaders, test_loaders)


def candidate_task_loss(cfg, logit: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    return tr._task_loss(
        cfg.dataset,
        logit.reshape(1, 1, -1),
        label.reshape(1, 1),
        torch.ones(1, 1, device=label.device),
        cfg.mosi_task_mode,
        cfg.task_regression_loss,
        cfg.task_smooth_l1_beta,
    )


@torch.no_grad()
def query_utility_from_batch(model, cfg, view: Mapping[str, torch.Tensor], batch_index: int = 0):
    """Build padded candidate records for one raw loader batch.

    All candidate task scores are produced by real ``model.forward`` calls.
    """
    device = view['incomplete'].device
    availability = view['availability']
    umask = view['umask']
    qmask = view['qmask']
    labels = view['labels']
    lengths = view['lengths']

    encoded, latents = model.observed_set(view['incomplete'], availability, umask)
    base_context, gap_context = model.osram.causal_read_contexts(
        encoded, latents, availability, qmask, umask, lengths
    )
    # PAM-E original predicted Text is only used as candidate-0 value (not key)
    # and for context diagnostics; its score comes from the real normal forward.
    pam = model.pam_episodic_memory(
        latents, availability, umask, base_context, gap_context[..., 1, :]
    )
    text_observed = availability[..., 1].bool() & umask.T.bool()
    text_missing = pam['text_missing_mask']

    # Candidate 0: one true normal Reader forward for the whole batch.
    normal_logits, _, _, _ = model(
        view['incomplete'], availability, qmask, umask, lengths
    )
    normal_scores = normal_logits[..., 0]

    query_specs = []
    for b in range(availability.shape[1]):
        for t in range(availability.shape[0]):
            if not bool(text_missing[t, b]):
                continue
            history = [i for i in range(t) if bool(text_observed[i, b])]
            query_specs.append((t, b, history))
    if not query_specs:
        return None

    q_count = len(query_specs)
    c_max = max(1 + len(history) for _, _, history in query_specs)
    values = torch.zeros(q_count, c_max, model.latent_dim, device=device)
    mask = torch.zeros(q_count, c_max, dtype=torch.bool, device=device)
    scores = torch.zeros(q_count, c_max, device=device)
    losses = torch.zeros(q_count, c_max, device=device)
    utility = torch.zeros(q_count, c_max, device=device)
    labels_out = torch.zeros(q_count, device=device)
    patterns = torch.zeros(q_count, 3, device=device)
    query_t = torch.zeros(q_count, dtype=torch.long, device=device)
    query_b = torch.zeros(q_count, dtype=torch.long, device=device)
    context = torch.zeros(q_count, 0, device=device)

    for q_index, (t, b, history) in enumerate(query_specs):
        candidates = [pam['z_hat_text'][t, b]] + [latents['text'][i, b] for i in history]
        count = len(candidates)
        values[q_index, :count] = torch.stack(candidates)
        mask[q_index, :count] = True
        labels_out[q_index] = labels[b, t]
        query_t[q_index] = t
        query_b[q_index] = b
        # Candidate 0 score is exactly the normal Reader score.
        scores[q_index, 0] = normal_scores[t, b]
        losses[q_index, 0] = candidate_task_loss(cfg, normal_scores[t, b].reshape(1), labels[b, t])
        if history:
            # One real full override forward for all historical candidates of
            # this query (each candidate is a separate batch column).
            c_hist = len(history)
            tiled_incomplete = view['incomplete'][:, b:b + 1].expand(-1, c_hist, -1)
            tiled_avail = availability[:, b:b + 1].expand(-1, c_hist, -1)
            tiled_qmask = qmask[b].unsqueeze(0).expand(c_hist, -1)
            tiled_umask = umask[b].unsqueeze(0).expand(c_hist, -1)
            override = torch.zeros(
                (*tiled_avail.shape[:2], model.latent_dim), device=device
            )
            override_mask = torch.zeros(tiled_avail.shape[:2], dtype=torch.bool, device=device)
            for h_index, source_index in enumerate(history):
                override[t, h_index] = latents['text'][source_index, b]
                override_mask[t, h_index] = True
            logits, _, _, _ = model(
                tiled_incomplete,
                tiled_avail,
                tiled_qmask,
                tiled_umask,
                [lengths[b]] * c_hist,
                pam_text_override=override,
                pam_override_mask=override_mask,
            )
            scores[q_index, 1:1 + c_hist] = logits[t, :, 0]
            for h_index in range(c_hist):
                losses[q_index, 1 + h_index] = candidate_task_loss(
                    cfg, logits[t, h_index], labels[b, t]
                )
        utility[q_index, :count] = losses[q_index, 0] - losses[q_index, :count]
        utility[q_index, 0] = 0.0
        a_mask = availability[t, b, 0]
        v_mask = availability[t, b, 2]
        patterns[q_index] = torch.stack((a_mask, v_mask, a_mask * v_mask))
        q_context = torch.cat(
            (
                latents['audio'][t, b] * a_mask,
                latents['visual'][t, b] * v_mask,
                patterns[q_index],
                base_context[t, b],
                gap_context[t, b, 1, :],
            ),
            dim=-1,
        ).reshape(1, -1)
        if context.shape[1] == 0:
            context = q_context.new_zeros((q_count, q_context.shape[1]))
        context[q_index] = q_context

    valid_flat = umask.T.bool()
    full_scores = normal_scores[valid_flat]
    full_labels = labels.T[valid_flat]
    full_availability = availability[valid_flat]
    full_positions = torch.full((*valid_flat.shape,), -1, dtype=torch.long, device=device)
    full_positions[valid_flat] = torch.arange(full_scores.shape[0], device=device)
    full_index = full_positions[ts, bs] if query_specs else torch.zeros(0, dtype=torch.long, device=device)
    return {
        'context': context.detach().cpu(),
        'values': values.detach().cpu(),
        'mask': mask.cpu(),
        'scores': scores.detach().cpu(),
        'losses': losses.detach().cpu(),
        'utility': utility.detach().cpu(),
        'labels': labels_out.detach().cpu(),
        'patterns': patterns.detach().cpu(),
        'query_t': query_t.cpu(),
        'query_b': query_b.cpu(),
        'batch_index': torch.full((q_count,), int(batch_index), dtype=torch.long),
        'full_index': full_index.cpu(),
        'full_scores': full_scores.detach().cpu(),
        'full_labels': full_labels.detach().cpu(),
        'full_availability': full_availability.detach().cpu(),
    }
