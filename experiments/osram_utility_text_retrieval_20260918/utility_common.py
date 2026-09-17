"""Frozen PAM-E Reader utility computation for utility-supervised retrieval."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Mapping

import torch
from torch.nn import functional as F

from gcnet_missing_m3 import train_gcnet as tr
from experiments.osram_pam_episodic_text_20260917.run import configuration as pam_e_configuration
from experiments.osram_pam_address_text_20260918.address_probe import build_model

READER_ROOT = Path('/data2/yb/remote_experiments/osram_pam_episodic_text_20260917/mosi')
FEATURE_NAMES = ('wav2vec-large-c-UTT', 'deberta-large-4-UTT', 'manet_UTT')
FEATURES = Path('/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset/CMUMOSI/features')
MODALITIES = ('audio', 'text', 'visual')


def reader_path(seed: int, rate: str) -> Path:
    tag = str(rate).replace('.', 'p')
    return READER_ROOT / f'seed_{seed}' / f'best_miss_{tag}.pt'


def build_reader(seed: int, rate: str, device: torch.device | None = None):
    cfg, _, _ = pam_e_configuration(seed)
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
    path = reader_path(seed, rate)
    if not path.exists():
        raise FileNotFoundError(f'Missing PAM-E Reader checkpoint: {path}')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    model = build_model(cfg, dims, shape, torch.device('cpu'))
    model.load_state_dict(payload['model'], strict=True)
    model.to(device or torch.device('cuda'))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, cfg, dims, shape, (train_loaders, val_loaders, test_loaders)


def _observed_memory_states(model, encoded, latents, availability, qmask, umask, lengths):
    valid = model.osram._validate_inputs(encoded, latents, availability, qmask, umask, lengths)
    keys, values, _ = model.osram._project_sequence(
        encoded, latents, availability, qmask, read_node=encoded, write_node=encoded
    )
    length, batch = availability.shape[:2]
    memory = encoded.new_zeros(
        batch, model.osram.num_heads, model.osram.value_dim, model.osram.key_dim
    )
    alpha = torch.sigmoid(model.osram.alpha_logits).to(dtype=encoded.dtype)
    beta = torch.sigmoid(model.osram.beta_logits).to(dtype=encoded.dtype)
    read_memories, slot_key_list = [], []
    for t in range(length):
        active = valid[t]
        slot_keys = torch.stack([keys[name][t] for name in MODALITIES], dim=-1)
        slot_values = torch.stack([values[name][t] for name in MODALITIES], dim=-1)
        slot_keys = slot_keys * availability[t].to(encoded.dtype).unsqueeze(1).unsqueeze(2)
        slot_values = slot_values * availability[t].to(encoded.dtype).unsqueeze(1).unsqueeze(2)
        decayed = memory * alpha.view(1, model.osram.num_heads, 1, 1)
        memory = torch.where(active.view(batch, 1, 1, 1), decayed, memory)
        read_memories.append(memory)
        slot_key_list.append(slot_keys)
        candidate = model.osram.block_write(memory, slot_keys, slot_values, availability[t], beta=beta)
        memory = torch.where(active.view(batch, 1, 1, 1), candidate, memory)
    return read_memories, slot_key_list


def _batched_hidden_logits(model, fused, read_memory, slot_keys, availability, qmask):
    """fused [Q,C,D], read_memory [Q,H,V,K], slot_keys [Q,H,K,3]."""
    q_count, c_count, latent_dim = fused.shape
    node_norm = model.osram.node_norm(fused)  # [Q,C,D]
    avail_embed = model.osram.availability_embedding(availability)  # [Q,3D]
    avail_embed = avail_embed.unsqueeze(1).expand(-1, c_count, -1)
    speaker = model.osram.speaker_embedding(qmask.long())  # [Q,3D]
    speaker = speaker.unsqueeze(1).expand(-1, c_count, -1)
    common = torch.cat((node_norm, avail_embed, speaker), dim=-1)  # [Q,C,3D]
    type_embedding = model.osram.query_type_embedding.weight.view(1, 1, 4, -1)
    type_embedding = type_embedding.expand(q_count, c_count, -1, -1)
    query_input = torch.cat((common.unsqueeze(2).expand(-1, -1, 4, -1), type_embedding), dim=-1)
    queries = model.osram.query_projector(query_input).view(
        q_count, c_count, 4, model.osram.num_heads, model.osram.key_dim
    )
    queries = F.normalize(queries, dim=-1)  # [Q,C,4,H,K]

    read_memory = read_memory.unsqueeze(1)  # [Q,1,H,V,K]
    base_read = torch.einsum("qchvk,qchk->qchv", read_memory.expand(-1, c_count, -1, -1, -1), queries[:, :, 0])
    base_read = base_read.reshape(q_count, c_count, -1)
    gap_halves = []
    for modality_index in range(3):
        residual_query = model.osram._address_residual(
            slot_keys.unsqueeze(1).expand(-1, c_count, -1, -1, -1).reshape(q_count * c_count, model.osram.num_heads, model.osram.key_dim, 3),
            queries[:, :, modality_index + 1].reshape(q_count * c_count, model.osram.num_heads, model.osram.key_dim),
        ).reshape(q_count, c_count, model.osram.num_heads, model.osram.key_dim)
        gap_read = torch.einsum("qchvk,qchk->qchv", read_memory.expand(-1, c_count, -1, -1, -1), residual_query)
        gap_read = gap_read.reshape(q_count, c_count, -1)
        missing = 1.0 - availability[:, modality_index].view(q_count, 1)
        gap_halves.append(gap_read * missing.view(q_count, 1, 1))
    gap_forward = torch.stack(gap_halves, dim=2)  # [Q,C,3,H*V]
    half_dim = base_read.shape[-1]
    base_context = torch.cat((base_read, base_read.new_zeros(q_count, c_count, half_dim)), dim=-1)
    gap_context = torch.cat(
        (gap_forward, gap_forward.new_zeros(q_count, c_count, 3, half_dim)), dim=-1
    )
    local = fused + model.osram.local_path(fused)
    missing_full = (1.0 - availability).unsqueeze(1).expand(-1, c_count, -1)
    emotion_input = torch.cat(
        (local, base_context, (gap_context * missing_full.unsqueeze(-1)).reshape(q_count, c_count, -1)),
        dim=-1,
    )
    hidden = model.osram.emotion_norm(
        model.osram.local_skip(local) + model.osram.emotion_adapter(emotion_input)
    )
    return model.smax_fc(hidden)  # [Q,C,n_out]


def _vector_task_loss(cfg, logits, labels):
    """Return [Q,C] existing task loss for each candidate."""
    if cfg.mosi_task_mode == "regression":
        scores = logits[..., 0]
        target = labels.view(-1, 1).expand_as(scores)
        if cfg.task_regression_loss == "mse":
            return (scores - target) ** 2
        if cfg.task_regression_loss == "smooth-l1":
            return F.smooth_l1_loss(scores, target, beta=cfg.task_smooth_l1_beta, reduction="none")
        raise ValueError(f"unsupported regression loss {cfg.task_regression_loss}")
    if cfg.mosi_task_mode == "binary":
        scores = logits[..., 0]
        target = (labels > 0).to(dtype=scores.dtype).view(-1, 1).expand_as(scores)
        return F.binary_cross_entropy_with_logits(scores, target, reduction="none")
    raise ValueError(f"utility task loss not implemented for {cfg.mosi_task_mode}")


@torch.no_grad()
def query_utility_from_batch(model, cfg, view: Mapping[str, torch.Tensor]):
    """Return padded tensors for all true Text-missing queries in a batch."""
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
    pam = model.pam_episodic_memory(latents, availability, umask, base_context, gap_context[..., 1, :])
    zero = encoded.new_zeros((*availability.shape[:2], model.latent_dim))
    reg = torch.stack((zero, pam['z_hat_text'], zero), dim=2)
    read_node_base = model.completed_read_fusion(
        encoded, latents, reg, availability, umask, active_mask=pam['text_missing_mask']
    )
    text_observed = availability[..., 1].bool() & umask.T.bool()
    text_missing = pam['text_missing_mask']
    read_memories, slot_key_list = _observed_memory_states(
        model, encoded, latents, availability, qmask, umask, lengths
    )

    query_specs = []
    for t in range(availability.shape[0]):
        for b in range(availability.shape[1]):
            if not bool(text_missing[t, b]):
                continue
            history = [i for i in range(t) if bool(text_observed[i, b])]
            candidates = [pam['z_hat_text'][t, b]] + [latents['text'][i, b] for i in history]
            query_specs.append((t, b, candidates))
    if not query_specs:
        return None

    q_count = len(query_specs)
    c_max = max(len(candidates) for _, _, candidates in query_specs)
    values = torch.zeros(q_count, c_max, model.latent_dim, device=device)
    mask = torch.zeros(q_count, c_max, dtype=torch.bool, device=device)
    ts = torch.tensor([item[0] for item in query_specs], device=device, dtype=torch.long)
    bs = torch.tensor([item[1] for item in query_specs], device=device, dtype=torch.long)
    for q_index, (_, _, candidates) in enumerate(query_specs):
        values[q_index, :len(candidates)] = torch.stack(candidates)
        mask[q_index, :len(candidates)] = True

    # Candidate read nodes through the frozen CompletedReadFusion.
    observed_q = encoded[ts, bs]  # [Q,D]
    latents_q = {name: latents[name][ts, bs] for name in MODALITIES}
    avail_q = availability[ts, bs]
    reg_q = torch.stack(
        (
            observed_q.new_zeros(q_count, c_max, model.latent_dim),
            values,
            observed_q.new_zeros(q_count, c_max, model.latent_dim),
        ),
        dim=2,
    )
    flat_q = q_count * c_max
    fused = model.completed_read_fusion(
        observed_q.unsqueeze(1).expand(-1, c_max, -1).reshape(1, flat_q, model.latent_dim),
        {name: latents_q[name].reshape(1, q_count, 1, model.latent_dim).expand(-1, -1, c_max, -1).reshape(1, flat_q, model.latent_dim) for name in MODALITIES},
        reg_q.reshape(1, flat_q, 3, model.latent_dim),
        avail_q.reshape(1, q_count, 1, 3).expand(-1, -1, c_max, -1).reshape(1, flat_q, 3),
        torch.ones(flat_q, 1, device=device),
        active_mask=torch.ones(1, flat_q, dtype=torch.bool, device=device),
    ).reshape(q_count, c_max, model.latent_dim)
    fused = fused.clone()
    fused[:, 0] = read_node_base[ts, bs].detach()

    read_memory_q = torch.stack([read_memories[t][b] for t, b, _ in query_specs], dim=0)
    slot_keys_q = torch.stack([slot_key_list[t][b] for t, b, _ in query_specs], dim=0)
    qmask_q = qmask[bs, ts]
    logits = _batched_hidden_logits(model, fused, read_memory_q, slot_keys_q, avail_q, qmask_q)
    labels_q = labels[bs, ts]
    losses = _vector_task_loss(cfg, logits, labels_q)
    scores = logits[..., 0].detach()
    utility = losses[:, :1] - losses
    mask_float = mask.to(dtype=losses.dtype)
    utility = utility * mask_float
    # Candidate 0 is always the frozen original score and has zero utility.
    utility[:, 0] = 0.0

    pattern = torch.stack(
        (avail_q[:, 0], avail_q[:, 2], avail_q[:, 0] * avail_q[:, 2]), dim=-1
    )
    context = torch.cat(
        (
            latents_q['audio'] * avail_q[:, 0:1],
            latents_q['visual'] * avail_q[:, 2:3],
            pattern,
            base_context[ts, bs],
            gap_context[ts, bs, 1, :],
        ),
        dim=-1,
    )
    return {
        'context': context.detach().cpu(),
        'values': values.detach().cpu(),
        'mask': mask.cpu(),
        'scores': scores.detach().cpu(),
        'losses': losses.detach().cpu(),
        'utility': utility.detach().cpu(),
        'labels': labels_q.detach().cpu(),
        'patterns': pattern.detach().cpu(),
    }
