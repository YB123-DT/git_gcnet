"""Frozen source-key audit. Every alternate state starts after a real joint write.

These are storage diagnostics, not emotion predictions. Source keys are used
only as offline probes; current Base/Gap queries and task heads are untouched.
"""

from __future__ import annotations

import math
import torch
from torch.nn import functional as F

MODALITIES = ('A', 'T', 'V')
BRANCHES = ('normal', 'decay_only', 'write_only', 'hold')


@torch.no_grad()
def candidate_pool(snapshots, source_index, batch_index, modality, count=4):
    """Target plus latest strict-past same-modality distractors, fixed per source.

Returns [candidate, head, dimension]. No other conversation or future is used.
Sources lacking exactly count candidates have no centered/retrieval score.
"""
    indices = [i for i in range(source_index, -1, -1)
               if bool(snapshots[i][4][batch_index, modality])][:count]
    if len(indices) != count:
        return None
    keys = torch.stack([snapshots[i][2][batch_index, ..., modality] for i in indices])
    values = torch.stack([snapshots[i][3][batch_index, ..., modality] for i in indices])
    return keys, values


@torch.no_grad()
def replay_forks(osram, snapshots, source_index, *, distances=(1, 2, 4, 8)):
    """Yield strictly-future pre-write states and normal replay parity error.

Retention executes before each future read. A joint write occurs only AFTER
that read, so the source_t+1 measurement cannot contain intervening overwrites.
Normal/write_only always use the original jointly observed A/T/V columns.
"""
    if not distances or any(d < 1 for d in distances):
        raise ValueError('distances must be positive')
    start = snapshots[source_index][-1]
    states = {name: start.clone() for name in BRANCHES}
    alpha = torch.sigmoid(osram.alpha_logits).to(start).view(1, -1, 1, 1)
    beta = torch.sigmoid(osram.beta_logits).to(start)
    source_time = snapshots[source_index][0]
    wanted = set(distances)
    for time_index, actual_read, keys, values, availability, _ in snapshots[source_index + 1:]:
        distance = time_index - source_time
        if distance > max(distances):
            break
        valid = availability.bool().any(-1)
        active = valid.view(-1, 1, 1, 1)
        for name in ('normal', 'decay_only'):
            states[name] = torch.where(active, states[name] * alpha, states[name])
        parity = (states['normal'] - actual_read)[valid]
        error = float(parity.abs().max()) if parity.numel() else 0.
        torch.testing.assert_close(states['normal'][valid], actual_read[valid],
                                   rtol=2e-5, atol=2e-5)
        if distance in wanted:
            yield distance, {name: value.clone() for name, value in states.items()}, error
        # Preserve one joint block solve per branch, rather than sequential writes.
        for name in ('normal', 'write_only'):
            updated = osram.block_write(states[name], keys, values, availability, beta=beta)
            states[name] = torch.where(active, updated, states[name])


def _safe_cosine(a, b):
    if float(a.norm()) <= 1e-8 or float(b.norm()) <= 1e-8:
        return None
    return float(F.cosine_similarity(a, b, dim=0, eps=1e-8))


def _metrics(memory, key, value, pool, head):
    prediction = memory @ key
    if not torch.isfinite(prediction).all():
        raise FloatingPointError('nonfinite source-key reconstruction')
    target_norm = float(value.norm())
    result = {
        'relative_error': float((prediction - value).norm()) / max(target_norm, 1e-8),
        'target_norm': target_norm,
        'prediction_norm': float(prediction.norm()),
        'norm_ratio': float(prediction.norm()) / max(target_norm, 1e-8),
        'raw_cosine': _safe_cosine(prediction, value),
        'centered_cosine': None, 'retrieval_top1': None, 'retrieval_mrr': None,
        'wrong_key_centered_cosine': None, 'wrong_key_relative_error': None,
        'candidate_count': 0,
    }
    if pool is None:
        return result
    keys, candidates = pool
    candidates = candidates[:, head]
    center = candidates.mean(0)
    result['candidate_count'] = len(candidates)
    result['centered_cosine'] = _safe_cosine(prediction - center, value - center)
    wrong = memory @ keys[1, head]
    result['wrong_key_centered_cosine'] = _safe_cosine(wrong - center, value - center)
    result['wrong_key_relative_error'] = float((wrong - value).norm()) / max(target_norm, 1e-8)
    scores = F.cosine_similarity(prediction[None] - center, candidates - center, dim=-1, eps=1e-8)
    # Constant/zero centered vectors cannot establish retrieval; mark unavailable.
    if float((prediction - center).norm()) > 1e-8 and bool((candidates - center).norm(dim=-1).gt(1e-8).all()):
        higher = int((scores > scores[0] + 1e-7).sum())
        ties = int((scores - scores[0]).abs().le(1e-7).sum())
        result['retrieval_top1'] = 1. / ties if higher == 0 else 0.
        result['retrieval_mrr'] = sum(1. / (higher + i + 1) for i in range(ties)) / ties
    return result


@torch.no_grad()
def _metric_arrays(memory, keys, values, pools):
    """Score all source/head pairs together and transfer scalars once per batch."""
    prediction = (memory @ keys.unsqueeze(-1)).squeeze(-1)
    norm = values.norm(dim=-1)
    pred_norm = prediction.norm(dim=-1)
    if not torch.isfinite(prediction).all():
        raise FloatingPointError('nonfinite source-key reconstruction')
    eligible = torch.tensor([p is not None for p in pools], device=keys.device)
    candidate_keys = torch.stack([p[0] if p is not None else keys.new_zeros(4, *keys.shape[1:])
                                 for p in pools])
    candidates = torch.stack([p[1] if p is not None else values.new_zeros(4, *values.shape[1:])
                              for p in pools])
    center = candidates.mean(dim=1)
    wrong = (memory @ candidate_keys[:, 1].unsqueeze(-1)).squeeze(-1)
    def cos(a, b, active):
        score = F.cosine_similarity(a, b, dim=-1, eps=1e-8)
        valid = active & a.norm(dim=-1).gt(1e-8) & b.norm(dim=-1).gt(1e-8)
        return score.masked_fill(~valid, float('nan'))
    pool_mask = eligible[:, None].expand_as(norm)
    centered_prediction = prediction - center
    scores = F.cosine_similarity(centered_prediction[:, None], candidates - center[:, None],
                                 dim=-1, eps=1e-8)
    higher = (scores > scores[:, :1] + 1e-7).sum(dim=1)
    ties = (scores - scores[:, :1]).abs().le(1e-7).sum(dim=1)
    retrieval_valid = pool_mask & centered_prediction.norm(dim=-1).gt(1e-8) & (
        candidates - center[:, None]).norm(dim=-1).gt(1e-8).all(dim=1)
    mrr = torch.zeros_like(norm)
    for index in range(4):
        mrr += torch.where(ties > index, 1. / (higher + index + 1), torch.zeros_like(norm))
    fields = {
        'relative_error': (prediction - values).norm(dim=-1) / norm.clamp_min(1e-8),
        'target_norm': norm, 'prediction_norm': pred_norm,
        'norm_ratio': pred_norm / norm.clamp_min(1e-8),
        'raw_cosine': cos(prediction, values, torch.ones_like(pool_mask)),
        'centered_cosine': cos(centered_prediction, values - center, pool_mask),
        'retrieval_top1': torch.where(higher == 0, 1. / ties, torch.zeros_like(norm)).masked_fill(
            ~retrieval_valid, float('nan')),
        'retrieval_mrr': (mrr / ties).masked_fill(~retrieval_valid, float('nan')),
        'wrong_key_centered_cosine': cos(wrong - center, values - center, pool_mask),
        'wrong_key_relative_error': ((wrong - values).norm(dim=-1) / norm.clamp_min(1e-8)).masked_fill(
            ~pool_mask, float('nan')),
        'candidate_count': pool_mask.to(norm.dtype) * 4,
    }
    arrays = torch.stack(list(fields.values()), dim=-1).cpu().numpy()
    return {name: arrays[..., i] for i, name in enumerate(fields)}


@torch.no_grad()
def audit_batch(osram, snapshots, sample_ids, rate, *, distances=(1, 2, 4, 8)):
    """Stream scalar per-source/per-head rows, never retain dataset states."""
    if not snapshots:
        return
    if len(sample_ids) != snapshots[0][1].shape[0]:
        raise ValueError('conversation IDs do not match state batch')
    if any(snapshots[i][0] >= snapshots[i + 1][0] for i in range(len(snapshots) - 1)):
        raise ValueError('snapshots must be chronological')
    availability_cpu = [s[4].bool().cpu().tolist() for s in snapshots]
    for source_index, (source_time, pre, keys, values, observed, post) in enumerate(snapshots):
        sources = [(b, m) for b in range(len(sample_ids)) for m in range(3)
                   if availability_cpu[source_index][b][m]]
        if not sources:
            continue
        pools = [candidate_pool(snapshots, source_index, b, m) for b, m in sources]
        source_keys = torch.stack([keys[b, ..., m] for b, m in sources])
        source_values = torch.stack([values[b, ..., m] for b, m in sources])
        source_batches = torch.tensor([b for b, m in sources], device=keys.device)
        after_error = ((post[source_batches] @ source_keys.unsqueeze(-1)).squeeze(-1)
                       - source_values).norm(dim=-1) / source_values.norm(dim=-1).clamp_min(1e-8)
        after_error = after_error.cpu().numpy()

        def rows_for(states, phase, distance, error):
            target_index = next((i for i in range(source_index, len(snapshots))
                                 if snapshots[i][0] == source_time + distance), None)
            for branch, memory in states.items():
                scored = _metric_arrays(memory[source_batches], source_keys, source_values, pools)
                for index, (b, m) in enumerate(sources):
                    if phase == 'future' and (target_index is None or not any(availability_cpu[target_index][b])):
                        continue
                    for head in range(osram.num_heads):
                        metrics = {name: (float(array[index, head]) if math.isfinite(float(array[index, head]))
                                          else None) for name, array in scored.items()}
                        metrics['candidate_count'] = int(metrics['candidate_count'])
                        yield {
                            'conversation_id': str(sample_ids[b]), 'rate': float(rate),
                            'source_t': int(source_time), 'target_t': int(source_time + distance),
                            'distance': int(distance), 'modality': MODALITIES[m], 'head': head,
                            'phase': phase, 'branch': branch, 'replay_error': float(error),
                            **metrics,
                            'write_after_relative_error': float(after_error[index, head]),
                        }
        yield from rows_for({'write_before': pre, 'write_after': post}, 'write', 0, 0.)
        for distance, branches, error in replay_forks(osram, snapshots, source_index, distances=distances):
            yield from rows_for(branches, 'future', distance, error)
