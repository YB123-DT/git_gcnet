"""Read-only original Flat traces and label-free, within-split memory donors.

Train features use one fixed offline GCNet-random view per rate and loader
batch. This is a diagnostic fitting view, not the source's cyclic training
schedule. Test features must replay the winning epoch's saved availability.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from experiments.osram_meld_gcnet_random_no_skip_evidence_20261014.evaluate import (
    CANONICAL_MASK_PROTOCOL, replay_view_from_availability, scatter_saved_availability,
)
from experiments.osram_meld_write_retention_20261014.run import WriteCapture, state_digest
from experiments.osram_meld_write_retention_20261014.task_replay import TaskCapture
from gcnet_modality_jepa.gcnet_legacy_random_mask import random_mask


def extract_split(model, loader, config, dimensions, rate, split, saved=None,
                  mask_seed=66, max_batches=None):
    """Capture one unmodified causal source forward per batch, before each write.

    Rows follow the source prediction NPZ: batch-major valid utterances, then
    successive loader batches. No padded state, future state, or other
    conversation enters a row. Source weights stay frozen after extraction.
    """
    from gcnet_missing_m3.train_gcnet import (
        _collect_predictions, _move_batch, _prepare_view_from_primary_masks,
    )

    if config.dataset != 'MELD':
        raise ValueError('Query-vs-Decision traces require MELD')
    if split == 'test' and saved is None:
        raise ValueError('Test requires the winning epoch saved availability and predictions')
    if max_batches is not None and max_batches < 1:
        raise ValueError('max_batches must be positive')
    if not np.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError('rate must be between zero and one')
    model.eval().requires_grad_(False)
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError('Frozen source must have no existing parameter gradients')
    before = state_digest(model)
    device = next(model.parameters()).device
    rng = np.random.RandomState(mask_seed)
    saved_arrays = None
    if saved is not None:
        saved_arrays = {name: np.asarray(saved[name]) for name in
                        ('availability', 'predictions', 'labels')}
        if len({len(v) for v in saved_arrays.values()}) != 1:
            raise ValueError('Saved availability/prediction/label counts disagree')
        availability = saved_arrays['availability']
        if availability.ndim != 2 or availability.shape[1] != 3:
            raise ValueError('Saved availability must have shape [N,3]')
        if not np.isin(availability, (0, 1)).all() or not availability.any(axis=1).all():
            raise ValueError('Saved valid availability must be binary and nonempty')

    parts = defaultdict(list)
    cursor = batches = 0
    with torch.no_grad():
        for raw in loader:
            if max_batches is not None and batches >= max_batches:
                break
            data = _move_batch(raw, device)
            valid = data[7].bool()
            if valid.ndim != 2 or not valid.any(dim=1).all():
                raise ValueError('Each conversation requires a nonempty [B,L] validity mask')
            if ((~valid[:, :-1]) & valid[:, 1:]).any():
                raise ValueError('Conversation validity must be a contiguous prefix')
            batch_size, length = valid.shape
            count = int(valid.sum())
            start = cursor
            if saved_arrays is not None:
                availability, cursor = scatter_saved_availability(
                    saved_arrays['availability'], cursor, data[7])
                view = replay_view_from_availability(data, availability, dimensions)
            else:
                sides = []
                for _ in ('host', 'guest'):
                    sampled = random_mask(3, length * batch_size, rate, rng=rng)
                    shaped = sampled.reshape(batch_size, length, 3).transpose(1, 0, 2)
                    side = torch.as_tensor(shaped, dtype=torch.float32, device=device)
                    sides.append(torch.where(valid.T[..., None], side, torch.zeros_like(side)))
                view = _prepare_view_from_primary_masks(data, *sides, dimensions)
                cursor += count
            with TaskCapture(model.osram) as task, WriteCapture(model.osram, view['umask']) as writer:
                logits, _, _, completion = model(
                    [view['incomplete']], view['availability'], view['qmask'],
                    view['umask'], view['lengths'], predict_missing=False)
            if (completion is not None or task.calls != 1 or task.flat.local_calls != 1
                    or task.flat.adapter_calls != 1 or len(writer.snapshots) != len(writer.times)):
                raise RuntimeError('Expected one original causal Flat forward with Local Skip')
            if not torch.equal(task.valid, valid.T):
                raise RuntimeError('Captured scan validity differs from model input')
            predicted, labels, _ = _collect_predictions(
                config.dataset, logits, view['labels'], view['umask'], config.mosi_task_mode)
            if saved_arrays is not None:
                np.testing.assert_array_equal(labels, saved_arrays['labels'][start:cursor])
                np.testing.assert_array_equal(predicted, saved_arrays['predictions'][start:cursor])

            memory = task.queries.new_zeros(length, batch_size, model.osram.num_heads,
                                            model.osram.value_dim, model.osram.key_dim)
            keys = task.queries.new_zeros(length, batch_size, model.osram.num_heads,
                                          model.osram.key_dim, 3)
            for t, prewrite, observed_keys, *_ in writer.snapshots:
                memory[t], keys[t] = prewrite, observed_keys
            observed = task.availability.bool()
            if not torch.equal(observed, view['availability'].bool()):
                raise RuntimeError('Captured availability differs from the fixed feature view')
            if observed[~valid.T].any() or not observed[valid.T].any(dim=-1).all():
                raise RuntimeError('Padding must be unavailable and valid rows must have an observation')
            history_seen = observed.to(torch.int64).cumsum(dim=0) - observed.to(torch.int64)
            tensors = dict(local=task.local, memory=memory, queries=task.queries,
                           keys=keys, availability=task.availability, logits=logits,
                           base=task.base, gap=task.gap)
            for name, tensor in tensors.items():
                packed = tensor.transpose(0, 1)[valid].float().cpu().numpy()
                if not np.isfinite(packed).all():
                    raise ValueError(f'Nonfinite frozen source trace: {name}')
                parts[name].append(packed)
            parts['labels'].append(labels.astype(np.int64, copy=False))
            ids = np.asarray([str(v) for v in view['conversation_ids']])
            if len(ids) != batch_size:
                raise ValueError('Conversation ID count does not match batch')
            times = torch.arange(length, device=device).expand(batch_size, -1)
            parts['conversation_ids'].append(np.broadcast_to(ids[:, None], valid.shape)[valid.cpu().numpy()])
            parts['utterance_indices'].append(times[valid].cpu().numpy())
            parts['has_history'].append((times[valid] > 0).cpu().numpy())
            parts['history_seen'].append(history_seen.transpose(0, 1)[valid].gt(0).cpu().numpy())
            batches += 1
    if not batches:
        raise ValueError('Cannot extract an empty split')
    full_coverage = saved_arrays is None or cursor == len(saved_arrays['labels'])
    if saved_arrays is not None and max_batches is None and not full_coverage:
        raise ValueError('Saved predictions were not fully consumed')
    after = state_digest(model)
    if before != after or any(p.grad is not None or p.requires_grad for p in model.parameters()):
        raise RuntimeError('Frozen source parameters or gradients changed during extraction')
    arrays = {name: np.concatenate(values) for name, values in parts.items()}
    checks = dict(split=split, rate=float(rate), batches=batches, utterances=cursor,
                  source_state_sha256_before=before, source_state_sha256_after=after,
                  source_state_unchanged=True, source_gradients_none=True,
                  memory_scan_count=batches, saved_predictions_exact=saved_arrays is not None,
                  saved_labels_exact=saved_arrays is not None,
                  saved_count=None if saved_arrays is None else len(saved_arrays['labels']),
                  full_saved_coverage=full_coverage, partial=max_batches is not None,
                  mask_protocol=CANONICAL_MASK_PROTOCOL, mask_seed=int(mask_seed),
                  mask_scope=('winning-epoch saved selected-speaker availability' if saved_arrays is not None
                              else 'one fixed rate-specific offline feature view per loader batch; '
                              'shared by all diagnostic head/query fits; not source cyclic training'))
    return arrays, checks


def make_donor_indices(data, seed=66):
    """Match only current availability/history status and a different dialogue.

    Labels and historical modality coverage are deliberately ignored. Groups
    have contiguous conversation blocks so exclusion costs O(1) per sampled row,
    with uniform sampling over eligible rows. Unmatched rows retain index -1.
    """
    conversations = np.asarray(data['conversation_ids']).astype(str)
    availability = np.asarray(data['availability'])
    has_history = np.asarray(data['has_history'], dtype=bool)
    count = len(conversations)
    if availability.shape != (count, 3) or has_history.shape != (count,):
        raise ValueError('Donor metadata shapes must be [N], [N,3], [N]')
    if not np.isin(availability, (0, 1)).all():
        raise ValueError('Donor availability must be binary')
    groups = defaultdict(lambda: defaultdict(list))
    for i in range(count):
        key = (*availability[i].astype(int).tolist(), bool(has_history[i]))
        groups[key][conversations[i]].append(i)
    rng = np.random.RandomState(seed)
    donors = np.full(count, -1, dtype=np.int64)
    for conversations_in_group in groups.values():
        if len(conversations_in_group) < 2:
            continue
        pool = np.asarray([i for rows in conversations_in_group.values() for i in rows], dtype=np.int64)
        start = 0
        for rows in conversations_in_group.values():
            own_count = len(rows)
            draws = rng.randint(0, len(pool) - own_count, size=own_count)
            draws = np.where(draws < start, draws, draws + own_count)
            donors[rows] = pool[draws]
            start += own_count
    return donors, donors >= 0
