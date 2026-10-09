"""Fixed, label-free, one-bit past interventions on frozen original cfg84."""
from __future__ import annotations

import numpy as np

MODALITIES = ('A', 'T', 'V')
FAMILIES = ('remove_T_vs_A', 'remove_T_vs_V') + tuple(
    f'{prefix}_{m}' for prefix in ('recent_vs_earlier', 'same_vs_other_speaker')
    for m in MODALITIES)


def delete_bit(availability, target, position):
    """Copy a single-conversation mask and delete exactly one valid past bit."""
    mask = np.asarray(availability)
    p, m = position
    if not (0 <= p < target < len(mask) and 0 <= m < 3):
        raise ValueError('Deletion must be strictly before the target')
    if mask[p, m] != 1 or mask[p].sum() <= 1:
        raise ValueError('Deletion must retain at least one observed modality')
    changed = mask.copy()
    changed[p, m] = 0
    return changed


def select_pairs(availability, target, speakers=None):
    """Prefer exact lag matching, then closest lag, then most recent positions.

    The recent/earlier control is the next earlier eligible position. Speaker
    comparisons require at least two real speaker IDs in the conversation.
    Labels and predictions are intentionally absent from this interface.
    """
    mask = np.asarray(availability)
    if mask.ndim != 2 or mask.shape[1] != 3 or not np.isin(mask, (0, 1)).all():
        raise ValueError('Expected binary availability [time,3]')
    if not 0 <= target < len(mask) or not (mask.sum(-1) > 0).all():
        raise ValueError('Expected valid target and nonempty utterances')
    eligible = [np.flatnonzero((mask[:target, m] == 1) &
                              (mask[:target].sum(-1) > 1)).tolist() for m in range(3)]
    pairs, skipped = [], {}

    def add(family, left, right):
        if not left or not right:
            skipped[family] = 'insufficient_eligible_history'
            return
        a, b = min(((a, b) for a in left for b in right),
                   key=lambda pair: (abs(pair[0][0] - pair[1][0]),
                                     -min(pair[0][0], pair[1][0]),
                                     -max(pair[0][0], pair[1][0])))
        pairs.append(dict(family=family, delete=a, control=b,
                          lag_delete=target - a[0], lag_control=target - b[0],
                          lag_mismatch=abs(a[0] - b[0])))

    for other in (0, 2):
        add(f'remove_T_vs_{MODALITIES[other]}', [(p, 1) for p in eligible[1]],
            [(p, other) for p in eligible[other]])
    for m, name in enumerate(MODALITIES):
        positions = eligible[m]
        add(f'recent_vs_earlier_{name}', [(positions[-1], m)] if positions else [],
            [(positions[-2], m)] if len(positions) > 1 else [])
    speaker = None if speakers is None else np.asarray(speakers)
    if speaker is not None and speaker.shape != (len(mask),):
        raise ValueError('Speaker IDs must have one entry per utterance')
    speaker_available = speaker is not None and len(np.unique(speaker)) > 1
    for m, name in enumerate(MODALITIES):
        family = f'same_vs_other_speaker_{name}'
        if not speaker_available:
            skipped[family] = 'speaker_unavailable'
        else:
            add(family, [(p, m) for p in eligible[m] if speaker[p] == speaker[target]],
                [(p, m) for p in eligible[m] if speaker[p] != speaker[target]])
    return pairs, skipped


def evaluate_interventions(model, view, dimensions, baseline, rate, split, seed=66):
    """Scan each unique deletion once, in batches of 16 independent conversations.

    Reuse the caller baseline and cache only requested target rows on CPU. Each
    deletion cache key includes conversation, position and modality, so reuse
    across targets and families cannot mix different counterfactual histories.
    """
    import torch
    from experiments.osram_history_drift_20261002.diagnostic import capture, vector_drift

    mask = view['availability'].detach().cpu().numpy()
    qmask = view['qmask'].detach().cpu().numpy()
    coverage = dict(targets=0, eligible=0, skipped=0, speaker_unique_counts={},
                    families={name: dict(eligible=0, skipped=0, reasons={}) for name in FAMILIES},
                    deterministic=True, seed=int(seed))
    plans = []
    for b, length in enumerate(view['lengths']):
        length = int(length)
        speakers = qmask[b, :length]
        # _prepare_view uses scalar speaker IDs; do not infer identities from features.
        if speakers.ndim != 1:
            speakers = None
        unique = 0 if speakers is None else int(len(np.unique(speakers)))
        coverage['speaker_unique_counts'][str(view['conversation_ids'][b])] = unique
        for t in range(length):
            coverage['targets'] += 1
            pairs, skipped = select_pairs(mask[:length, b], t, speakers)
            for pair in pairs:
                plans.append(dict(pair, batch_index=b, target=t))
                coverage['families'][pair['family']]['eligible'] += 1
                coverage['eligible'] += 1
            for family, reason in skipped.items():
                stats = coverage['families'][family]
                stats['skipped'] += 1
                stats['reasons'][reason] = stats['reasons'].get(reason, 0) + 1
                coverage['skipped'] += 1

    def snapshot(chunk, arm):
        indices = torch.tensor([p['batch_index'] for p in chunk], device=view['incomplete'].device)
        availability = view['availability'].index_select(1, indices).clone()
        incomplete = view['incomplete'].index_select(1, indices).clone()
        offsets = np.cumsum((0,) + tuple(dimensions)).tolist()
        if arm is not None:
            for j, plan in enumerate(chunk):
                pos, mod = plan[arm]
                # Validate the exact intervention used by the tensor model call.
                delete_bit(mask[:int(view['lengths'][plan['batch_index']]), plan['batch_index']],
                           plan['target'], (pos, mod))
                availability[pos, j, mod] = 0
                incomplete[pos, j, offsets[mod]:offsets[mod + 1]] = 0
                assert torch.equal(incomplete[plan['target']:, j],
                                   view['incomplete'][plan['target']:, plan['batch_index']])
        speaker = view['qmask'].index_select(0, indices)
        valid = view['umask'].index_select(0, indices)
        lengths = [int(view['lengths'][p['batch_index']]) for p in chunk]
        return capture(model, lambda: model([incomplete], availability, speaker, valid,
                                             lengths, predict_missing=False))

    def drift(a, b):
        result = vector_drift(a.reshape(-1), b.reshape(-1))
        return {key: (bool(value) if key == 'valid' else
                      (float(value) if torch.isfinite(value) else None))
                for key, value in result.items()}

    keys = ('local', 'base', 'gap', 'prediction')
    requested = {}
    for plan in plans:
        for arm in ('delete', 'control'):
            key = (plan['batch_index'], *plan[arm])
            requested.setdefault(key, set()).add(plan['target'])
    jobs = [dict(batch_index=b, delete=(pos, mod), target=min(targets))
            for (b, pos, mod), targets in requested.items()]
    coverage.update(unique_deletions=len(jobs), scans=0, original_scans=0,
                    avoided_repeated_deletions=2 * len(plans) - len(jobs))
    cache, originals = {}, {}
    rows = []
    with torch.no_grad():
        for plan in plans:
            t, b = plan['target'], plan['batch_index']
            if (b, t) not in originals:
                originals[b, t] = {key: baseline[key][t, b].detach().cpu().clone() for key in keys}
        for start in range(0, len(jobs), 16):
            chunk = jobs[start:start + 16]
            captured = snapshot(chunk, 'delete')
            coverage['scans'] += 1
            for j, job in enumerate(chunk):
                key = (job['batch_index'], *job['delete'])
                cache[key] = {t: {name: captured[name][t, j].detach().cpu().clone() for name in keys}
                              for t in requested[key]}
            del captured
        for start in range(0, len(plans), 16):
            chunk = plans[start:start + 16]
            for j, plan in enumerate(chunk):
                t, b = plan['target'], plan['batch_index']
                real = originals[b, t]
                deleted = cache[(b, *plan['delete'])][t]
                control = cache[(b, *plan['control'])][t]
                y = float(view['labels'][b, t])
                pred = float(real['prediction'].reshape(-1)[0])
                row = dict(conversation_id=str(view['conversation_ids'][b]), utterance_index=t,
                           rate=float(rate), split=split, family=plan['family'], y=y,
                           pattern=int(np.dot(mask[t, b], [1, 2, 4])), pred_real=pred,
                           lag_mismatch=plan['lag_mismatch'], deleted_bits=1,
                           control_deleted_bits=1, current_local_max_error=0.,
                           current_local_relative_error=0.)
                if not np.isfinite([y, pred]).all():
                    raise AssertionError('Nonfinite label or original prediction')
                original_correct = (pred > 0) == (y > 0)
                for arm, snapshot_ in (('delete', deleted), ('control', control)):
                    p = float(snapshot_['prediction'].reshape(-1)[0])
                    if not np.isfinite(p):
                        raise AssertionError('Nonfinite intervention prediction')
                    error = (real['local'] - snapshot_['local']).double()
                    absolute = float(error.abs().max())
                    relative = float(error.norm() / (real['local'].double().norm() + 1e-8))
                    if absolute > 1e-5 or relative > 1e-6:
                        raise AssertionError(f'Current local changed: {absolute}, {relative}')
                    row['current_local_max_error'] = max(row['current_local_max_error'], absolute)
                    row['current_local_relative_error'] = max(row['current_local_relative_error'], relative)
                    correct = (p > 0) == (y > 0)
                    pos, mod = plan[arm]
                    row.update({f'pred_{arm}': p, f'delta_mse_{arm}': (p-y)**2 - (pred-y)**2,
                                f'flip_{arm}': (p > 0) != (pred > 0),
                                f'correction_{arm}': y != 0 and not original_correct and correct,
                                f'harm_{arm}': y != 0 and original_correct and not correct,
                                f'position_{arm}': pos, f'modality_{arm}': MODALITIES[mod],
                                f'lag_{arm}': t-pos, f'local_max_error_{arm}': absolute})
                    row[f'base_drift_{arm}'] = drift(real['base'], snapshot_['base'])
                    active = torch.as_tensor(1-mask[t, b], device=real['gap'].device).bool()
                    gap0 = real['gap'] * active[:, None]
                    gap1 = snapshot_['gap'] * active[:, None]
                    row[f'gap_drift_{arm}'] = drift(gap0, gap1)
                    row[f'gap_all_drift_{arm}'] = drift(real['gap'], snapshot_['gap'])
                row['delta_mse_delete_minus_control'] = row['delta_mse_delete'] - row['delta_mse_control']
                rows.append(row)
    return dict(rows=rows, coverage=coverage)
