"""Frozen, label-evaluated write utility; labels never select interventions.

Rows and aggregate counts are PAIR-weighted, not unique-utterance metrics.
One target may appear under multiple source or missing-modality conditions.
"""
from collections import defaultdict
from dataclasses import asdict

import torch

from .mc_fwc import forward_view, _target_loss, _validate_pairs, _rng_state, _set_rng


_MODALITIES = ('A', 'T', 'V')


def _aggregate(rows):
    n = len(rows)
    return dict(
        pair_count=n,
        loss_keep_mean=sum(r['loss_keep'] for r in rows) / max(1, n),
        loss_omit_mean=sum(r['loss_omit'] for r in rows) / max(1, n),
        utility_mean=sum(r['utility'] for r in rows) / max(1, n),
        utility_positive_count=sum(r['utility'] > 0 for r in rows),
        utility_negative_count=sum(r['utility'] < 0 for r in rows),
        utility_zero_count=sum(r['utility'] == 0 for r in rows),
        polarity_eligible_count=sum(r['keep_correct'] is not None for r in rows),
        corrections=sum(r['correction'] for r in rows),
        harms=sum(r['harm'] for r in rows),
    )


def _summarize(rows, omit_forwards):
    summary = dict(overall=_aggregate(rows), omit_forwards=omit_forwards,
                   counting_unit='legal source-target pair; targets may repeat',
                   unique_target_count=len({(r['conversation_id'], r['future_j']) for r in rows}))
    summary['repeated_target_pair_count'] = len(rows) - summary['unique_target_count']
    for field, name in [('source_modality', 'by_source_modality'),
                        ('missing_modality', 'by_missing_modality'),
                        ('lag', 'by_lag'), ('regime', 'by_regime')]:
        groups = defaultdict(list)
        for row in rows:
            key = _MODALITIES[row[field]] if field.endswith('modality') else str(row[field])
            groups[key].append(row)
        summary[name] = {key: _aggregate(values) for key, values in sorted(groups.items())}
    return summary


def summarize_write_utility(rows, omit_forwards=0):
    """Aggregate concatenated batch rows with the same pair-count convention.

    For multiple evaluation conditions, aggregate each condition separately;
    conversation/time identity alone does not identify a condition trajectory.
    """
    return _summarize(list(rows), int(omit_forwards))


@torch.no_grad()
def evaluate_write_utility(model, view, pairs, dataset='CMUMOSI'):
    """Run one Keep and independent per-conversation Omit interventions.

    The caller must set eval mode. Different pairs from the same conversation
    require different rounds, whereas independent conversations share a round.
    Inputs/availability remain fixed and only the one write column is removed.
    Uses the original regression threshold >0, excluding zero gold labels from
    polarity statistics. CE uses argmax. All task losses include neutral labels.
    """
    if model.training:
        raise ValueError('write utility evaluation requires model.eval()')
    if dataset not in ('CMUMOSI', 'CMUMOSEI', 'IEMOCAPFour', 'IEMOCAPSix'):
        raise ValueError('unsupported task dataset')
    pairs = list(pairs)
    _validate_pairs(view, pairs)
    if not pairs:
        return dict(rows=[], summary=_summarize([], 0))
    device = view['incomplete'].device
    initial_rng = _rng_state(device)
    context = getattr(model, 'last_osram_context', None)
    diagnostics = getattr(model.osram, 'last_diagnostics', None)
    rows = [None] * len(pairs)
    groups = defaultdict(list)
    for i, pair in enumerate(pairs):
        groups[pair.batch].append(i)
    rounds = [[ids[r] for ids in groups.values() if r < len(ids)]
              for r in range(max(map(len, groups.values())))]
    try:
        keep = forward_view(model, view)
        for indices in rounds:
            override = view['availability'].clone()
            for i in indices:
                p = pairs[i]
                override[p.source_t, p.batch, p.source_modality] = 0
            # Eval should be deterministic; restoring also protects stochastic
            # custom evaluation layers without changing caller RNG state.
            _set_rng(initial_rng, device)
            omit = forward_view(model, view, write_mask_override=override)
            for i in indices:
                p = pairs[i]
                loss_keep = float(_target_loss(keep, view, p, dataset))
                loss_omit = float(_target_loss(omit, view, p, dataset))
                if not torch.isfinite(torch.tensor([loss_keep, loss_omit])).all():
                    raise FloatingPointError('nonfinite write utility evaluation loss')
                y = view['labels'][p.batch, p.future_j].item()
                kp, op = keep[p.future_j, p.batch], omit[p.future_j, p.batch]
                if dataset in ('CMUMOSI', 'CMUMOSEI'):
                    prediction_keep, prediction_omit = float(kp.reshape(())), float(op.reshape(()))
                    positive_keep, positive_omit = prediction_keep > 0, prediction_omit > 0
                    kc = positive_keep == (y > 0) if y != 0 else None
                    oc = positive_omit == (y > 0) if y != 0 else None
                    scores_keep, scores_omit = prediction_keep, prediction_omit
                else:
                    prediction_keep, prediction_omit = int(kp.argmax()), int(op.argmax())
                    positive_keep = positive_omit = None  # class IDs have no binary polarity semantics
                    kc, oc = prediction_keep == int(y), prediction_omit == int(y)
                    scores_keep, scores_omit = kp.tolist(), op.tolist()
                rows[i] = dict(asdict(p), lag=p.future_j-p.source_t,
                               loss_keep=loss_keep, loss_omit=loss_omit, utility=loss_omit-loss_keep,
                               prediction_keep=prediction_keep, prediction_omit=prediction_omit,
                               scores_keep=scores_keep, scores_omit=scores_omit, label=y,
                               positive_keep=positive_keep, positive_omit=positive_omit,
                               keep_correct=kc, omit_correct=oc,
                               correction=bool(kc is True and oc is False),
                               harm=bool(kc is False and oc is True))
    finally:
        _set_rng(initial_rng, device)
        model.last_osram_context = context
        model.osram.last_diagnostics = diagnostics
    return dict(rows=rows, summary=_summarize(rows, len(rounds)))
