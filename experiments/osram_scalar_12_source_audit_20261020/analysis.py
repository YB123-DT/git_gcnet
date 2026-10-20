"""Frozen MOSI/MOSEI source-removal statistics, never model training.

Historical ``logits_*`` array names contain scalar regression predictions.
Polarity metrics exclude zero gold labels and use prediction > 0 exactly.
"""
import numpy as np

from experiments.osram_12_source_evidence_audit_20261019.analysis import (
    COALITIONS, EPS, ROWS, SOURCES, _shapley_values, _summary,
)


def _metrics(labels, predictions):
    labels = np.asarray(labels, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    eligible = np.abs(labels) > 0
    n, npol = len(labels), int(eligible.sum())
    result = dict(N=n, N_polarity=npol, ACC=None, WF1=None,
                  MSE=float(np.mean((labels - predictions) ** 2)) if n else None)
    if npol:
        gold, pred = labels[eligible] > 0, predictions[eligible] > 0
        confusion = np.array([[(gold == a)[pred == b].sum() for b in (False, True)]
                              for a in (False, True)], dtype=np.int64)
        support, predicted = confusion.sum(1), confusion.sum(0)
        correct = confusion.diagonal()
        denominator = support + predicted
        f1 = np.divide(2 * correct, denominator, out=np.zeros(2), where=denominator > 0)
        result.update(ACC=float(100 * correct.sum() / npol),
                      WF1=float(100 * np.dot(support / npol, f1)))
    return result


def _paired(labels, full, removed):
    original, intervention = _metrics(labels, full), _metrics(labels, removed)
    eligible = np.abs(labels) > 0
    npol = int(eligible.sum())
    counts = dict(corrections=None, harms=None, both_correct=None, both_wrong=None)
    if npol:
        correct_full = (full[eligible, 0] > 0) == (labels[eligible] > 0)
        correct_removed = (removed[eligible, 0] > 0) == (labels[eligible] > 0)
        counts = dict(corrections=int((~correct_full & correct_removed).sum()),
                      harms=int((correct_full & ~correct_removed).sum()),
                      both_correct=int((correct_full & correct_removed).sum()),
                      both_wrong=int((~correct_full & ~correct_removed).sum()))
    return dict(N=len(labels), N_polarity=npol, full_metrics=original,
                removed_metrics=intervention, **counts,
                **{'delta_' + name: None if original[name] is None else intervention[name] - original[name]
                   for name in ('ACC', 'WF1', 'MSE')})


def _validate(arrays):
    required = ('labels', 'logits_full', 'logits_removed', 'logits_coalitions', 'source_reads',
                'row_active', 'trajectory', 'state_source_norms', 'state_joint_norm',
                'availability', 'conversation_ids', 'utterance_indices')
    data = {name: np.asarray(arrays[name]) for name in required}
    n = len(data['labels'])
    shapes = dict(labels=(n,), logits_full=(n, 1), logits_removed=(n, 4, 3, 1),
                  logits_coalitions=(n, 8, 1), row_active=(n, 4), trajectory=(n, 3, 4),
                  state_source_norms=(n, 3), state_joint_norm=(n,), availability=(n, 3),
                  conversation_ids=(n,), utterance_indices=(n,))
    for name, shape in shapes.items():
        if data[name].shape != shape:
            raise ValueError(f'{name} shape {data[name].shape} != {shape}')
    if data['source_reads'].ndim != 4 or data['source_reads'].shape[:3] != (n, 4, 3):
        raise ValueError('source_reads must have shape [N,4,3,D]')
    for name in required:
        if name != 'conversation_ids' and not np.isfinite(data[name]).all():
            raise ValueError(f'{name} must be finite')
    availability = data['availability']
    if not np.isin(availability, (0, 1)).all() or np.any(availability.sum(1) < 1):
        raise ValueError('availability must be binary and nonempty')
    expected_active = np.column_stack((np.ones(n, dtype=bool), ~availability.astype(bool)))
    if not np.array_equal(data['row_active'], expected_active):
        raise ValueError('row_active must enable Base and only currently missing Gap rows')
    if np.any(data['source_reads'][~expected_active] != 0):
        raise ValueError('inactive evidence source reads must be zero')
    if not np.allclose(data['logits_coalitions'][:, 7], data['logits_full'], atol=1e-6, rtol=1e-6):
        raise ValueError('full coalition does not reproduce full scalar prediction')
    trajectory = data['trajectory']
    if (np.any(trajectory[:, :, 0] < 0) or np.any(trajectory[:, :, 1] < 0)
            or np.any(trajectory[:, :, 1] > 1) or not np.isin(trajectory[:, :, 3], (0, 1)).all()
            or not np.array_equal(trajectory[:, :, 0] > 0, trajectory[:, :, 3].astype(bool))):
        raise ValueError('invalid count/coverage/ever-seen trajectory')
    if np.any(data['state_source_norms'] < 0) or np.any(data['state_joint_norm'] < 0):
        raise ValueError('state norms must be nonnegative')
    return data


def analyze_arrays(arrays, dataset=None):
    """Same condition schema as the categorical audit, but ACC/WF1/MSE only.

    All per-slot comparisons use exactly the active row cohort; classification
    metrics/counts further exclude zero-label utterances. MSE includes them.
    """
    if dataset is not None and dataset not in ('CMUMOSI', 'CMUMOSEI', 'MOSI', 'MOSEI'):
        raise ValueError('Only MOSI/MOSEI scalar regression is supported')
    data = _validate(arrays)
    labels, full = data['labels'].astype(np.float64), data['logits_full']
    active = data['row_active'].astype(bool)
    reads = data['source_reads'].astype(np.float64)
    norms = np.linalg.norm(reads, axis=-1)
    norm_sum = norms.sum(-1)
    cancellation = np.clip(1 - np.linalg.norm(reads.sum(2), axis=-1) / np.maximum(norm_sum, EPS), 0, 1)
    losses = (labels[:, None] - data['logits_coalitions'][:, :, 0].astype(np.float64)) ** 2
    shapley = _shapley_values(losses)

    def cohort(mask):
        n = int(mask.sum())
        slots, cancelled = {}, {}
        for row_index, row in enumerate(ROWS):
            eligible = mask & active[:, row_index]
            slots[row] = {}
            for source_index, source in enumerate(SOURCES):
                item = _paired(labels[eligible], full[eligible],
                               data['logits_removed'][eligible, row_index, source_index])
                item['read_norm'] = _summary(norms[eligible, row_index, source_index])
                slots[row][source] = item
            cancelled[row] = _summary(cancellation[eligible & (norm_sum[:, row_index] > EPS), row_index])
        credits = shapley[mask]
        gain = losses[mask, 0] - losses[mask, 7]
        metric = _metrics(labels[mask], full[mask])
        return dict(N=n, N_polarity=metric['N_polarity'], full_metrics=metric, slots=slots,
                    cancellation=cancelled,
                    source_states={s: _summary(data['state_source_norms'][mask, i]) for i, s in enumerate(SOURCES)},
                    joint_state_norm=_summary(data['state_joint_norm'][mask]),
                    coalitions={name: _metrics(labels[mask], data['logits_coalitions'][mask, i])
                                for i, name in enumerate(COALITIONS)},
                    shapley=dict(mean={s: float(credits[:, i].mean()) if n else None for i, s in enumerate(SOURCES)},
                                 summary={s: _summary(credits[:, i]) for i, s in enumerate(SOURCES)},
                                 full_minus_zero_gain=float(gain.mean()) if n else None,
                                 efficiency_max_abs_error=float(np.abs(credits.sum(1) - gain).max()) if n else None))

    output = cohort(np.ones(len(labels), dtype=bool))
    current_bits = data['availability'].astype(int) @ np.array([1, 2, 4])
    ever = data['trajectory'][:, :, 3].astype(bool)
    history_bits = ever.astype(int) @ np.array([1, 2, 4])
    strata = dict(availability={COALITIONS[i]: cohort(current_bits == i) for i in range(1, 8)},
                  history_ever={COALITIONS[i]: cohort(history_bits == i) for i in range(8)}, source_history={})
    for i, source in enumerate(SOURCES):
        seen, coverage = ever[:, i], data['trajectory'][:, i, 1]
        bins = {'never': ~seen}
        for lower, upper, name in ((0, .25, '(0,0.25]'), (.25, .5, '(0.25,0.5]'),
                                   (.5, .75, '(0.5,0.75]'), (.75, 1, '(0.75,1]')):
            bins[name] = seen & (coverage > lower) & (coverage <= upper)
        strata['source_history'][source] = dict(ever_seen={'never': cohort(~seen), 'seen': cohort(seen)},
                                                coverage={name: cohort(mask) for name, mask in bins.items()})
    output['strata'] = strata
    output['definitions'] = dict(
        metrics='ACC/WF1 percent: abs(gold)>0, positive iff prediction>0; MSE uses all valid labels',
        delta='removed minus original, same active-row cohort; positive ACC/WF1 means removal improves score',
        corrections_harms='Only nonzero gold labels: full-wrong -> removed-correct / full-correct -> removed-wrong',
        shapley='Exact source-group Shapley, payoff=-per-utterance MSE, bits A1/T2/V4; not W-F1 attribution',
        source='Joint-write Value-slot contribution, not pure modality semantics or causal attribution',
        inactive='Inactive Gap/empty cohorts are N/A; zero-label-only cohorts retain MSE but not polarity metrics',
        limitation='Final readout source subtraction can be OOD; norms/cancellation are not emotion conflict')
    return output
