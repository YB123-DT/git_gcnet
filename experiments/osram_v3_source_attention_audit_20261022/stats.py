"""Full-population frozen V3 attention summaries; no learned/causal inference.

``summarize`` consumes aligned utterance arrays.  Attention must already include
the architecture's role/source masks: inactive rows are zero, active heads sum
to one.  Invalid arrays raise rather than being silently renormalized.
"""
from collections.abc import Mapping

import numpy as np


ROLES = ('Base', 'Gap-A', 'Gap-T', 'Gap-V')
SOURCES = ('A', 'T', 'V')
PATTERNS = ('A', 'T', 'V', 'AT', 'AV', 'TV', 'ATV')
TOLERANCE = 2e-5


def _validate(arrays):
    required = ('attention', 'role_active', 'seen', 'availability', 'trajectory',
                'local', 'labels', 'prediction', 'conversation_ids', 'utterance_indices')
    missing = set(required) - set(arrays)
    if missing:
        raise ValueError(f'Missing arrays: {sorted(missing)}')
    a = {key: np.asarray(arrays[key]) for key in required}
    n = len(a['attention'])
    expected = {'attention': (n, 4, 4, 3), 'role_active': (n, 4),
                'seen': (n, 3), 'availability': (n, 3), 'trajectory': (n, 3, 4),
                'local': (n, 256), 'labels': (n,), 'conversation_ids': (n,),
                'utterance_indices': (n,)}
    for key, shape in expected.items():
        if a[key].shape != shape:
            raise ValueError(f'{key}: expected {shape}, received {a[key].shape}')
    if a['prediction'].ndim != 2 or a['prediction'].shape[0] != n or a['prediction'].shape[1] < 1:
        raise ValueError('prediction must have shape [N,1] or [N,C]')
    for key in required:
        if key != 'conversation_ids' and not np.isfinite(a[key]).all():
            raise ValueError(f'{key} contains non-finite values')
    for key in ('role_active', 'seen', 'availability'):
        if not np.isin(a[key], [0, 1]).all():
            raise ValueError(f'{key} is not binary')
        a[key] = a[key].astype(bool)
    if not np.array_equal(a['trajectory'][..., 3].astype(bool), a['seen']):
        raise ValueError('seen differs from trajectory ever_seen')
    if (a['role_active'][:, 1:] & a['availability']).any():
        raise ValueError('observed modality has active Gap')
    if (a['role_active'] & ~a['seen'].any(-1, keepdims=True)).any():
        raise ValueError('active role without historical source')
    attention = a['attention']
    if (attention < 0).any():
        raise ValueError('attention contains negative weights')
    allowed = a['role_active'][:, :, None, None] & a['seen'][:, None, None, :]
    if np.any(np.abs(np.where(allowed, 0, attention)) > TOLERANCE):
        raise ValueError('attention leaks through inactive role/unseen source')
    expected_sum = np.broadcast_to(a['role_active'][..., None], attention.shape[:-1])
    error = np.abs(attention.sum(-1) - expected_sum)
    if np.any(error > TOLERANCE):
        raise ValueError('active attention heads do not sum to one')
    return a, {'max_row_sum_error': float(error.max(initial=0)),
               'max_inactive_or_unseen_weight': float(np.abs(np.where(allowed, 0, attention)).max(initial=0)),
               'tolerance': TOLERANCE}


def _entropy(weights):
    return -np.sum(weights * np.log(np.maximum(weights, np.finfo(float).tiny)), axis=-1)


def _correlation(x, y):
    if len(x) < 3 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _row(a, selection, role, group):
    """Population statistics, preserving individual heads and unseen-slot zeros."""
    attention = a['attention'][selection, role].astype(np.float64)
    n = len(attention)
    row = {'role': ROLES[role], 'group': group, 'N': n,
           'weight_mean': None, 'weight_std': None,
           'per_head_weight_mean': None, 'per_head_weight_std': None,
           'normalized_entropy_head_average_mean': None,
           'normalized_entropy_per_head_mean': None,
           'mean_max_weight': None, 'per_head_mean_max_weight': None,
           'argmax_proportion': None, 'per_head_argmax_proportion': None,
           'selection_N': 0, 'entropy_normalization': 'log(number of seen sources); only counts 2/3',
           'observational_correlations': None}
    if not n:
        return row
    average = attention.mean(axis=1)
    row.update(weight_mean=average.mean(0).tolist(), weight_std=average.std(0).tolist(),
               per_head_weight_mean=attention.mean(0).tolist(),
               per_head_weight_std=attention.std(0).tolist())
    count = a['seen'][selection].sum(-1)
    eligible = count >= 2
    row['selection_N'] = int(eligible.sum())
    if eligible.any():
        norm = np.log(count[eligible])
        row['normalized_entropy_head_average_mean'] = float((_entropy(average[eligible]) / norm).mean())
        row['normalized_entropy_per_head_mean'] = float((_entropy(attention[eligible]) / norm[:, None]).mean())
        row['mean_max_weight'] = float(average[eligible].max(-1).mean())
        row['per_head_mean_max_weight'] = attention[eligible].max(-1).mean(0).tolist()
        # Split ties rather than giving uniform rows an artificial source-A bias.
        winners = np.isclose(average[eligible], average[eligible].max(-1, keepdims=True), atol=1e-12, rtol=0)
        row['argmax_proportion'] = (winners / winners.sum(-1, keepdims=True)).mean(0).tolist()
        head_winners = np.isclose(attention[eligible], attention[eligible].max(-1, keepdims=True), atol=1e-12, rtol=0)
        row['per_head_argmax_proportion'] = (head_winners / head_winners.sum(-1, keepdims=True)).mean(0).tolist()
    local_norm = np.linalg.norm(a['local'][selection], axis=-1)
    text_coverage = a['trajectory'][selection, 1, 1]
    row['observational_correlations'] = {
        'local_norm_vs_source_weight': [_correlation(local_norm, average[:, s]) for s in range(3)],
        'historical_T_coverage_vs_source_weight': [_correlation(text_coverage, average[:, s]) for s in range(3)],
        'warning': 'Descriptive associations, not proof of content conditioning or causal source contribution.',
    }
    return row


def summarize(arrays: Mapping[str, np.ndarray], dataset: str, rate, fold=None):
    """Return JSON-safe per-role/group and matched Base/Gap-T focus summaries.

    Expected arrays are documented in this module's tests. Rows use all aligned
    utterances for descriptive statistics, and only active roles for attention.
    Zero/one historical-source cases are counted separately; they are not evidence
    of learned source selection. Regression polarity excludes zero gold labels.
    """
    a, validation = _validate(arrays)
    n = len(a['attention'])
    seen_count = a['seen'].sum(-1)
    patterns = np.array([''.join(s for s, present in zip(SOURCES, avail) if present)
                         for avail in a['availability']], dtype=str)
    regression = a['prediction'].shape[-1] == 1
    if regression:
        labeled = a['labels'] != 0
        correct = (a['prediction'][:, 0] > 0) == (a['labels'] > 0)
    else:
        labeled = ((a['labels'] >= 0) & (a['labels'] < a['prediction'].shape[-1])
                   & (a['labels'] == np.floor(a['labels'])))
        correct = a['prediction'].argmax(-1) == a['labels']
    groups = [('all', np.ones(n, bool))]
    groups += [(f'source_count={c}', seen_count == c) for c in range(4)]
    groups += [(f'availability={pattern}', patterns == pattern) for pattern in PATTERNS]
    groups += [('seen_T=0', ~a['seen'][:, 1]), ('seen_T=1', a['seen'][:, 1]),
               ('correct', labeled & correct), ('error', labeled & ~correct)]
    coverage = a['trajectory'][:, 1, 1]
    groups += [('T_coverage=0', coverage == 0),
               ('T_coverage=(0,1/3]', (coverage > 0) & (coverage <= 1/3)),
               ('T_coverage=(1/3,2/3]', (coverage > 1/3) & (coverage <= 2/3)),
               ('T_coverage=(2/3,1]', (coverage > 2/3) & (coverage <= 1))]
    age = a['trajectory'][:, 1, 2]
    seen_text = a['seen'][:, 1]
    groups += [('T_age=never', ~seen_text), ('T_age=1', seen_text & (age == 1)),
               ('T_age=2-4', seen_text & (age >= 2) & (age <= 4)),
               ('T_age=5+', seen_text & (age >= 5))]
    rows = [_row(a, mask & a['role_active'][:, role], role, name)
            for role in range(4) for name, mask in groups]
    focus_mask = (~a['availability'][:, 1] & (seen_count == 3)
                  & a['role_active'][:, 0] & a['role_active'][:, 2])
    focus_rows = [_row(a, focus_mask, role, 'T_missing_all_three_sources_seen') for role in (0, 2)]
    focus = {'N': int(focus_mask.sum()), 'rows': focus_rows,
             'base_gap_T_weight_L1_mean': None, 'base_gap_T_argmax_agreement': None,
             'argmax_tie_convention': 'fractional uniform among tied maximizers'}
    if focus_mask.any():
        weights = a['attention'][focus_mask].mean(2)
        base, gap = weights[:, 0], weights[:, 2]
        focus['base_gap_T_weight_L1_mean'] = float(np.abs(base-gap).sum(-1).mean())
        base_max = np.isclose(base, base.max(-1, keepdims=True), atol=1e-12, rtol=0)
        gap_max = np.isclose(gap, gap.max(-1, keepdims=True), atol=1e-12, rtol=0)
        focus['base_gap_T_argmax_agreement'] = float(
            ((base_max / base_max.sum(-1, keepdims=True))
             * (gap_max / gap_max.sum(-1, keepdims=True))).sum(-1).mean())
    return {
        'metadata': {'dataset': str(dataset), 'rate': str(rate),
                     'fold': None if fold is None else str(fold), 'N': n,
                     'roles': list(ROLES), 'sources': list(SOURCES),
                     'std_convention': 'population; ddof=0',
                     'prediction_correctness': 'nonzero-label polarity at >0' if regression else 'argmax class'},
        'validation': validation,
        'source_count_utterances': {str(c): int((seen_count == c).sum()) for c in range(4)},
        'rows': rows, 'focus': focus,
        'limitations': [
            'Final source tokens already mix sources in Layer 2; attention is not original-source causal attribution.',
            'Counts zero/one are not learned source selection; entropy is reported only for counts two/three.',
            'Correct/error labels are used for offline stratification only.',
            'Groups overlap; do not sum group supports or treat fold/rate samples as independent evidence.',
        ],
    }
