"""Read-only full-dataset V3 attention statistics, not causal attribution."""
import json

import numpy as np
import pytest


def summarize(*args, **kwargs):
    from experiments.osram_v3_source_attention_audit_20261022.stats import summarize as impl
    return impl(*args, **kwargs)


def sample():
    n = 6
    seen = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0],
                     [1, 1, 1], [1, 1, 1], [1, 1, 1]], bool)
    availability = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0],
                             [1, 0, 0], [0, 0, 1], [1, 1, 1]], bool)
    active = np.concatenate([np.ones((n, 1), bool), ~availability], -1)
    active &= seen.any(-1, keepdims=True)
    attention = np.zeros((n, 4, 4, 3), dtype=np.float64)
    for i in range(n):
        if seen[i].any():
            attention[i] = active[i, :, None, None] * (seen[i] / seen[i].sum())
    # Individual heads selective, head-average weights exactly uniform.
    attention[3, 0] = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1/3, 1/3, 1/3]]
    trajectory = np.zeros((n, 3, 4))
    trajectory[..., 0] = seen * 2
    trajectory[..., 1] = seen * .5
    trajectory[..., 2] = seen * 2
    trajectory[..., 3] = seen
    return dict(attention=attention, role_active=active, seen=seen,
                availability=availability, trajectory=trajectory,
                local=np.arange(n*256).reshape(n, 256) / 1000,
                labels=np.array([1, -1, 0, 1, -1, 1]),
                prediction=np.array([[1], [1], [1], [1], [-1], [1]]),
                conversation_ids=np.array(['a'] * n), utterance_indices=np.arange(n))


def row(result, role, group):
    return next(x for x in result['rows'] if x['role'] == role and x['group'] == group)


def test_reports_head_average_and_per_head_entropy_separately():
    result = summarize(sample(), 'MOSI', .7, None)
    focus = result['focus']
    assert focus['N'] == 2
    base = focus['rows'][0]
    assert base['role'] == 'Base'
    assert len(base['per_head_weight_mean']) == 4
    assert base['normalized_entropy_head_average_mean'] == pytest.approx(1.)
    assert base['normalized_entropy_per_head_mean'] == pytest.approx(.625)
    assert base['weight_mean'] == pytest.approx([1/3] * 3)
    assert base['normalized_entropy_head_average_mean'] > base['normalized_entropy_per_head_mean']
    json.dumps(result, allow_nan=False)


def test_count_zero_one_and_neutral_do_not_fake_selection_or_correctness():
    result = summarize(sample(), 'CMUMOSI', .2, 0)
    assert result['source_count_utterances'] == {'0': 1, '1': 1, '2': 1, '3': 3}
    assert row(result, 'Base', 'source_count=1')['normalized_entropy_head_average_mean'] is None
    assert row(result, 'Base', 'source_count=1')['mean_max_weight'] is None
    assert row(result, 'Base', 'source_count=1')['argmax_proportion'] is None
    assert row(result, 'Base', 'correct')['N'] == 3
    assert row(result, 'Base', 'error')['N'] == 1
    assert row(result, 'Gap-T', 'availability=AV')['N'] == 0
    assert row(result, 'Gap-T', 'source_count=2')['N'] == 1


def test_classification_argmax_and_label_filter():
    arrays = sample()
    arrays['labels'] = np.array([0, 1, -1, 1, 0, 1])
    arrays['prediction'] = np.array([[2, 1], [2, 1], [2, 1], [1, 2], [2, 1], [1, 2]])
    result = summarize(arrays, 'IEMOCAP6', .3, 5)
    assert row(result, 'Base', 'correct')['N'] == 3
    assert row(result, 'Base', 'error')['N'] == 1


@pytest.mark.parametrize('problem', ['inactive', 'unseen', 'sum', 'nan', 'negative'])
def test_rejects_invalid_attention_instead_of_normalizing_it(problem):
    arrays = sample()
    if problem == 'inactive':
        arrays['attention'][0, 0, 0, 0] = 1
    elif problem == 'unseen':
        arrays['attention'][1, 0, 0] = [0, 1, 0]
    elif problem == 'sum':
        arrays['attention'][3, 0, 0] *= .9
    elif problem == 'nan':
        arrays['attention'][3, 0, 0, 0] = np.nan
    else:
        arrays['attention'][3, 0, 0] = [-1, 1, 1]
    with pytest.raises(ValueError):
        summarize(arrays, 'MOSI', .7, None)


def test_empty_input_is_json_safe():
    arrays = {k: v[:0] for k, v in sample().items()}
    result = summarize(arrays, 'MOSI', .7, None)
    assert result['focus']['N'] == 0
    assert row(result, 'Base', 'all')['weight_mean'] is None
    json.dumps(result, allow_nan=False)
