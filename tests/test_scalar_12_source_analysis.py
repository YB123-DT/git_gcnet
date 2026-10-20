import importlib
import json

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score


MODULE = 'experiments.osram_scalar_12_source_audit_20261020'


def arrays(labels=(-1., 1., 0., 1.), predictions=(-.2, 0., 99., .7)):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    n = len(labels)
    availability = np.tile([1, 0, 1], (n, 1))
    trajectory = np.zeros((n, 3, 4))
    trajectory[:, :, 0] = 1
    trajectory[:, :, 1] = .5
    trajectory[:, :, 2] = 1
    trajectory[:, :, 3] = 1
    reads = np.zeros((n, 4, 3, 2))
    reads[:, 0] = 1
    reads[:, 2] = 1
    return dict(labels=labels, logits_full=predictions[:, None],
                logits_removed=np.broadcast_to(predictions[:, None, None, None], (n, 4, 3, 1)).copy(),
                logits_coalitions=np.broadcast_to(predictions[:, None, None], (n, 8, 1)).copy(),
                source_reads=reads, row_active=np.tile([1, 0, 1, 0], (n, 1)),
                trajectory=trajectory, state_source_norms=np.ones((n, 3)),
                state_joint_norm=np.ones(n), availability=availability,
                conversation_ids=np.array(['a'] * n), utterance_indices=np.arange(n))


def analysis():
    return importlib.import_module(MODULE + '.analysis')


def test_scalar_polarity_threshold_zero_filter_and_mse_all_labels():
    data = arrays()
    result = analysis().analyze_arrays(data, dataset='CMUMOSI')
    metric = result['full_metrics']
    eligible = np.abs(data['labels']) > 0
    expected = data['labels'][eligible] > 0
    predicted = data['logits_full'][eligible, 0] > 0
    assert metric['N'] == 4
    assert metric['N_polarity'] == 3
    assert metric['ACC'] == pytest.approx(100 * accuracy_score(expected, predicted))
    assert metric['WF1'] == pytest.approx(100 * f1_score(expected, predicted, average='weighted'))
    assert metric['MSE'] == pytest.approx(np.mean((data['labels'] - data['logits_full'][:, 0]) ** 2))
    assert 'UA' not in metric and 'CE' not in metric


def test_active_cohort_corrections_harms_and_empty_gap_na():
    data = arrays()
    data['logits_removed'][:, 2, 1, 0] = [.2, .2, -999., .0]
    result = analysis().analyze_arrays(data)
    slot = result['slots']['Gap-T']['T']
    assert (slot['N'], slot['N_polarity']) == (4, 3)
    assert (slot['corrections'], slot['harms'], slot['both_correct'], slot['both_wrong']) == (1, 2, 0, 0)
    assert slot['delta_ACC'] == pytest.approx(-100 / 3)
    inactive = result['slots']['Gap-A']['A']
    assert inactive['N'] == 0
    assert inactive['full_metrics']['ACC'] is None
    assert inactive['removed_metrics']['MSE'] is None
    assert inactive['corrections'] is None
    assert inactive['read_norm']['mean'] is None


def test_zero_label_only_cohort_has_mse_not_polarity_metrics():
    result = analysis().analyze_arrays(arrays(labels=(0.,), predictions=(2.,)))
    assert result['full_metrics'] == dict(N=1, N_polarity=0, ACC=None, WF1=None, MSE=4.)
    assert result['slots']['Base']['A']['delta_ACC'] is None
    assert result['slots']['Base']['A']['delta_MSE'] == 0.
    assert result['slots']['Base']['A']['corrections'] is None


def test_shapley_efficiency_uses_negative_mse():
    data = arrays()
    data['logits_coalitions'][:, :7, 0] = np.arange(7)[None, :] / 3
    result = analysis().analyze_arrays(data)
    losses = (data['labels'][:, None] - data['logits_coalitions'][:, :, 0]) ** 2
    expected_gain = np.mean(losses[:, 0] - losses[:, 7])
    assert sum(result['shapley']['mean'].values()) == pytest.approx(expected_gain)
    assert result['shapley']['full_minus_zero_gain'] == pytest.approx(expected_gain)
    assert result['shapley']['efficiency_max_abs_error'] < 1e-10


@pytest.mark.parametrize('change', ['shape', 'inactive', 'coalition'])
def test_invalid_scalar_contract_rejected(change):
    data = arrays()
    if change == 'shape':
        data['logits_full'] = np.ones((4, 7))
    elif change == 'inactive':
        data['row_active'][:, 1] = 1
    else:
        data['logits_coalitions'][:, 7] += 1
    with pytest.raises(ValueError):
        analysis().analyze_arrays(data)


def fixture_run(tmp_path):
    report = importlib.import_module(MODULE + '.report')
    summary = dict(status='complete', no_training=True, backbone_unchanged=True,
                   source_commit='test-sha', dataset='CMUMOSI',
                   anchor=dict(checkpoint_sha256='anchor-sha', checkpoint='protected.pt', seed=66, fold=1),
                   conditions={})
    stat = analysis().analyze_arrays(arrays())
    for group, names in [('random', report.RANDOM_RATES), ('persistent', report.PERSISTENT_PATTERNS)]:
        for name in names:
            key = group + '_' + name.replace('.', 'p')
            checks = dict(availability_sha256='mask-sha', source_read_max_abs_error=1e-7)
            condition = dict(source_commit='test-sha', anchor_sha256='anchor-sha',
                             group=group, condition=name, checks=checks, statistics=stat)
            filename = key + '.json'
            (tmp_path / filename).write_text(json.dumps(condition, allow_nan=False))
            summary['conditions'][key] = dict(path=filename, group=group, condition=name,
                                               full_metrics=stat['full_metrics'], checks=checks)
    (tmp_path / 'SUMMARY.json').write_text(json.dumps(summary))
    return report


def test_report_all_conditions_and_both_per_rate_metric_tables(tmp_path):
    report = fixture_run(tmp_path)
    paths = report.summarize_run(tmp_path)
    result = (tmp_path / 'RESULT.md').read_text()
    assert 'INTERNAL DIAGNOSTIC ONLY' in result
    assert 'random 0.0' in result and 'persistent ATV' in result
    assert 'ACC' in result and 'W-F1' in result
    assert 'N_polarity' in (tmp_path / 'SLOTS.csv').read_text()
    assert 'Gap-A | 0 | 0 | N/A' in result
    for filename in ('ACC_PER_RATE.md', 'WF1_PER_RATE.md', 'PERSISTENT.md'):
        assert (tmp_path / filename).exists()
    machine = json.loads((tmp_path / 'REPORT.json').read_text())
    assert len(machine['conditions']) == 15
    assert machine['aggregates']['random8']['ACC'] == pytest.approx(200 / 3)
    assert 'UA' not in machine['aggregates']['random8']
    assert paths['result'].endswith('RESULT.md')


def test_report_rejects_incomplete_or_anchor_mismatch(tmp_path):
    report = fixture_run(tmp_path)
    condition = tmp_path / 'random_0p0.json'
    data = json.loads(condition.read_text())
    data['anchor_sha256'] = 'different'
    condition.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        report.summarize_run(tmp_path)
