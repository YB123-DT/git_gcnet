import copy
import json

import numpy as np
import pytest


def arrays():
    labels = np.array([-1., 1., -1., 1., 0.])
    learned = np.array([-1., 1., 1., -1., .3])
    active = np.zeros((5, 4, 3), dtype=bool)
    active[:2, 0, 0] = True
    active[2:4, 2, 1] = True
    reset = np.broadcast_to(learned[:, None, None], (5, 4, 3)).copy()
    reset[0, 0, 0] = 1.
    reset[2, 2, 1] = -1.
    availability = np.ones((5, 3))
    availability[2:4, 1] = 0
    return dict(labels=labels, availability=availability, active=active,
                gamma=np.where(active, 1.2, 0.), prediction_learned=learned,
                prediction_neutral=np.array([1., 1., -1., -1., .2]),
                prediction_reset=reset, prediction_flat=labels.copy())


def test_slot_gain_is_learned_minus_reset_with_reset_to_learned_counts():
    from experiments.osram_mosi_source_role_neutral_audit_20261020.analysis import analyze_arrays
    result = analyze_arrays(arrays())
    base_a = result['slots']['Base/A']
    assert base_a['active_cohort']['N'] == 2
    assert base_a['active_cohort']['corrections'] == 1
    assert base_a['active_cohort']['harms'] == 0
    assert base_a['full_test']['gain_WF1_pp'] > 0
    gap_t = result['slots']['Gap-T/T']
    assert gap_t['active_cohort']['harms'] == 1
    assert gap_t['full_test']['gain_WF1_pp'] < 0
    assert result['models']['learned']['N_polarity'] == 4
    assert result['models']['learned']['MSE'] == pytest.approx(8.09 / 5)
    assert base_a['gamma']['mean'] == pytest.approx(1.2)


def test_absent_slot_is_na_in_active_but_zero_full_test_gain():
    from experiments.osram_mosi_source_role_neutral_audit_20261020.analysis import analyze_arrays
    result = analyze_arrays(arrays())
    slot = result['slots']['Gap-V/V']
    assert slot['active_cohort']['N'] == 0
    assert slot['active_cohort']['gain_WF1_pp'] is None
    assert slot['full_test']['gain_WF1_pp'] == 0
    assert slot['gamma']['mean'] is None
    json.dumps(result, allow_nan=False)


def test_mainpath_plus_direct_equals_total_metric_difference():
    from experiments.osram_mosi_source_role_neutral_audit_20261020.analysis import analyze_arrays
    result = analyze_arrays(arrays())
    for metric in ('WF1_pp', 'ACC_pp', 'MSE'):
        differences = result['comparisons']
        assert differences['total'][metric] == pytest.approx(
            differences['mainpath'][metric] + differences['direct'][metric])


@pytest.mark.parametrize('mutation', ['shape', 'nonfinite', 'inactive_change',
                                     'observed_gap', 'inactive_gamma', 'negative_gamma'])
def test_malformed_or_unpaired_arrays_fail_closed(mutation):
    from experiments.osram_mosi_source_role_neutral_audit_20261020.analysis import analyze_arrays
    data = arrays()
    if mutation == 'shape':
        data['prediction_reset'] = data['prediction_reset'][:, :3]
    elif mutation == 'nonfinite':
        data['prediction_learned'][0] = np.nan
    elif mutation == 'inactive_change':
        data['prediction_reset'][0, 3, 2] += 1
    elif mutation == 'observed_gap':
        data['availability'][2:4, 1] = 1
    elif mutation == 'inactive_gamma':
        data['gamma'][0, 3, 2] = .4
    elif mutation == 'negative_gamma':
        data['gamma'][0, 0, 0] = -.4
    with pytest.raises(ValueError):
        analyze_arrays(data)


def test_report_requires_all_24_and_averages_rates_before_seeds(tmp_path):
    from experiments.osram_mosi_source_role_neutral_audit_20261020.analysis import analyze_arrays
    from experiments.osram_mosi_source_role_neutral_audit_20261020.report import summarize_run
    conditions = {}
    for seed in (66, 67, 68):
        for index in range(8):
            key = f'{seed}/{index / 10:.1f}'
            record = dict(seed=seed, rate=f'{index / 10:.1f}',
                          statistics=analyze_arrays(arrays()), checks={'same_scan': True})
            # Distinct per-rate values expose accidental pooling/seed averaging.
            record['statistics']['slots']['Base/A']['full_test']['gain_WF1_pp'] = index + seed - 66
            name = f'condition_{seed}_{index}.json'
            (tmp_path / name).write_text(json.dumps(record))
            conditions[key] = dict(seed=seed, rate=f'{index / 10:.1f}', path=name)
    summary = dict(conditions=conditions)
    result = summarize_run(tmp_path, summary)
    assert result['label'] == 'INTERNAL DIAGNOSTIC ONLY'
    assert len(result['per_seed']) == 3
    assert len(result['slots']) == 12
    base = result['slots']['Base/A']
    assert base['per_seed_eight_rate_full_test_gain_WF1_pp']['66'] == 3.5
    assert base['across_seed'] == dict(mean=4.5, std=1.)
    for name in ('RESULT.md', 'PER_RATE.md', 'SLOTS.csv', 'METRICS.csv', 'REPORT.json'):
        assert (tmp_path / name).is_file()
    text = (tmp_path / 'RESULT.md').read_text()
    assert 'gamma=1' in text and 'gamma=0' in text
    missing = copy.deepcopy(summary)
    missing['conditions'].pop(next(iter(conditions)))
    with pytest.raises(ValueError):
        summarize_run(tmp_path, missing)
