import numpy as np
import pytest


def logits(classes):
    return np.eye(6)[classes]


def test_class_zero_is_a_real_emotion_and_ua_is_fixed_six_class():
    from experiments.osram_signature_order_audit_20261025.iemocap import paired_metrics
    result = paired_metrics(np.array([0, 0, 0, 1]), logits([0, 0, 0, 0]), logits([1, 1, 1, 1]))
    assert result['N'] == 4
    assert result['ACC'] == 75
    assert result['UA'] == pytest.approx(100 / 6)
    assert result['support'] == [3, 1, 0, 0, 0, 0]
    assert result['recall'] == [100, 0, 0, 0, 0, 0]
    assert result['corrections_vs_reference'] == 3
    assert result['harms_vs_reference'] == 1


def test_empty_cohorts_have_na_scores():
    from experiments.osram_signature_order_audit_20261025.iemocap import paired_metrics
    result = paired_metrics(np.empty(0), np.empty((0, 6)), np.empty((0, 6)))
    assert result['N'] == 0
    assert result['ACC'] is result['UA'] is result['WF1'] is None


@pytest.mark.parametrize('labels,prediction', [([6], [[0]*6]), ([.5], [[0]*6]), ([0], [[np.nan]*6]), ([0], [[1]])])
def test_reject_invalid_classification_records(labels, prediction):
    from experiments.osram_signature_order_audit_20261025.iemocap import paired_metrics
    with pytest.raises(ValueError):
        paired_metrics(np.asarray(labels), np.asarray(prediction), np.asarray(prediction))


def test_scope_has_only_five_folds_and_original_eight_rates():
    from experiments.osram_signature_order_audit_20261025.iemocap import validate_scope
    assert validate_scope([1, 5], ['0.0', '0.7']) == ([1, 5], ['0.0', '0.7'])
    for folds, rates in [([6], ['0.0']), ([True], ['0.0']), ([1, 1], ['0.0']), ([1], ['0.8']), ([], ['0.0'])]:
        with pytest.raises(ValueError):
            validate_scope(folds, rates)


def test_full_mean_excludes_incomplete_scope_and_is_not_size_weighted():
    from experiments.osram_signature_order_audit_20261025.iemocap import aggregate_records, RATES
    records = []
    for fold in range(1, 6):
        for rate in RATES:
            metrics = dict(ACC=float(fold), UA=float(fold + 5), WF1=float(fold + 10), N=fold*100)
            records.append(dict(fold=fold, rate=rate, groups={'all': {'normal': metrics}}))
    assert not any(r['scope'] == 'fivefold_eight_rate_mean' for r in aggregate_records(records[:-1]))
    full = next(r for r in aggregate_records(records) if r['scope'] == 'fivefold_eight_rate_mean')
    assert full['ACC'] == 3
    assert full['UA'] == 8
    assert full['cells'] == 40
    assert full['aggregation'] == 'equal fold/rate cell arithmetic mean; not pooled predictions'


def test_empty_cell_does_not_silently_become_zero_in_mean():
    from experiments.osram_signature_order_audit_20261025.iemocap import aggregate_records, RATES
    records = [dict(fold=1, rate=r, groups={'no_history': {'normal': dict(ACC=None, UA=None, WF1=None, N=0)}}) for r in RATES]
    mean = aggregate_records(records)[0]
    assert mean['ACC'] is None and mean['UA'] is None
    assert mean['nonempty_cells'] == 0


def test_report_keeps_acc_and_ua_without_mse(tmp_path):
    from experiments.osram_signature_order_audit_20261025.iemocap import write_report
    score = dict(N=2, ACC=50., UA=25., WF1=45., recall=[0]*6, support=[1,1,0,0,0,0],
                 corrections_vs_reference=1, harms_vs_reference=1)
    record = dict(fold=5, rate='0.7', groups={'all': {'normal': score}})
    result = write_report(tmp_path, [record], {'seed': 68})
    assert result['status'] == 'subset_complete'
    text = (tmp_path/'RESULT.md').read_text()
    assert 'ACC' in text and 'UA' in text and 'MSE' not in text
