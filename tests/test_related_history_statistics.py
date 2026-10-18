import json
import math

import numpy as np
import pytest


def api():
    from experiments.osram_meld_related_history_diagnostic_20261018 import metrics

    return metrics


def test_perfect_and_reversed_discrimination():
    good = api().relation_metrics([0.1, 0.9, 0.2, 0.8], [0, 1, 0, 1])
    bad = api().relation_metrics([0.9, 0.1, 0.8, 0.2], [0, 1, 0, 1])
    assert good['N'] == 4 and good['positives'] == 2
    assert good['accuracy'] == good['balanced_accuracy'] == good['auroc'] == 1
    assert good['precision'] == good['recall'] == good['f1'] == good['auprc'] == 1
    assert bad['accuracy'] == bad['auroc'] == 0
    assert good['bce'] < bad['bce']
    json.dumps(good, allow_nan=False)


def test_calibration_hand_computed_and_endpoint_bins():
    result = api().relation_metrics([0, 0.2, 0.8, 1], [0, 1, 0, 1])
    assert result['ece'] == pytest.approx(0.4)
    assert result['brier'] == pytest.approx(0.32)
    assert result['calibration_bins'][2]['count'] == 1
    assert result['calibration_bins'][2]['positive_fraction'] == 1
    assert result['calibration_bins'][9]['count'] == 1
    assert result['probability']['fraction_le_0p1'] == 0.25
    assert result['probability']['fraction_ge_0p9'] == 0.25
    assert math.isfinite(result['bce'])


def test_constant_half_tie_and_prevalence():
    result = api().relation_metrics([0.5] * 4, [0, 0, 0, 1])
    assert result['prevalence'] == 0.25
    assert result['constant_half']['accuracy'] == 0.25
    assert result['constant_half']['bce'] == pytest.approx(math.log(2))
    assert result['constant_half']['brier'] == 0.25
    assert result['majority_accuracy'] == 0.75
    assert result['auroc'] == 0.5
    assert result['probability']['quantiles']['p50'] == 0.5


@pytest.mark.parametrize('targets', [[0, 0], [1, 1]])
def test_single_class_is_explicitly_undefined(targets):
    result = api().relation_metrics([0.2, 0.8], targets)
    assert result['auroc'] is None
    assert result['auprc'] is None
    assert result['balanced_accuracy'] is None
    json.dumps(result, allow_nan=False)


def test_empty_arrays_have_no_nan():
    result = api().relation_metrics([], [])
    assert result['N'] == 0
    assert result['accuracy'] is None and result['ece'] is None
    assert len(result['calibration_bins']) == 10
    json.dumps(result, allow_nan=False)
    pair = api().paired_comparison([], np.zeros((0, 7)), np.zeros((0, 7)))
    assert pair['N'] == 0 and pair['mean_delta_ce'] is None
    json.dumps(pair, allow_nan=False)


def test_weights_are_secondary_not_label_reweighting():
    result = api().relation_metrics([0.2, 0.8], [0, 0], weights=[1, 3])
    assert result['accuracy'] == 0.5
    assert result['weighted_metrics']['accuracy'] == 0.25
    assert result['weighted_metrics']['brier'] == pytest.approx(0.49)
    assert result['weighted_metrics']['ece'] == pytest.approx(0.65)
    assert result['weighted_metrics']['calibration_bins'][8]['weight_sum'] == 3
    json.dumps(result, allow_nan=False)


def test_paired_counts_and_ce_use_same_rows():
    labels = [0, 0, 0, 0]
    ref = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], float)
    new = np.array([[0, 1], [1, 0], [1, 0], [0, 1]], float)
    result = api().paired_comparison(labels, ref, new)
    for key in ['corrections', 'harms', 'both_correct', 'both_wrong']:
        assert result[key] == 1
    assert result['prediction_changes'] == result['class_disagreement_count'] == 2
    assert result['mean_delta_ce'] == pytest.approx(0)
    extreme = api().paired_comparison([1], [[1000, -1000]], [[-1000, 1000]])
    assert extreme['mean_delta_ce'] == pytest.approx(-2000)


@pytest.mark.parametrize('p,y,w', [
    ([0.2], [0, 1], None), ([1.1], [0], None), ([-0.1], [0], None),
    ([np.nan], [0], None), ([0.2], [2], None), ([0.2], [np.nan], None),
    ([0.2], [0], [-1]), ([0.2], [0], [0]), ([0.2], [0], [np.inf]),
    ([0.2], [0], [1, 2]), ([[0.2]], [0], None),
])
def test_reject_invalid_relation_data(p, y, w):
    with pytest.raises(ValueError):
        api().relation_metrics(p, y, w)


@pytest.mark.parametrize('labels,ref,new', [
    ([0], [[1, 2]], [[1, 2], [2, 3]]),
    ([2], [[1, 2]], [[1, 2]]),
    ([0], [[np.nan, 2]], [[1, 2]]),
    ([0.5], [[1, 2]], [[1, 2]]),
])
def test_reject_invalid_paired_data(labels, ref, new):
    with pytest.raises(ValueError):
        api().paired_comparison(labels, ref, new)
