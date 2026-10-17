import importlib

import numpy as np
import pytest


def analysis():
    try:
        return importlib.import_module('experiments.osram_meld_information_availability_20261014.correctness_cross')
    except ModuleNotFoundError:
        pytest.fail('Offline correctness cross analysis is not implemented')


def test_exact_four_cells_and_net_accuracy_identity():
    y = np.array([0, 1, 2, 3])
    left = np.array([0, 0, 2, 0])
    right = np.array([0, 1, 0, 0])
    row = analysis().cross_table(y, left, right)
    assert [row[k] for k in ['both_correct', 'local_correct_other_wrong', 'local_wrong_other_correct', 'both_wrong']] == [1, 1, 1, 1]
    assert row['N'] == 4 and row['delta_ACC_pp'] == 0.


def test_local_is_filtered_by_boolean_inclusion_before_alignment():
    y = np.array([0, 1, 2, 3])
    local = dict(labels=y, predictions=np.array([0, 1, 0, 3]))
    other = dict(labels=np.array([0, 2, 3]), predictions=np.array([1, 2, 3]))
    labels, left, right = analysis().align_predictions(y, local, other, np.array([True, False, True, True]))
    np.testing.assert_array_equal(labels, [0, 2, 3])
    np.testing.assert_array_equal(left, [0, 0, 3])
    assert analysis().cross_table(labels, left, right)['N'] == 3


def test_label_order_or_nonboolean_inclusion_is_rejected():
    y = np.array([0, 1, 2])
    local = dict(labels=y, predictions=y)
    other = dict(labels=np.array([1, 0, 2]), predictions=y)
    with pytest.raises(ValueError, match='label|order'):
        analysis().align_predictions(y, local, other, np.ones(3, dtype=bool))
    with pytest.raises(ValueError, match='boolean'):
        analysis().align_predictions(y, local, local, np.ones(3, dtype=int))


def test_net_gain_uses_the_common_cohort_not_full_local_accuracy():
    row = analysis().cross_table(np.array([0, 1, 2]), np.array([0, 0, 0]), np.array([0, 1, 0]))
    assert row['local_wrong_other_correct'] == 1
    assert row['local_correct_other_wrong'] == 0
    assert row['delta_ACC_pp'] == pytest.approx(100/3)
