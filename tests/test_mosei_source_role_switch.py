import importlib

import numpy as np
import pytest


def test_text_groups_are_disjoint_and_use_strict_past_not_current():
    from experiments.osram_mosei_source_role_switch_20261021.analysis import text_groups
    availability = np.array([[1, 1, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0]])
    trajectory = np.zeros((4, 3, 4))
    trajectory[1, 1, 3] = 1
    groups = text_groups(availability, trajectory)
    np.testing.assert_array_equal(groups['T_visible'], [1, 0, 0, 1])
    np.testing.assert_array_equal(groups['T_missing_seen'], [0, 1, 0, 0])
    np.testing.assert_array_equal(groups['T_missing_unseen'], [0, 0, 1, 0])
    np.testing.assert_array_equal(sum(groups.values()), np.ones(4))


def test_gamma_distribution_ignores_inactive_slots_and_reports_variance():
    from experiments.osram_mosei_source_role_switch_20261021.analysis import distribution
    result = distribution(np.array([.5, 1., 1.5]))
    assert result['mean'] == 1. and result['variance'] == pytest.approx(1 / 6)
    assert result['min'] == .5 and result['max'] == 1.5
    assert result['within_0p01_of_one_fraction'] == pytest.approx(1 / 3)
    assert distribution(np.array([]))['mean'] is None


def test_analysis_counts_neutral_to_learned_corrections_and_harms():
    pytest.importorskip('sklearn')
    from experiments.osram_mosei_source_role_switch_20261021.analysis import analyze
    n = 4
    active = np.zeros((n, 4, 3), bool)
    active[:, 0, 0] = True
    data = dict(labels=np.array([1., -1., 1., 0.]),
        prediction_learned=np.array([1., 1., 1., .2]),
        prediction_neutral=np.array([-1., -1., 1., .1]),
        availability=np.array([[1,1,1],[1,0,0],[1,0,0],[1,1,1]]),
        trajectory=np.zeros((n,3,4)), active=active, gamma=np.where(active,1.2,0.),
        read_relative_change=np.zeros((n,4)))
    data['trajectory'][1,1,3] = 1
    result = analyze(data)
    assert result['groups']['all']['corrections'] == 1
    assert result['groups']['all']['harms'] == 1
    assert result['groups']['all']['N_polarity'] == 3
    assert result['gamma']['all']['N'] == 4  # not48 incl padding/inactive
    assert result['groups']['T_missing_unseen']['N'] == 1
    assert result['groups']['T_missing_unseen']['gain_WF1_pp'] == 0.
    data['gamma'][0,3,1] = .5
    with pytest.raises(ValueError): analyze(data)


def test_anchor_requires_original_completed_direct_mosei66():
    from experiments.osram_mosei_source_role_switch_20261021.run import validate_anchor
    with pytest.raises(ValueError): validate_anchor({'status':'running'})
