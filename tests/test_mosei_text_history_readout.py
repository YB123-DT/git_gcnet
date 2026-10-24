import numpy as np


def test_cohorts_use_only_strict_past_and_never_cross_dialogues():
    from experiments.osram_mosei_text_history_readout_20261024.evaluate import cohort_masks

    av = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0],
                   [1, 0, 0], [1, 0, 0], [0, 0, 1]])
    groups = cohort_masks(av, np.array(['a', 'a', 'a', 'a', 'b', 'b', 'a']),
                          np.array([0, 1, 2, 3, 0, 1, 4]))
    np.testing.assert_array_equal(groups['T_missing_history_seen_T'],
                                  [False, False, False, True, False, False, True])
    np.testing.assert_array_equal(groups['T_missing_history_never_T'],
                                  [True, True, False, False, True, True, False])


def test_correction_counts_exclude_neutral_and_use_positive_strictly_above_zero():
    from experiments.osram_mosei_text_history_readout_20261024.evaluate import paired_vs

    labels = np.array([-1., 1., 0., -1.])
    old = np.array([-1., -1., 9., 1.])
    new = np.array([1., 1., -8., 0.])
    got = paired_vs(old, new, labels, np.ones(4, dtype=bool))
    assert got == dict(N=3, corrections=2, harms=1, both_correct=0, both_wrong=0)


def test_cohorts_reject_missing_or_reordered_history_indices():
    from experiments.osram_mosei_text_history_readout_20261024.evaluate import cohort_masks
    import pytest

    with pytest.raises(ValueError, match='contiguous'):
        cohort_masks(np.array([[1, 0, 0], [1, 0, 0]]),
                     np.array(['a', 'a']), np.array([0, 2]))
