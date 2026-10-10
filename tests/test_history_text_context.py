import importlib
import numpy as np


def api():
    return importlib.import_module('experiments.osram_history_text_context_20261010.diagnostic')


def test_four_masks_keep_evidence_identity_and_only_change_past():
    original = np.ones((5, 3), dtype=np.int64)
    plan = api().select_plan(original, 4)
    masks = api().four_masks(original, 4, plan)
    assert plan['text_position'] == 3
    assert plan['other_position'] != 3
    assert masks['A+'][3, 1] == masks['B+'][3, 1] == 1
    assert masks['A-'][3, 1] == masks['B-'][3, 1] == 0
    for value in masks.values():
        assert np.array_equal(value[4:], original[4:])
        assert (value.sum(-1) >= 1).all()
    assert np.array_equal(masks['A+'][3], masks['B+'][3])
    assert np.count_nonzero(masks['A+'] - masks['A-']) == 1
    assert np.count_nonzero(masks['B+'] - masks['B-']) == 1
    assert np.count_nonzero(masks['A+'] - masks['B+']) == 1
    assert np.array_equal(original, np.ones((5, 3)))


def test_no_legal_other_history_is_skipped():
    mask = np.array([[1, 1, 0], [0, 1, 0]])
    assert api().select_plan(mask, 1) is None
    assert api().select_plan(mask, 0) is None


def test_selection_is_deterministic_and_text_only_is_not_deleted():
    mask = np.array([[1, 1, 1], [1, 0, 1], [0, 1, 0], [1, 1, 1]])
    plan = api().select_plan(mask, 3)
    assert plan == api().select_plan(mask.copy(), 3)
    assert plan['text_position'] == 0
    assert plan['other_position'] == 1


def test_contribution_sign_and_prediction_threshold():
    result = api().contributions(1., {'A+': .5, 'A-': -.5, 'B+': -.5, 'B-': .5})
    assert result['delta_A'] == 2.
    assert result['delta_B'] == -2.
    assert result['interaction'] == -4.
    assert result['A_rescue'] and result['B_harm']
