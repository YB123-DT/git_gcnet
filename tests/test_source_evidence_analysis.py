"""Frozen readout interventions: exact cohorts, signs, and nonlinear credit."""
import json

import numpy as np
import pytest

from experiments.osram_12_source_evidence_audit_20261019.analysis import analyze_arrays


def fixture_arrays():
    labels = np.array([0, 1, 0, 1])
    full = np.array([[3., 0., -2.], [0., 3., -2.], [0., 3., -2.], [3., 0., -2.]])
    reads = np.zeros((4, 4, 3, 512))
    reads[:, 0, 0, 0] = 2
    reads[:, 0, 1, 0] = -1
    availability = np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1], [1, 1, 1]])
    active = np.column_stack((np.ones(4, dtype=bool), ~availability.astype(bool)))
    trajectory = np.zeros((4, 3, 4))
    trajectory[1:, :, :] = [1., .5, 1., 1.]
    coalitions = np.repeat(full[:, None], 8, axis=1)
    # Arbitrary nonlinear table; full coalition remains the original logits.
    for mask in range(7):
        coalitions[:, mask, 0] += (mask % 3) ** 2 + 0.3
    return dict(labels=labels, logits_full=full,
                logits_removed=np.repeat(full[:, None, None], 12, axis=1).reshape(4, 4, 3, 3),
                logits_coalitions=coalitions, source_reads=reads, row_active=active,
                trajectory=trajectory, state_source_norms=np.ones((4, 3)),
                state_joint_norm=np.ones(4), availability=availability,
                conversation_ids=np.array(["a", "a", "b", "b"]),
                utterance_indices=np.array([0, 1, 0, 1]))


def test_metrics_explicit_classes_and_empty_inactive_slots():
    result = analyze_arrays(fixture_arrays(), classes=3)
    metrics = result["full_metrics"]
    assert metrics["ACC"] == pytest.approx(50)
    assert metrics["UA"] == pytest.approx(100 / 3)
    assert metrics["WF1"] == pytest.approx(50)
    assert metrics["class_support"] == [2, 2, 0]
    inactive = result["slots"]["Gap-A"]["A"]
    assert inactive["N"] == 0
    assert inactive["delta_CE"] is None
    assert inactive["corrections"] is None
    assert result["slots"]["Gap-T"]["A"]["N"] == 2
    json.dumps(result, allow_nan=False)


def test_removal_sign_and_corrections_harms_use_same_active_cohort():
    arrays = fixture_arrays()
    # Removing this slot breaks one full-correct prediction: helpful evidence.
    arrays["logits_removed"][0, 2, 0] = [0, 3, -2]
    result = analyze_arrays(arrays, 3)
    slot = result["slots"]["Gap-T"]["A"]
    assert slot["delta_CE"] > 0
    assert slot["delta_ACC"] == pytest.approx(-50)
    assert slot["corrections"] == 0
    assert slot["harms"] == 1
    assert slot["full_metrics"]["N"] == slot["removed_metrics"]["N"] == 2


def test_cancellation_excludes_zero_and_inactive_rows():
    result = analyze_arrays(fixture_arrays(), 3)
    assert result["cancellation"]["Base"]["N"] == 4
    assert result["cancellation"]["Base"]["mean"] == pytest.approx(2 / 3)
    assert result["cancellation"]["Gap-T"]["N"] == 0
    assert result["cancellation"]["Gap-T"]["mean"] is None


def test_nonlinear_three_player_shapley_efficiency_and_single_player_dummy():
    arrays = fixture_arrays()
    result = analyze_arrays(arrays, 3)
    shapley = result["shapley"]
    assert sum(shapley["mean"].values()) == pytest.approx(shapley["full_minus_zero_gain"])
    assert shapley["efficiency_max_abs_error"] < 1e-12
    # Only A changes the payoff; T and V must get exactly zero credit.
    for mask in range(8):
        arrays["logits_coalitions"][:, mask] = arrays["logits_full"] + (0 if mask & 1 else [1, 0, 0])
    result = analyze_arrays(arrays, 3)
    assert result["shapley"]["mean"]["T"] == pytest.approx(0, abs=1e-14)
    assert result["shapley"]["mean"]["V"] == pytest.approx(0, abs=1e-14)
    assert result["shapley"]["mean"]["A"] == pytest.approx(result["shapley"]["full_minus_zero_gain"])


def test_strata_preserve_current_availability_and_strict_history_support():
    result = analyze_arrays(fixture_arrays(), 3)
    assert result["strata"]["availability"]["AV"]["N"] == 2
    assert result["strata"]["availability"]["ATV"]["N"] == 2
    assert result["strata"]["availability"]["A"]["N"] == 0
    assert result["strata"]["history_ever"]["none"]["N"] == 1
    assert result["strata"]["history_ever"]["ATV"]["N"] == 3
    assert result["strata"]["source_history"]["T"]["coverage"]["never"]["N"] == 1
    assert result["strata"]["source_history"]["T"]["coverage"]["(0.25,0.5]"]["N"] == 3


def test_rejects_coalition_full_mismatch_and_nonfinite_arrays():
    arrays = fixture_arrays()
    arrays["logits_coalitions"][0, 7, 0] += 0.1
    with pytest.raises(ValueError, match="coalition"):
        analyze_arrays(arrays, 3)
    arrays = fixture_arrays()
    arrays["source_reads"][0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        analyze_arrays(arrays, 3)


def test_removing_harmful_evidence_has_negative_delta_ce_and_positive_correction():
    arrays = fixture_arrays()
    arrays["logits_removed"][2, 0, 1] = [3, 0, -2]
    item = analyze_arrays(arrays, 3)["slots"]["Base"]["T"]
    assert item["delta_CE"] < 0
    assert item["delta_ACC"] == pytest.approx(25)
    assert item["corrections"] == 1
    assert item["harms"] == 0


def test_explicit_noncontiguous_class_ids_and_extreme_logits_are_supported():
    arrays = fixture_arrays()
    arrays["labels"] = np.array([10, 20, 10, 20])
    arrays["logits_full"] *= 1000
    arrays["logits_removed"] *= 1000
    arrays["logits_coalitions"] *= 1000
    result = analyze_arrays(arrays, [10, 20, 30])
    assert result["full_metrics"]["ACC"] == 50
    assert np.isfinite(result["full_metrics"]["CE"])
    json.dumps(result, allow_nan=False)


def test_empty_input_remains_json_safe_and_all_groups_na():
    arrays = {name: value[:0] for name, value in fixture_arrays().items()}
    result = analyze_arrays(arrays, 3)
    assert result["N"] == 0
    assert result["full_metrics"]["ACC"] is None
    assert result["shapley"]["efficiency_max_abs_error"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("field,value", [("availability", [0, 0, 0]),
                                          ("row_active", [1, 1, 1, 1])])
def test_invalid_observation_or_inactive_gap_contract_is_rejected(field, value):
    arrays = fixture_arrays()
    arrays[field][0] = value
    with pytest.raises(ValueError):
        analyze_arrays(arrays, 3)
