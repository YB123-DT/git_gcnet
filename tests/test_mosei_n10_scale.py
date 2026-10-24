import numpy as np
import pytest


def inputs():
    raw = tuple(np.full((2, 1, 3), v, dtype=np.float32) for v in (1., 2., 3.))
    corrected = tuple(x + 2 for x in raw)
    return raw, corrected


def test_exact_endpoints_return_raw_or_corrected_without_arithmetic():
    from experiments.osram_mosei_n10_scale_20261024.run import scaled_inputs
    raw, corrected = inputs()
    assert all(x is y for x, y in zip(scaled_inputs(raw, corrected, 0.), raw))
    assert all(x is y for x, y in zip(scaled_inputs(raw, corrected, 1.), corrected))


def test_half_scales_all_three_inputs_without_mutating_them():
    from experiments.osram_mosei_n10_scale_20261024.run import scaled_inputs
    raw, corrected = inputs()
    for i, result in enumerate(scaled_inputs(raw, corrected, .5)):
        np.testing.assert_array_equal(result, raw[i]+1)
        np.testing.assert_array_equal(raw[i], np.full_like(raw[i], i+1))
        np.testing.assert_array_equal(corrected[i], np.full_like(corrected[i], i+3))


def test_scaling_cannot_expand_the_authorized_grid_or_change_shapes():
    from experiments.osram_mosei_n10_scale_20261024.run import scaled_inputs
    raw, corrected = inputs()
    for scale in (.25, 1.5, np.nan):
        with pytest.raises(ValueError):
            scaled_inputs(raw, corrected, scale)
    with pytest.raises(ValueError):
        scaled_inputs(raw, (np.ones(2), *corrected[1:]), .5)


def test_paired_counts_use_nonzero_gold_and_predicted_positive_only_above_zero():
    from experiments.osram_mosei_n10_scale_20261024.run import paired_counts
    y = np.array([1., 1., -1., -1., 0.])
    old = np.array([0., 1., 0., 1., 1.])
    new = np.array([1., 0., 1., 0., 0.])
    assert paired_counts(y, old, new) == dict(N=4, corrections=2, harms=2,
        both_correct=0, both_wrong=0, N_positive=2, N_negative=2)
    assert paired_counts(y, old, new, np.array([1,1,0,0,0],bool))['N_negative'] == 0


def test_aggregate_requires_all_eight_rates_and_equal_rate_weight():
    from experiments.osram_mosei_n10_scale_20261024.run import aggregate
    rates = {f'{i/10:.1f}': dict(ACC=float(i), WF1=2.*i, MSE=.1*i) for i in range(8)}
    result = aggregate(rates)
    assert result['mean']['WF1'] == 7.
    assert result['high']['ACC'] == 6.
    with pytest.raises(ValueError):
        aggregate({k:v for k,v in rates.items() if k!='0.7'})


def test_checkpoint_identity_normalizes_json_sequences_but_not_configuration_values():
    from experiments.osram_mosei_n10_scale_20261024.run import validate_checkpoint_identity
    state = dict(epoch=12, identity=dict(config=dict(train_missing_rates=(0.,.1), osram_write_step=.6)))
    expected = dict(train_missing_rates=[0.,.1], osram_write_step=.6)
    validate_checkpoint_identity(state, dict(epoch=12), expected)
    with pytest.raises(ValueError):
        validate_checkpoint_identity(state, dict(epoch=13), expected)
    with pytest.raises(ValueError):
        validate_checkpoint_identity(state, dict(epoch=12), dict(expected,osram_write_step=.4))
