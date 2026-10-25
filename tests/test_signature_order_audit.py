import numpy as np
import pytest


def test_metrics_keep_gold_zero_for_mse_but_not_polarity():
    from experiments.osram_signature_order_audit_20261025.audit import paired_metrics
    y = np.array([-1., 0., 1.])
    reference = np.array([-1., 0., -1.])
    candidate = np.array([-1., 2., 1.])
    result = paired_metrics(y, candidate, reference)
    assert result['N'] == 3 and result['N_polarity'] == 2
    assert result['ACC'] == 100.
    assert result['MSE'] == pytest.approx(4/3)
    assert result['corrections_vs_reference'] == 1
    assert result['harms_vs_reference'] == 0


def test_empty_stratum_is_na_not_a_zero_score():
    from experiments.osram_signature_order_audit_20261025.audit import paired_metrics
    result = paired_metrics(np.array([]), np.array([]), np.array([]))
    assert result['N'] == 0
    assert result['ACC'] is None and result['MSE'] is None


def test_conversation_bootstrap_is_deterministic_and_pairs_samples():
    from experiments.osram_signature_order_audit_20261025.audit import mse_bootstrap
    ids = np.array(['a', 'a', 'b', 'b'])
    y = np.zeros(4)
    normal = np.ones(4)
    control = np.ones(4)*2
    result = mse_bootstrap(y, control, normal, ids, repeats=100)
    assert result == mse_bootstrap(y, control, normal, ids, repeats=100)
    assert result['lower'] == result['upper'] == 3.
    assert result['unit'] == 'conversation'


def test_scope_excludes_new_training_and_missing_rates():
    from experiments.osram_signature_order_audit_20261025.audit import validate_rates
    assert validate_rates(['0.0', '0.7']) == ['0.0', '0.7']
    with pytest.raises(ValueError):
        validate_rates(['0.8'])
    with pytest.raises(ValueError):
        validate_rates(['0.7', '0.7'])
