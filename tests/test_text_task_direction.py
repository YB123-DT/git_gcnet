import numpy as np


def test_raw_probe_direction_accounts_for_standardization_and_intercept():
    from experiments.osram_supervised_teacher_20260914.text_transfer_audit import fit_text_probe
    from experiments.osram_supervised_teacher_20260914.task_direction_audit import raw_direction
    rng = np.random.default_rng(9)
    x = rng.normal(size=(50, 6)) * np.arange(1, 7) + 12
    y = x[:, 0] - x[:, 3] + 4
    probe = fit_text_probe(x, y)
    w, b = raw_direction(probe)
    np.testing.assert_allclose(x @ w + b, probe.predict(x), atol=1e-10)


def test_direction_metrics_detect_perfect_scalar_prediction():
    from experiments.osram_supervised_teacher_20260914.task_direction_audit import direction_metrics
    r = np.array([-2., -1., 1., 3.])
    result = direction_metrics(r, r.copy(), r + .2, .2, 66)
    assert result['correlation'] > .999999
    assert result['mae'] == 0
    assert result['teacher_sign_agreement'] == 1
    assert result['sentiment']['weighted_f1'] == 1
