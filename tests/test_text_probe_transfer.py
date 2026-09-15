import numpy as np


def test_same_text_latent_has_identical_probe_transfer():
    from experiments.osram_supervised_teacher_20260914.text_transfer_audit import fit_text_probe, transfer
    rng = np.random.default_rng(7)
    x, v = rng.normal(size=(40, 6)), rng.normal(size=(12, 6))
    y, z = x[:, 0], v[:, 0]
    probe = fit_text_probe(x, y)
    out = transfer(probe, v, v.copy(), z)
    assert out['real_text'] == out['predicted_text']
    assert out['score_mae_to_real'] == 0


def test_transfer_never_refits_on_validation_predictions():
    from experiments.osram_supervised_teacher_20260914.text_transfer_audit import fit_text_probe, transfer
    rng = np.random.default_rng(4)
    x, v = rng.normal(size=(40, 6)), rng.normal(size=(12, 6))
    probe = fit_text_probe(x, x[:, 0])
    before = probe.predict(v).copy()
    transfer(probe, v, np.zeros_like(v), -v[:, 0])
    np.testing.assert_array_equal(before, probe.predict(v))
