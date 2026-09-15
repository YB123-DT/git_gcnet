import numpy as np


def test_label_free_alignment_matches_train_only_multioutput_ridge():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from experiments.osram_supervised_teacher_20260914.text_alignment_audit import fit_alignment
    rng = np.random.default_rng(11)
    p, t, v = rng.normal(size=(40, 5)), rng.normal(size=(40, 7)), rng.normal(size=(9, 5))
    expected = make_pipeline(StandardScaler(), Ridge(alpha=10., solver='cholesky')).fit(p, t)
    actual = fit_alignment(p, t)
    np.testing.assert_allclose(actual.predict(v), expected.predict(v))


def test_alignment_does_not_modify_latent_arrays():
    from experiments.osram_supervised_teacher_20260914.text_alignment_audit import fit_alignment
    rng = np.random.default_rng(17)
    p, t = rng.normal(size=(30, 5)), rng.normal(size=(30, 7))
    before_p, before_t = p.copy(), t.copy()
    fit_alignment(p, t).predict(p)
    np.testing.assert_array_equal(p, before_p)
    np.testing.assert_array_equal(t, before_t)
