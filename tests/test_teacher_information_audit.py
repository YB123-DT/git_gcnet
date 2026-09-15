import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def test_probe_uses_training_only_scaling_and_fixed_ridge():
    from experiments.osram_supervised_teacher_20260914.information_audit import probe
    x = np.arange(30., dtype=float).reshape(10, 3)
    y = np.linspace(-2, 2, 10)
    v = x[:4] + 80
    z = np.array([-1., 0., 1., 2.])
    scaler = StandardScaler().fit(x)
    expected = Ridge(alpha=10., solver='cholesky').fit(scaler.transform(x), y).predict(scaler.transform(v))
    result, scores = probe(x, y, v, z)
    np.testing.assert_allclose(scores, expected)
    assert result['n_binary'] == 3
    assert result['dimension'] == 3


def test_validation_labels_cannot_change_probe_predictions():
    from experiments.osram_supervised_teacher_20260914.information_audit import probe
    rng = np.random.default_rng(2)
    x, v = rng.normal(size=(20, 5)), rng.normal(size=(8, 5))
    y, z = rng.normal(size=20), rng.normal(size=8)
    _, first = probe(x, y, v, z)
    _, second = probe(x, y, v, -z)
    np.testing.assert_array_equal(first, second)
