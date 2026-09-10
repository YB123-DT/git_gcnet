import importlib.util
from pathlib import Path
import numpy as np


def test_magnitude_centering_and_zero_semantics():
    path=Path(__file__).resolve().parents[1]/'experiments/osram_cross_substitution_20260910/magnitude.py'
    assert path.exists(), 'magnitude audit missing'
    spec=importlib.util.spec_from_file_location('magnitude',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    b=np.array([[1.,0.],[0.,2.],[0.,0.]])
    v=m.measure(b,b)
    np.testing.assert_allclose(v['difference_norm'],0)
    np.testing.assert_allclose(v['cosine'][:2],1)
    assert np.isnan(v['cosine'][2])
    v=m.measure(b,-b)
    np.testing.assert_allclose(v['sum_norm'],0)
    np.testing.assert_allclose(v['relative_difference'][:2],1)
    np.testing.assert_allclose(v['cosine'][:2],-1)
    v=m.measure(b+np.array([10.,20.]),b+np.array([-20.,10.]))
    np.testing.assert_allclose(v['centered_cosine'],1)
    assert not np.allclose(v['cosine'],1)
    assert np.isnan(m.measure(b[:1],b[:1])['centered_cosine']).all()
