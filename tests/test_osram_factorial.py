import importlib.util
from pathlib import Path
import numpy as np


def test_effects_and_exact_reconstruction():
    path=Path(__file__).resolve().parents[1]/'experiments/osram_cross_substitution_20260910/factorial.py'
    assert path.exists(), 'factorial analysis not implemented'
    spec=importlib.util.spec_from_file_location('factorial',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    # grand=10; content effect=4; slot effect=6; interaction=0 or 8
    bb=np.array([9.,7.]);bg=np.array([15.,17.])
    gb=np.array([5.,7.]);gg=np.array([11.,9.])
    out=module.effects(bb,bg,gb,gg)
    np.testing.assert_allclose(out['content'],4)
    np.testing.assert_allclose(out['slot'],6)
    np.testing.assert_allclose(out['interaction'],[0,8])
    for observed,c,s in ((bb,1,-1),(bg,1,1),(gb,-1,-1),(gg,-1,1)):
        recovered=out['grand']+c*out['content']/2+s*out['slot']/2+c*s*out['interaction']/4
        np.testing.assert_allclose(recovered,observed)
