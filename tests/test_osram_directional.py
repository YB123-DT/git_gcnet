import importlib.util
from pathlib import Path
import torch


def test_matched_directions_and_linear_finite_difference():
    p=Path(__file__).resolve().parents[1]/'experiments/osram_cross_substitution_20260910/directional.py'
    assert p.exists(), 'directional diagnostic missing'
    spec=importlib.util.spec_from_file_location('directional',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    d=torch.tensor([[1.,2.,0.,0.],[3.,1.,0.,0.],[1.,-1.,0.,0.]],dtype=torch.float64)
    dirs,donor=m.directions(d,123)
    assert torch.all(donor!=torch.arange(3))
    for pair in dirs.values():
        torch.testing.assert_close((pair[0].square().sum(-1)+pair[1].square().sum(-1)),torch.ones(3,dtype=d.dtype))
        assert torch.count_nonzero(pair[0][:,2:])==0 and torch.count_nonzero(pair[1][:,2:])==0
    for key in ('real','shuffle','random'):
        assert torch.equal(dirs[key][0],-dirs[key][1])
    assert torch.equal(dirs['symmetric'][0],dirs['symmetric'][1])
    assert torch.equal(dirs['random'][1],dirs['symmetric'][1])
    b=d.clone();g=d*2
    f=lambda b,g: b.sum(-1)+3*g.sum(-1)
    for pair in dirs.values():
        expected=(pair[0].sum(-1)+3*pair[1].sum(-1)).abs()
        torch.testing.assert_close(m.finite_difference(f,b,g,pair,1e-3),expected,atol=1e-10,rtol=1e-9)
