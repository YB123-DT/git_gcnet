import importlib.util
from pathlib import Path
import pytest
import torch


@pytest.mark.parametrize('target', [0,1,2])
def test_fixed_sum_exact_endpoints_and_unchanged_ineligible(target):
    path=Path(__file__).resolve().parents[1]/'experiments/osram_cross_substitution_20260910/interpolate.py'
    assert path.exists(), 'interpolation not implemented'
    spec=importlib.util.spec_from_file_location('interpolate',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    torch.manual_seed(9)
    local=torch.randn(4,1,2);base=torch.randn(4,1,3);gap=torch.randn(4,1,3,3)
    a=torch.ones(4,1,3);a[0,0,target]=0;a[1,0,:2]=0;a[3,0,target]=0
    valid=torch.tensor([[True],[True],[True],[False]])
    originals=[x.clone() for x in (local,base,gap)]
    out=m.interpolated_inputs(local,base,gap,a,valid)
    assert len(out)==5
    normal=torch.cat([local,base,(gap*(1-a)[...,None]).flatten(-2)],-1)
    assert torch.equal(out['normal'],normal)
    for key,value in out.items():
        assert torch.equal(value[..., :2],local)
        assert torch.equal(value[1:],normal[1:])
        b=value[0,0,2:5];g=value[0,0,5:].reshape(3,3)
        assert torch.count_nonzero(g[[i for i in range(3) if i!=target]])==0
        torch.testing.assert_close(b+g[target],base[0,0]+gap[0,0,target])
    assert torch.equal(out['t_minus1'][0,0,2:5],gap[0,0,target])
    assert torch.equal(out['t_minus1'][0,0,5:].reshape(3,3)[target],base[0,0])
    assert torch.equal(out['t_zero'][0,0,2:5],out['t_zero'][0,0,5:].reshape(3,3)[target])
    assert all(torch.equal(x,y) for x,y in zip(originals,(local,base,gap)))
