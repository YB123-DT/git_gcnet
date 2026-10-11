import pytest
import torch
from gcnet_missing_m3.meaningful_new40_registry import NESTED_SWEEP
from gcnet_missing_m3.nested_sweep import build

def test_config_changes_only_method():
    pytest.importorskip('torch_geometric')
    import json
    from pathlib import Path
    from experiments.osram_core20_20261005.run import candidate_config
    ref=json.loads((Path(__file__).resolve().parents[1]/'experiments/osram_core20_20261005/REFERENCE_CONFIG.json').read_text())
    for suffix in ('mlp256','mlp512','dim128','dim256'):
        cfg,delta=candidate_config(ref,'nested_direct_random_'+suffix)
        assert set(delta)=={'osram_meaningful_block'}
        assert cfg.epochs==100 and cfg.osram_output_dim==1600

@pytest.mark.parametrize('suffix,dim,hidden', [('mlp256',64,256),('mlp512',64,512),('dim128',128,128),('dim256',256,256)])
def test_direct_capacity(suffix,dim,hidden):
    name='nested_direct_random_'+suffix
    assert name in NESTED_SWEEP
    model=build(name)
    assert model.residual is False
    assert model.core.layers[0][0].out_features==hidden
    assert model.local_decoder.in_features==dim
    assert model.local_decoder.weight.count_nonzero()
    active=torch.tensor([[True,False,True,False]])
    local=torch.randn(1,256)
    memory=torch.randn(1,4,512)
    out=model(local,memory,active,torch.tensor([[1.,0.,1.]]))
    assert out[0].shape==local.shape and out[1].shape==memory.shape
    assert not out[1][:,~active[0]].count_nonzero()
    loss=out[0].square().mean()+out[1].square().mean()
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.core.layers[0][0].weight.grad.abs().sum()>0
    with torch.no_grad():
        model.local_decoder.weight.zero_();model.local_decoder.bias.zero_()
        for d in model.memory_decoders: d.weight.zero_();d.bias.zero_()
    zero=model(local,memory,active,torch.tensor([[1.,0.,1.]]))
    assert not zero[0].count_nonzero() and not zero[1].count_nonzero()
