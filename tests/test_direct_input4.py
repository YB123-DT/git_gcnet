import pytest
import torch
from gcnet_missing_m3.meaningful_input import INPUT_METHODS, MeaningfulInputAdapter

METHODS = ('m03_gatv2_direct','m05_pna_direct','cwn_cellular_direct','perceiver_io_direct')


@pytest.mark.parametrize('method',METHODS)
def test_direct_contract_no_bypass_masks_and_core_update(method):
    assert method in INPUT_METHODS
    torch.set_num_threads(1)
    torch.manual_seed(66)
    block=MeaningfulInputAdapter(256,1024,1600,method,8,64)
    local,base,gap=torch.randn(3,1,256),torch.randn(3,1,1024),torch.randn(3,1,3,1024)
    base[...,512:]=0.; gap[...,512:]=0.
    av=torch.tensor([[[1.,0.,1.]]]*3); av[2]=0.
    mask=torch.tensor([[1.,1.,0.]])
    dirty=torch.where((av<.5)[...,None],gap,float('nan'))
    optimizer=torch.optim.Adam(block.parameters(),lr=.001)
    initial={k:v.detach().clone() for k,v in block.named_parameters()}
    out=block(local,base,dirty,av,mask)
    assert torch.equal(out[0][0],local[0]) and torch.equal(out[1][0],base[0])
    assert all(torch.isfinite(x).all() and not x[2].count_nonzero() for x in out)
    assert not out[2][1,0,[0,2]].count_nonzero()
    assert not out[1][...,512:].count_nonzero() and not out[2][...,512:].count_nonzero()
    sum(x[1].square().mean() for x in out).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in block.parameters())
    optimizer.step()
    changed=[k for k,p in block.named_parameters() if not torch.equal(initial[k],p)]
    assert any(not any(s in k for s in ('decoder','bridge','local_decoder','memory_decoders')) for k in changed)
    # Zero every external decoder: direct valid-history inputs must become ZERO,
    # not original evidence+zero. First turn bypass must remain unchanged.
    with torch.no_grad():
        for name,module in block.core.named_modules():
            if isinstance(module,torch.nn.Linear) and ('decoder' in name):
                module.weight.zero_()
                if module.bias is not None: module.bias.zero_()
    out=block(local,base,dirty,av,mask)
    assert all(not x[1].count_nonzero() for x in out)
    assert torch.equal(out[0][0],local[0])


def test_original_graph_pooling_uses_exact_same_node_update():
    from gcnet_missing_m3.priority40_relations import GATv2,PNA
    x=torch.randn(3,5,64)
    mask=torch.tensor([[1,1,0,1,0],[1,1,1,1,1],[1,1,0,0,0]],dtype=torch.bool)
    clean=torch.where(mask[...,None],x,0.)
    for cls in (GATv2,PNA):
        model=cls()
        assert hasattr(model,'encode_roles')
        assert torch.equal(model(clean,mask),model.pool(model.encode_roles(clean,mask),mask))


def test_cwn_direct_preserves_core_initialization_rng_and_parameter_count():
    from gcnet_missing_m3.meaningful_input_new40 import build_new40
    assert 'cwn_cellular_direct' in INPUT_METHODS
    torch.manual_seed(66)
    old=build_new40('cwn_cellular_evidence')
    rng=torch.get_rng_state().clone()
    torch.manual_seed(66)
    new=build_new40('cwn_cellular_direct')
    assert torch.equal(rng,torch.get_rng_state())
    assert sum(p.numel() for p in old.parameters())==sum(p.numel() for p in new.parameters())
    assert all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items()
               if not ('decoder' in k))


@pytest.mark.parametrize('method',METHODS)
def test_full_model_keeps_flat_and_raw_skip(method):
    pytest.importorskip('torch_geometric')
    from tests.test_meaningful_block_integration import config,_build_model
    torch.set_num_threads(1)
    base=_build_model(config(),(3,4,5))
    model=_build_model(config(method),(3,4,5)).eval()
    assert model.osram.meaningful_input_mode
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in base.state_dict().items())
    seen={}
    def pre(module,inputs): seen['local']=inputs[0].detach().clone()
    def skip(module,inputs): seen['skip']=inputs[0].detach().clone()
    def head(module,inputs): seen['hidden']=inputs[0].detach().clone()
    handles=[model.osram.meaningful_block.register_forward_pre_hook(pre),
             model.osram.local_skip.register_forward_pre_hook(skip),
             model.smax_fc.register_forward_pre_hook(head)]
    u=torch.tensor([[1.,1.,1.],[1.,0.,0.]])
    av=torch.tensor([[[1.,0.,1.],[1.,1.,1.]]]*3); av[~u.T.bool()]=0.
    try:
        pred=model([torch.randn(3,2,12)],av,torch.zeros(2,3,dtype=torch.long),u,[3,1],predict_missing=False)[0]
    finally:
        for h in handles: h.remove()
    assert torch.equal(seen['local'],seen['skip'])
    assert torch.isfinite(pred).all() and not seen['hidden'][~u.T.bool()].count_nonzero()
