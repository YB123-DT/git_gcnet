import torch
from gcnet_missing_m3.meaningful_input import INPUT_METHODS, MeaningfulInputAdapter
from gcnet_missing_m3.priority40_pooling_geometry import XCiTXCA


def test_xca_direct_registered_and_no_input_addition():
    assert 'm28_xcit_xca_direct' in INPUT_METHODS
    from gcnet_missing_m3.meaningful_input_priority40 import XCADirectInput
    torch.manual_seed(66)
    core = XCADirectInput()
    local = torch.randn(3,256)
    evidence = torch.randn(3,4,512)
    active = torch.tensor([[1,0,1,1],[1,1,0,0],[1,0,0,0]],dtype=torch.bool)
    dirty = torch.where(active[...,None], evidence, float('nan'))
    av = (~active[:,1:]).float()
    l,c = core(local,dirty,active,av)
    assert torch.isfinite(l).all() and torch.isfinite(c).all()
    assert c[~active].count_nonzero() == 0
    assert l.count_nonzero() and c[active].count_nonzero()
    with torch.no_grad():
        core.local_decoder.weight.zero_(); core.local_decoder.bias.zero_()
        core.memory_decoder.weight.zero_(); core.memory_decoder.bias.zero_()
    l,c = core(local,dirty,active,av)
    assert l.count_nonzero() == 0 and c.count_nonzero() == 0


def test_original_xca_pool_matches_unchanged_equations():
    import torch.nn.functional as F
    torch.manual_seed(66)
    core = XCiTXCA()
    x = torch.randn(3,5,64)
    mask = torch.tensor([[1,1,0,1,0],[1,1,1,1,1],[1,1,0,0,0]],dtype=torch.bool)
    clean = torch.where(mask[...,None],x,0.)
    qkv = torch.where(mask[...,None],core.qkv(clean),0.).reshape(3,5,3,4,16)
    q,k,v = qkv.permute(2,0,3,4,1).unbind(0)
    q,k = F.normalize(q,dim=-1,eps=1e-6),F.normalize(k,dim=-1,eps=1e-6)
    a = ((q@k.transpose(-1,-2))*core.temperature).softmax(-1)
    updated = (a@v).permute(0,3,1,2).reshape(3,5,64)
    expected = torch.where(mask[...,None],core.output(updated),0.).sum(1)/mask.sum(1,keepdim=True)
    assert torch.equal(core(clean,mask),expected)


def test_direct_wrapper_preserves_first_turn_and_padding_and_updates_xca():
    assert 'm28_xcit_xca_direct' in INPUT_METHODS
    torch.manual_seed(66)
    block = MeaningfulInputAdapter(256,1024,1600,'m28_xcit_xca_direct',8,64)
    local,base,gap = torch.randn(3,1,256),torch.randn(3,1,1024),torch.randn(3,1,3,1024)
    base[...,512:] = 0.; gap[...,512:] = 0.
    av = torch.tensor([[[1.,0.,1.]],[[1.,0.,1.]],[[0.,0.,0.]]])
    mask = torch.tensor([[1.,1.,0.]])
    dirty = torch.where((av<.5)[...,None],gap,float('nan'))
    result = block(local,base,dirty,av,mask)
    assert torch.equal(result[0][0],local[0]) and torch.equal(result[1][0],base[0])
    assert all(torch.isfinite(x).all() and not x[2].count_nonzero() for x in result)
    assert not result[2][1,0,[0,2]].count_nonzero()
    assert not result[1][...,512:].count_nonzero() and not result[2][...,512:].count_nonzero()
    optimizer = torch.optim.Adam(block.parameters(),lr=.001)
    old = block.core.features.operator.qkv.weight.detach().clone()
    sum(x[1].square().mean() for x in result).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in block.parameters())
    assert block.core.features.operator.qkv.weight.grad.norm() > 0
    optimizer.step()
    assert not torch.equal(old,block.core.features.operator.qkv.weight)


def test_nps_direct_removes_only_evidence_bypass_and_preserves_default_rng():
    assert 'neural_production_direct' in INPUT_METHODS
    from gcnet_missing_m3.meaningful_input_new40 import build_new40
    from gcnet_missing_m3.meaningful_new40_conditional import NeuralProduction, TokenReadout
    torch.manual_seed(66)
    old = TokenReadout(NeuralProduction(),256,8,64)
    old_rng = torch.get_rng_state().clone()
    torch.manual_seed(66)
    direct = build_new40('neural_production_direct')
    assert torch.equal(old_rng,torch.get_rng_state())
    for k,v in old.state_dict().items():
        if not k.startswith('bridges.'):
            assert torch.equal(v,direct.state_dict()[k])
    local,evidence = torch.randn(2,256),torch.randn(2,4,512)
    active = torch.tensor([[1,0,1,0],[1,1,0,1]],dtype=torch.bool)
    evidence = torch.where(active[...,None],evidence,float('nan'))
    av = (~active[:,1:]).float()
    l,c = direct(local,evidence,active,av)
    assert torch.equal(l,local) and torch.isfinite(c).all() and not c[~active].count_nonzero()
    c.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in direct.parameters())
    assert sum(p.grad.abs().sum() for p in direct.core.parameters() if p.grad is not None) > 0
    with torch.no_grad():
        for bridge in direct.bridges:
            bridge.weight.zero_(); bridge.bias.zero_()
    l,c = direct(local,evidence,active,av)
    assert torch.equal(l,local) and not c.count_nonzero()
    l,c = old(local,evidence,active,av)
    assert torch.equal(l,local) and torch.equal(c,torch.where(active[...,None],evidence,0.))


def test_direct_full_models_retain_original_skip_and_parameter_initialization():
    import pytest
    pytest.importorskip('torch_geometric')
    from tests.test_meaningful_block_integration import config, _build_model
    torch.set_num_threads(1)
    baseline = _build_model(config(),(3,4,5))
    for method in ('m28_xcit_xca_direct','neural_production_direct'):
        model = _build_model(config(method),(3,4,5)).eval()
        assert model.osram.meaningful_input_mode
        for k,v in baseline.state_dict().items():
            assert torch.equal(v,model.state_dict()[k]), (method,k)
        seen = {}
        def pre(module, inputs):
            seen['local'] = inputs[0].detach().clone()
        def skip(module, inputs):
            seen['skip'] = inputs[0].detach().clone()
        def head(module, inputs):
            seen['hidden'] = inputs[0].detach().clone()
        h1=model.osram.meaningful_block.register_forward_pre_hook(pre)
        h2=model.osram.local_skip.register_forward_pre_hook(skip)
        h3=model.smax_fc.register_forward_pre_hook(head)
        x = torch.randn(3,2,12)
        u = torch.tensor([[1.,1.,1.],[1.,0.,0.]])
        av = torch.tensor([[[1.,0.,1.],[1.,1.,1.]]]*3)
        av[~u.T.bool()] = 0.
        try:
            pred=model([x],av,torch.zeros(2,3,dtype=torch.long),u,[3,1],predict_missing=False)[0]
        finally:
            h1.remove(); h2.remove(); h3.remove()
        assert torch.equal(seen['local'],seen['skip'])
        assert torch.isfinite(pred).all() and not seen['hidden'][~u.T.bool()].count_nonzero()
        # Original task head has bias: padded logits equal bias, and task loss
        # excludes them with umask. Do not change the original head to force zero.
        assert torch.equal(pred[~u.T.bool()],model.smax_fc.bias.expand_as(pred[~u.T.bool()]))
