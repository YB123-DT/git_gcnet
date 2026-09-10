"""Only emotion fusion changes: active evidence, subject path, scan invariants."""
from unittest.mock import patch
import pytest
import torch

from gcnet_missing_m3 import osram


def sample():
    torch.manual_seed(28)
    local=torch.randn(4,2,8)
    base=torch.randn(4,2,12)
    gap=torch.randn(4,2,3,12)
    availability=torch.tensor([[[1,1,1],[1,0,1]],[[0,1,0],[1,1,0]],
                               [[0,0,1],[0,0,0]],[[0,0,0],[0,0,0]]]).float()
    umask=torch.tensor([[1,1,1,0],[1,1,0,0]]).float()
    return local,base,gap,availability,umask


def fusion():
    return osram.LocalCenteredContextFusion(8,12,10,interaction_dim=4,type_dim=2,dropout=0)


def awaken(m):
    torch.nn.init.normal_(m.context_out[-1].weight,std=.2)


def test_zero_init_shape_active_masks_and_padding():
    m=fusion().eval(); args=sample()
    h=m(*args); valid=args[-1].T.bool()
    expected=m.emotion_norm(m.local_skip(args[0])).masked_fill(~valid[...,None],0)
    assert h.shape==(4,2,10) and torch.equal(h,expected)
    active=torch.cat([valid[...,None], valid[...,None] & ~args[3].bool()],-1)
    assert torch.equal(m.last_active,active)
    assert torch.count_nonzero(m.last_gates[~active])==0
    assert not m.last_gates.requires_grad
    assert m.last_diagnostics['context_residual_norm']==0
    complete=list(args);complete[3]=torch.ones_like(args[3]);m(*complete)
    assert all(m.last_diagnostics['mean_gate'][k] is None for k in ('gap_audio','gap_text','gap_visual'))


def test_inactive_evidence_ignored_active_base_gap_and_local_change():
    m=fusion().eval();awaken(m);args=sample();h=m(*args)
    old_gates=m.last_gates.clone();active=m.last_active.clone()
    changed=list(args);changed[2]=args[2].clone()
    changed[2][~active[...,1:]]=float('nan')
    assert torch.equal(h,m(*changed))
    changed=list(args);changed[2]=args[2].clone()
    changed[2][1,0,0]=torch.randn(12)*9
    assert not torch.equal(h,m(*changed))
    changed=list(args);changed[1]=torch.randn_like(args[1])*3
    assert not torch.equal(h,m(*changed))
    changed=list(args);changed[0]=torch.randn_like(args[0])*3;m(*changed)
    assert all(torch.any(old_gates[...,i][active[...,i]]!=m.last_gates[...,i][active[...,i]]) for i in range(4))


def test_no_softmax_and_active_count_average():
    m=fusion().eval();awaken(m);args=sample()
    for p in m.gate.parameters():torch.nn.init.zeros_(p)
    m(*args);active=m.last_active
    assert torch.equal(m.last_gates[active],torch.full_like(m.last_gates[active],.5))
    captured=[]
    hook=m.context_out.register_forward_pre_hook(lambda _,a: captured.append(a[0]))
    m(*args);hook.remove()
    evidence=torch.cat([args[1].unsqueeze(2),args[2]],2)
    evidence=torch.where(active[...,None],evidence,torch.zeros_like(evidence))
    v=m.value(m.context_norm(evidence))
    expected=(.5*active[...,None]*v).sum(2)/active.sum(-1).clamp_min(1)[...,None]
    torch.testing.assert_close(captured[0],expected,rtol=0,atol=0)


def test_two_steps_reach_gate_query_key_value_after_zero_init():
    m=fusion();args=sample();opt=torch.optim.SGD(m.parameters(),lr=.1)
    target=torch.randn(4,2,10)
    for step in range(2):
        opt.zero_grad();loss=(m(*args)-target).square().mean();loss.backward()
        if step==0:
            assert m.context_out[-1].weight.grad.abs().sum()>0
            assert m.query.weight.grad.abs().sum()==0
        else:
            for name in ('query','key','value','gate','context_out'):
                grads=[p.grad for p in getattr(m,name).parameters() if p.grad is not None]
                assert sum(g.abs().sum().item() for g in grads)>0,name
                assert all(torch.isfinite(g).all() for g in grads)
        opt.step()


def test_gated_preserves_scan_contexts_and_shared_initialization():
    kw=dict(latent_dim=8,output_dim=10,num_heads=2,key_dim=3,value_dim=3,
            dropout=0,n_speakers=1,bidirectional=False,write_step=.6)
    torch.manual_seed(31);flat=osram.OSRAMBackbone(**kw).eval();rng=torch.get_rng_state()
    torch.manual_seed(31);gated=osram.OSRAMBackbone(**kw,osram_readout_fusion='local-gated').eval()
    assert torch.equal(rng,torch.get_rng_state())
    for name,p in flat.state_dict().items():assert torch.equal(p,gated.state_dict()[name]),name
    local,_,_,a,u=sample();zs={k:torch.randn_like(local) for k in osram.MODALITIES}
    args=(local,zs,a,torch.zeros(2,4).long(),u,[3,2])
    traces=[];outputs=[]
    for m in (flat,gated):
        states=[];old=m.block_write
        def capture(*aa,**kk):
            post=old(*aa,**kk);states.append(post.detach().clone());return post
        with patch.object(m,'block_write',side_effect=capture):outputs.append(m(*args))
        traces.append(states)
    for left,right in zip(*traces):assert torch.equal(left,right)
    for name in ('local','base','gap'):assert torch.equal(outputs[0][1][name],outputs[1][1][name])
    assert torch.equal(outputs[0][0],outputs[1][0])
    for left,right in zip(flat._project_sequence(*args[:4]),gated._project_sequence(*args[:4])):
        if isinstance(left,dict):
            for k in left:assert torch.equal(left[k],right[k])
        else:assert torch.equal(left,right)


def test_shape_checks_and_empty_valid_batch():
    m=fusion();args=list(sample());args[-1]=torch.zeros_like(args[-1])
    assert not m(*args).count_nonzero()
    assert all(v is None for v in m.last_diagnostics['mean_gate'].values())
    with pytest.raises(ValueError):m(args[0],args[1][:-1],*args[2:])
