from types import SimpleNamespace
import torch
import pytest

from test_osram import _inputs
from test_source_role_integration import make
from experiments.osram_12_source_evidence_audit_20261019.run import capture_flat


def fixture(bias):
    torch.manual_seed(193)
    m=make(osram_source_role=True,osram_source_role_mode='direct').eval().requires_grad_(False)
    torch.nn.init.normal_(m.emotion_adapter[-1].weight,std=.2)
    torch.nn.init.constant_(m.source_role.net[-1].bias,bias)
    head=torch.nn.Linear(10,1).eval().requires_grad_(False)
    gamma=[];hook=m.source_role.register_forward_hook(lambda _,x,y:gamma.append(y[1].clone()))
    try:
        with torch.no_grad(),capture_flat(SimpleNamespace(osram=m)) as cap:
            hidden,_=m(*_inputs());full=head(hidden)
    finally:hook.remove()
    trace=m.last_source_trace
    sources=torch.cat((trace['source_base'].unsqueeze(2),trace['source_gap']),2)
    av=_inputs()[2];umask=_inputs()[4]
    return m,head,cap,sources,gamma[0],trace['trajectory'],av,umask,full


def replay(args):
    from experiments.osram_mosi_source_role_neutral_audit_20261020.readout import replay_neutral
    m,head,cap,sources,gamma,trajectory,av,umask,full=args
    return replay_neutral(m,head,cap['local'],cap['emotion_input'],sources,gamma,trajectory,av,umask,full)


def test_neutral_unit_weights_reproduce_every_readout_without_new_scan():
    args=fixture(0.)
    rng=torch.get_rng_state().clone();state={k:v.clone() for k,v in args[0].state_dict().items()}
    result=replay(args)
    assert torch.equal(result['prediction_neutral'],args[-1])
    assert torch.equal(result['prediction_reset'],args[-1][:,:,None,None,:].expand(-1,-1,4,3,-1))
    assert torch.equal(rng,torch.get_rng_state())
    for k,v in state.items():assert torch.equal(v,args[0].state_dict()[k])


def test_nonunit_resets_use_one_source_weight_not_source_deletion():
    args=fixture(.7);m,head,cap,sources,gamma,trajectory,av,umask,full=args
    result=replay(args);fd=8;ld=8
    active=result['active'];safe=torch.where(active[...,None],sources,torch.zeros_like(sources))
    test=cap['emotion_input'].clone();test[...,ld:].reshape(4,2,4,16)[...,:fd]=safe.sum(-2)
    hidden=m.emotion_norm(m.local_skip(cap['local'])+m.emotion_adapter(test))
    expected=head(hidden*umask.T[...,None])
    assert torch.equal(result['prediction_neutral'],expected)
    assert not result['active'][0].any() and not result['active'][-1].any()
    for r in range(4):
        for s in range(3):
            inactive=~active[...,r,s]
            assert torch.equal(result['prediction_reset'][...,r,s,:][inactive],full[inactive])
    assert (result['prediction_neutral']-full).abs().max()>0


def test_replay_rejects_nonzero_backward_evidence():
    args=list(fixture(.7));args[2]=dict(args[2]);args[2]['emotion_input']=args[2]['emotion_input'].clone()
    args[2]['emotion_input'][0,0,8+8]=1.
    with pytest.raises(ValueError,match='backward'):replay(args)
