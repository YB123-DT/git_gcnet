"""Original numerical path, causal source tracking, and opt-in TASR contracts."""
import copy
from pathlib import Path
import subprocess

import pytest
import torch

from gcnet_missing_m3.osram import OSRAMBackbone
from test_osram import _inputs


def model(**kwargs):
    return OSRAMBackbone(latent_dim=8,output_dim=10,num_heads=2,key_dim=3,value_dim=4,
        n_speakers=2,dropout=kwargs.pop('dropout',0.),bidirectional=False,write_step=.6,**kwargs)


def test_disabled_matches_pre_tasr_state_forward_gradients_and_rng():
    source=subprocess.check_output(['git','show','ea27c08:gcnet_missing_m3/osram.py'],
        cwd=Path(__file__).resolve().parents[1],text=True)
    namespace={'__name__':'gcnet_missing_m3._legacy_tasr','__package__':'gcnet_missing_m3'}
    exec(compile(source,'pre_tasr_osram.py','exec'),namespace)
    torch.manual_seed(94)
    old=namespace['OSRAMBackbone'](latent_dim=8,output_dim=10,num_heads=2,key_dim=3,value_dim=4,
        n_speakers=2,dropout=.2,bidirectional=False,write_step=.6)
    after_init=torch.get_rng_state().clone()
    torch.manual_seed(94);new=model(dropout=.2,osram_tasr=False,osram_source_tracking=False)
    assert torch.equal(after_init,torch.get_rng_state())
    assert list(new.state_dict())==list(old.state_dict())
    for n,p in old.state_dict().items():assert torch.equal(new.state_dict()[n],p)
    torch.nn.init.normal_(old.emotion_adapter[-1].weight,std=.1)
    new.load_state_dict(old.state_dict(),strict=True);args=_inputs()
    rng=torch.get_rng_state().clone();a,ac=old(*args);a.square().sum().backward();after=torch.get_rng_state().clone()
    torch.set_rng_state(rng);b,bc=new(*args);b.square().sum().backward()
    assert torch.equal(a,b) and torch.equal(after,torch.get_rng_state())
    for k in ac:assert torch.equal(ac[k],bc[k])
    for n,p in old.named_parameters():
        q=dict(new.named_parameters())[n]
        assert (p.grad is None)==(q.grad is None)
        if p.grad is not None:assert torch.equal(p.grad,q.grad),n


@pytest.mark.parametrize('tracking,tasr',[(True,False),(False,True)])
def test_zero_init_or_tracking_only_keeps_legacy_output_exact(tracking,tasr):
    torch.manual_seed(18);base=model().eval();rng=torch.get_rng_state().clone()
    torch.manual_seed(18);augmented=model(osram_source_tracking=tracking,osram_tasr=tasr).eval()
    assert torch.equal(rng,torch.get_rng_state())
    for n,p in base.state_dict().items():assert torch.equal(p,augmented.state_dict()[n]),n
    torch.nn.init.normal_(base.emotion_adapter[-1].weight,std=.1)
    state=augmented.state_dict();state.update(base.state_dict());augmented.load_state_dict(state)
    args=_inputs();a,ac=base(*args);b,bc=augmented(*args)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    for k in ('base','gap','local'):torch.testing.assert_close(ac[k],bc[k],rtol=0,atol=0)
    trace=augmented.last_source_trace
    torch.testing.assert_close(trace['source_gap'].sum(3),bc['gap'][...,:8],rtol=3e-5,atol=3e-6)
    assert not trace['source_gap'][0].any()
    assert trace['diagnostics']['memory_sum_max_abs_error']<1e-5


def test_frozen_backbone_only_adapter_updates_and_remains_causal():
    m=model(osram_tasr=True).eval()
    torch.nn.init.normal_(m.emotion_adapter[-1].weight,std=.1)
    m.requires_grad_(False);m.tasr.requires_grad_(True)
    before={n:p.detach().clone() for n,p in m.named_parameters()}
    optimizer=torch.optim.Adam(m.tasr.parameters(),lr=.003)
    args=_inputs()
    for step in range(3):
        optimizer.zero_grad(set_to_none=True);h,_=m(*args)
        loss=(h[...,0]-.7).square().mean();loss.backward()
        grads=[p.grad for p in m.tasr.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(g.ne(0).any() for g in grads)
        optimizer.step()
    assert any(not torch.equal(before[n],p) for n,p in m.named_parameters() if n.startswith('tasr.'))
    assert all(torch.equal(before[n],p) for n,p in m.named_parameters() if not n.startswith('tasr.'))
    changed=list(args);changed[0]=args[0].clone();changed[0][2]+=5
    changed[1]={n:v.clone() for n,v in args[1].items()}
    for v in changed[1].values():v[2]+=8
    a,_=m(*args);b,_=m(*changed);torch.testing.assert_close(a[:2],b[:2],rtol=0,atol=0)
    assert not a[3].any()
    clone=model(osram_tasr=True).eval();clone.load_state_dict(m.state_dict(),strict=True)
    torch.testing.assert_close(a,clone(*args)[0],rtol=0,atol=0)


def test_conversations_do_not_share_source_states():
    m=model(osram_source_tracking=True).eval();args=list(_inputs())
    _,c=m(*args);source=m.last_source_trace['source_gap'].clone()
    args[0]=args[0].clone();args[0][:,0]+=12
    args[1]={k:v.clone() for k,v in args[1].items()}
    for v in args[1].values():v[:,0]-=4
    _,d=m(*args)
    torch.testing.assert_close(source[:,1],m.last_source_trace['source_gap'][:,1],rtol=0,atol=0)
    torch.testing.assert_close(c['gap'][:,1],d['gap'][:,1],rtol=0,atol=0)


def test_invalid_tracking_modes_are_rejected():
    with pytest.raises(ValueError,match='TASR|source'):
        model(osram_tasr=True,beta_mode='external-head')


@pytest.mark.parametrize('switch', ['osram_ablation', 'osram_emotion_ablation'])
def test_runtime_gap_ablation_cannot_be_bypassed(switch):
    m=model(osram_tasr=True).eval()
    torch.nn.init.normal_(m.emotion_adapter[-1].weight, std=.2)
    torch.nn.init.constant_(m.tasr.score[-1].bias, 1.)
    setattr(m, switch, 'local-base')
    output,_=m(*_inputs())
    m.osram_tasr=False
    expected,_=m(*_inputs())
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
