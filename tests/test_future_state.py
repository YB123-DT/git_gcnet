"""Post-write memory, rather than emotion hidden, predicts the next Local state."""
import copy
from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F
from gcnet_missing_m3.model import MissingM3GraphModel
from test_osram import _inputs
from gcnet_missing_m3.osram import OSRAMBackbone


def make(enabled=False, **extra):
    torch.manual_seed(34)
    kw = dict(dropout=0, latent_dim=8, projector_dropout=0, predictor_dropout=0,
              backbone_type='osram', osram_num_heads=2, osram_key_dim=3,
              osram_value_dim=3, osram_output_dim=12, fusion_type='mean',
              osram_bidirectional=False, osram_write_step=.6,
              future_state_jepa=enabled)
    kw.update(extra)
    return MissingM3GraphModel('LSTM',3,4,5,4,4,2,2,2,1, **kw)


def batch():
    x = torch.randn(4,2,12)
    a = torch.tensor([[[1,1,1],[1,0,1]], [[1,0,1],[0,1,0]],
                      [[0,1,0],[0,0,0]], [[0,0,0],[0,0,0]]]).float()
    u = torch.tensor([[1,1,1,0],[1,1,0,0]]).float()
    q = torch.tensor([[0,1,0,0],[1,0,0,0]])
    mask = torch.cat([a[...,i:i+1].expand(-1,-1,d) for i,d in enumerate((3,4,5))],-1)
    return x,a,u,q,mask


def call(m,x,a,u,q,mask):
    return m([x*mask],a,q,u,[3,2])


def test_shared_initialization_rng_strict_restore_and_inference_bypass():
    old=make(); rng=torch.get_rng_state(); new=make(True)
    assert torch.equal(rng,torch.get_rng_state())
    for k,v in old.state_dict().items(): assert torch.equal(v,new.state_dict()[k]),k
    old.load_state_dict(old.state_dict(),strict=True)
    new.load_state_dict(new.state_dict(),strict=True)
    x,a,u,q,mask=batch();old.eval();new.eval()
    with (patch.object(new.future_state,'observe',side_effect=AssertionError('state query')),
         patch.object(new.future_state.predictor,'forward',side_effect=AssertionError('predictor')),
         patch.object(new.future_state,'target',side_effect=AssertionError('teacher'))):
        assert torch.equal(call(old,x,a,u,q,mask)[0],call(new,x,a,u,q,mask)[0])


def test_postwrite_observer_does_not_modify_scan_and_is_after_write():
    m=OSRAMBackbone(8,10, num_heads=2,key_dim=3,value_dim=4,n_speakers=2,
                    dropout=0,bidirectional=False)
    args=_inputs(); baseline=m(*args)[0]; states=[]; writes=[]
    original=m.block_write
    def write(*a,**kw):
        out=original(*a,**kw);writes.append(out.detach().clone());return out
    def observer(t,memory,valid):
        assert memory.requires_grad
        states.append(memory.detach().clone())
        memory.zero_()  # A faulty observer must not mutate the persistent state.
    with patch.object(m,'block_write',side_effect=write):
        actual=m(*args,post_write_observer=observer)[0]
    assert torch.equal(actual,baseline)
    assert len(states)==3
    for s,w in zip(states,writes):assert torch.equal(s,w)


def test_causal_current_write_and_no_future_dependency():
    m=make(True).train();x,a,u,q,mask=batch()
    call(m,x,a,u,q,mask); before=m.future_state.compact_states().detach().clone()
    changed=x.clone();changed[1:]+=torch.randn_like(changed[1:])*5
    call(m,changed,a,u,q,mask); after=m.future_state.compact_states()
    assert torch.equal(before[0],after[0])
    changed=x.clone();changed[0]+=torch.randn_like(changed[0])*5
    call(m,changed,a,u,q,mask)
    assert not torch.equal(before[0],m.future_state.compact_states()[0])
    assert torch.count_nonzero(before[~u.T.bool()])==0


def test_future_loss_gradient_and_no_target_input_gradient():
    m=make(True);x,a,u,q,mask=batch();x.requires_grad_()
    call(m,x,a,u,q,mask)
    loss,n=m.future_state_loss(x,u)
    assert n==3 and torch.isfinite(loss)
    loss.backward()
    assert m.future_state.state_query.grad.norm()>0
    for module in (m.future_state.predictor,m.osram.key_projectors,m.osram.value_projectors,m.observed_set):
        assert sum(p.grad.abs().sum() for p in module.parameters() if p.grad is not None)>0
    assert all(p.grad is None for p in m.future_state.teacher_encoder.parameters())
    assert all(p.grad is None for p in m.future_state.teacher_local.parameters())
    assert torch.count_nonzero(x.grad[mask==0])==0


def test_shift_manual_loss_counts_and_teacher_current_only():
    m=make(True);x,a,u,q,mask=batch();call(m,x,a,u,q,mask)
    states=m.future_state.compact_states()
    chosen=u.T.bool()[:-1]&u.T.bool()[1:]
    target=m.future_state.target(x,u)
    pred=m.future_state.predictor(states[:-1][chosen])
    expected=F.smooth_l1_loss(F.layer_norm(pred,(8,)),F.layer_norm(target[1:][chosen],(8,)))
    actual,n=m.future_state_loss(x,u)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert n==int((u.sum(-1)-1).sum())==3
    full=u.T[...,None].expand(-1,-1,3)
    call(m,x,full,u,q,torch.ones_like(mask));assert m.future_state_loss(x,u)[1]==n
    changed=x.clone();changed[2:]+=5
    with patch.object(m.osram,'forward',side_effect=AssertionError('teacher memory')):
        assert torch.equal(target[:2],m.future_state.target(changed,u)[:2])
    assert not target.requires_grad


def test_parameter_count_nearest_solution_and_ema():
    m=make(True);head=m.future_state
    counts=[head.parameter_count(w) for w in range(1,1000)]
    online=sum(p.numel() for p in head.parameters() if p.requires_grad)
    assert online==head.parameter_count(head.width)
    assert abs(online-head.parameter_budget)==min(abs(c-head.parameter_budget) for c in counts)
    previous=copy.deepcopy(head.teacher_encoder.state_dict())
    with torch.no_grad():
        for p in m.observed_set.parameters():p.add_(.1)
    m.update_teacher(.8)
    assert int(head.ema_updates)==1 and m.ema_step==1
    for k,v in head.teacher_encoder.state_dict().items():
        torch.testing.assert_close(v,previous[k]*.8+m.observed_set.state_dict()[k]*.2)
    m.train();assert not head.teacher_encoder.training and not head.teacher_local.training


def test_training_observer_preserves_emotion_forward_and_gradients():
    old=make(dropout=.2,projector_dropout=.1)
    new=make(True,dropout=.2,projector_dropout=.1)
    torch.nn.init.normal_(old.osram.emotion_adapter[-1].weight,std=.1)
    new.osram.load_state_dict(old.osram.state_dict(),strict=True)
    x,a,u,q,mask=batch()
    rng=torch.get_rng_state()
    expected=call(old,x,a,u,q,mask)[0]
    torch.set_rng_state(rng)
    actual=call(new,x,a,u,q,mask)[0]
    assert torch.equal(actual,expected)
    expected.square().sum().backward();actual.square().sum().backward()
    for name,p in old.named_parameters():
        other=dict(new.named_parameters())[name]
        if p.grad is None: assert other.grad is None
        else: assert torch.equal(p.grad,other.grad),name


def test_single_utterance_has_no_future_target_and_eval_clears_cache():
    m=make(True);x,a,u,q,mask=batch()
    u.zero_();u[:,0]=1;a[1:]=0;q.zero_()
    m([x],a,q,u,[1,1])
    with patch.object(m.future_state,'target',side_effect=AssertionError('empty teacher')):
        loss,n=m.future_state_loss(x,u)
    assert n==0 and loss.item()==0;loss.backward()
    m.eval()
    with pytest.raises(ValueError,match='training forward'):
        m.future_state.compact_states()


@pytest.mark.parametrize('latent,heads,dk,dv,hidden',[(256,8,32,32,700),(16,3,4,5,27)])
def test_parameter_budget_matches_actual_complete_state_head(latent,heads,dk,dv,hidden):
    from gcnet_missing_m3.future_state import FutureStateJEPA
    from gcnet_missing_m3.complete_state import CompleteViewLocalStateJEPA
    encoder=torch.nn.Linear(latent,latent);local=torch.nn.Identity()
    future=FutureStateJEPA(encoder,local,hidden,latent,heads,dk,dv)
    old=CompleteViewLocalStateJEPA(encoder,local,hidden,latent)
    assert future.parameter_budget==sum(p.numel() for p in old.parameters() if p.requires_grad)
    count=sum(p.numel() for p in future.parameters() if p.requires_grad)
    assert count==future.parameter_count(future.width)
    for w in (max(1,future.width-1),future.width+1):
        assert abs(count-future.parameter_budget)<=abs(future.parameter_count(w)-future.parameter_budget)


@pytest.mark.parametrize('extra',[dict(write_state_completion=True),dict(complete_state_jepa=True),
    dict(completion_path='pre_osram_b2'),dict(classification_completion=True),
    dict(osram_bidirectional=True),dict(osram_write_step=1),dict(osram_readout_fusion='local-gated')])
def test_reject_conflicting_modes(extra):
    with pytest.raises(ValueError,match='future-state'):make(True,**extra)
