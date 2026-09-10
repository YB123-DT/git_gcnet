import copy
from unittest.mock import patch
import pytest
import torch
from gcnet_missing_m3.model import MissingM3GraphModel


def make(enabled=False):
    torch.manual_seed(34)
    return MissingM3GraphModel('LSTM',3,4,5,4,4,1,2,2,1,dropout=0,
        latent_dim=8,projector_dropout=0,predictor_dropout=0,backbone_type='osram',
        osram_num_heads=2,osram_key_dim=3,osram_value_dim=3,osram_output_dim=12,
        osram_bidirectional=False,osram_write_step=.6,complete_state_jepa=enabled)


def batch():
    x=torch.randn(4,1,12)
    a=torch.tensor([[[1,1,1]],[[1,0,1]],[[0,1,0]],[[0,0,0]]]).float()
    u=torch.tensor([[1,1,1,0]]).float(); q=torch.zeros(1,4).long()
    mask=torch.cat([a[...,0:1].expand(-1,-1,3),a[...,1:2].expand(-1,-1,4),a[...,2:3].expand(-1,-1,5)],-1)
    return x,a,u,q,mask


def test_shared_init_rng_forward_and_teacher_no_memory():
    old=make();rng=torch.get_rng_state();new=make(True)
    assert torch.equal(rng,torch.get_rng_state())
    for k,v in old.state_dict().items(): assert torch.equal(v,new.state_dict()[k]),k
    x,a,u,q,mask=batch();old.eval();new.eval()
    before=old([x*mask],a,q,u,[3])[0]
    with patch.object(new.missing_predictor,'forward',side_effect=AssertionError('legacy predictor')):
        after,h,_,_=new([x*mask],a,q,u,[3])
        assert torch.equal(before,after)
        with patch.object(new.osram,'forward',side_effect=AssertionError('teacher memory')):
            target=new.state_jepa.target(x,u)
            assert target.shape==(4,1,8) and not target.requires_grad
            altered=x.clone();altered[2]+=10
            assert torch.equal(target[:2],new.state_jepa.target(altered,u)[:2])
    assert torch.count_nonzero(target[3])==0


def test_loss_missing_only_empty_gradient_and_target_isolation():
    m=make(True);x,a,u,q,mask=batch();x.requires_grad_()
    _,h,_,_=m([x*mask],a,q,u,[3])
    loss,n=m.complete_state_loss(h,x,a,u)
    assert n==2 and torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None for p in m.state_jepa.teacher_encoder.parameters())
    assert all(p.grad is None for p in m.state_jepa.teacher_local.parameters())
    for module in (m.state_jepa.predictor,m.observed_set,m.osram):
        assert sum(p.grad.abs().sum() for p in module.parameters() if p.grad is not None)>0
    assert torch.count_nonzero(x.grad[mask==0])==0
    complete=u.T[...,None].expand(-1,-1,3)
    z,n=m.complete_state_loss(h.detach().requires_grad_(),x,complete,u)
    assert n==0 and z.item()==0;z.backward()
    changed=x.detach().clone();changed[0]+=100;changed[3]=float('nan')
    h=h.detach()
    assert torch.equal(m.complete_state_loss(h,x.detach(),a,u)[0],m.complete_state_loss(h,changed,a,u)[0])


def test_exact_ema_eval_and_checkpoint_restore():
    m=make(True);m.train()
    assert not m.state_jepa.teacher_encoder.training and not m.state_jepa.teacher_local.training
    previous=copy.deepcopy(m.state_jepa.teacher_encoder.state_dict())
    with torch.no_grad():
        for p in m.observed_set.parameters():p.add_(.1)
    m.update_teacher(.8)
    for k,v in m.state_jepa.teacher_encoder.state_dict().items():
        expected=previous[k]*.8+m.observed_set.state_dict()[k]*.2
        torch.testing.assert_close(v,expected)
    assert m.ema_step==1
    restored=make(True);restored.load_state_dict(m.state_dict(),strict=True)
    for k,v in m.state_dict().items():assert torch.equal(v,restored.state_dict()[k])


def test_pattern_encoding_all_seven_and_padding():
    m=make(True)
    a=torch.tensor([[(i>>2&1,i>>1&1,i&1)] for i in range(8)]).float()
    u=torch.tensor([[0,1,1,1,1,1,1,1]]).float()
    h=torch.randn(8,1,12)
    seen=[]
    hook=m.state_jepa.pattern.register_forward_pre_hook(lambda _,args:seen.append(args[0]))
    pred=m.state_jepa.predict(h,a,u);hook.remove()
    assert pred.shape==(8,1,8) and torch.count_nonzero(pred[0])==0
    assert torch.equal(seen[0],torch.arange(1,8))
