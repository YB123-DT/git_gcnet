import copy
from unittest.mock import patch
import torch
from gcnet_missing_m3.model import MissingM3GraphModel


def make(enabled=True):
    torch.manual_seed(61)
    return MissingM3GraphModel('LSTM',3,4,5,4,4,1,2,2,1,dropout=0,
        latent_dim=8,projector_dropout=0,predictor_dropout=0,backbone_type='osram',
        osram_num_heads=2,osram_key_dim=3,osram_value_dim=3,osram_output_dim=12,
        osram_bidirectional=False,osram_write_step=.6,write_state_completion=enabled)


def data():
    x=torch.randn(4,2,12);a=torch.ones(4,2,3);a[0,0,1]=0;a[1,0,0]=0
    u=torch.tensor([[1,1,1,1],[1,1,0,0]]).float();a[2:,1]=0
    q=torch.zeros(2,4).long()
    mask=torch.repeat_interleave(a,torch.tensor([3,4,5]),dim=-1)
    return x,a,u,q,mask


def test_default_exact_complete_init_and_budget():
    old=make(False);rng=torch.get_rng_state();new=make()
    assert torch.equal(rng,torch.get_rng_state())
    for k,v in old.state_dict().items():assert torch.equal(v,new.state_dict()[k]),k
    x,a,u,q,mask=data();a=u.T[...,None].expand(-1,-1,3)
    old.eval();new.eval()
    assert torch.equal(old([x],a,q,u,[4,2])[0],new([x],a,q,u,[4,2])[0])
    p=new.write_state.predictor
    actual=sum(v.numel() for v in p.parameters())
    assert actual==p.parameter_count(p.width)
    assert abs(actual-p.parameter_budget)<=min(abs(p.parameter_count(w)-p.parameter_budget) for w in (max(1,p.width-1),p.width+1))


def test_read_before_write_future_only_observed_copy_and_one_block_write():
    m=make().eval();x,a,u,q,mask=data();saved=[]
    original=m.osram.block_write
    def capture(mem,k,v,avail,**kw):
        saved.append((k.detach().clone(),v.detach().clone(),avail.clone()))
        return original(mem,k,v,avail,**kw)
    with patch.object(m.osram,'block_write',side_effect=capture):
        m([x*mask],a,q,u,[4,2]);first={k:m.last_osram_context[k].clone() for k in ('base','gap')}
    assert len(saved)==4
    encoded,latents=m.observed_set(x*mask,a,u)
    keys,values,_=m.osram._project_sequence(encoded,latents,a,q)
    for t,(ks,vs,write_mask) in enumerate(saved):
        for j,name in enumerate(('audio','text','visual')):
            obs=a[t,:,j].bool()
            assert torch.equal(ks[obs,...,j],keys[name][t,obs])
            assert torch.equal(vs[obs,...,j],values[name][t,obs])
        assert torch.equal(write_mask,u[:,t,None].expand(-1,3))
    original_predict=m.write_state.predictor.forward
    def changed(*args):
        k,v=original_predict(*args)
        return k,v+10
    with patch.object(m.write_state.predictor,'forward',side_effect=changed):
        m([x*mask],a,q,u,[4,2])
    assert torch.equal(first['base'][0],m.last_osram_context['base'][0])
    assert not torch.equal(first['base'][1:,0],m.last_osram_context['base'][1:,0])
    assert torch.equal(first['base'][:,1],m.last_osram_context['base'][:,1])
    assert torch.count_nonzero(m.last_osram_context['base'][2:,1])==0


def test_auxiliary_and_future_emotion_gradients_teacher_isolation_ema_restore():
    m=make();x,a,u,q,mask=data()
    torch.nn.init.normal_(m.osram.emotion_adapter[-1].weight,std=.1)
    logits,_,_,_=m([x*mask],a,q,u,[4,2])
    logits[-1,0].sum().backward(retain_graph=True)
    assert sum(p.grad.abs().sum() for p in m.write_state.predictor.parameters() if p.grad is not None)>0
    m.zero_grad();loss,n=m.write_state_loss(x,a,q,u);assert n==2
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    assert sum(p.grad.abs().sum() for p in m.write_state.predictor.parameters() if p.grad is not None)>0
    assert all(p.grad is None for p in m.write_state.teacher.parameters())
    before=copy.deepcopy(m.write_state.teacher.encoder.state_dict())
    with torch.no_grad():
        for p in m.observed_set.parameters():p.add_(.1)
    m.update_teacher(.8)
    for k,v in m.write_state.teacher.encoder.state_dict().items():
        torch.testing.assert_close(v,before[k]*.8+m.observed_set.state_dict()[k]*.2)
    assert m.write_state.ema_updates.item()==1
    restored=make();restored.load_state_dict(m.state_dict(),strict=True)
    assert restored.write_state.ema_updates.item()==1
    with patch.object(m.write_state.teacher,'forward',side_effect=AssertionError('test target')):
        m.eval();m([x*mask],a,q,u,[4,2])


def test_teacher_key_uses_observed_node_and_original_mask_no_future():
    m=make().eval();x,a,u,q,mask=data()
    teacher=m.write_state.teacher
    calls=[]
    hook=teacher.availability_embedding.register_forward_pre_hook(lambda _,args:calls.append(args[0].clone()))
    k,v=teacher(x,a,q,u);hook.remove()
    assert torch.equal(calls[0],a)
    y=x.clone();y[3]+=99
    ky,vy=teacher(y,a,q,u)
    assert torch.equal(k[:3],ky[:3]) and torch.equal(v[:3],vy[:3])
    assert not k.requires_grad and not v.requires_grad
    changed=x.clone();changed[0,0,3:7]+=3
    m([x*mask],a,q,u,[4,2]);before=m.last_osram_context['base'].clone()
    m([changed*mask],a,q,u,[4,2])
    assert torch.equal(before,m.last_osram_context['base'])


def test_future_features_cannot_change_earlier_hidden_and_empty_loss():
    m=make().eval();x,a,u,q,mask=data()
    before=m([x*mask],a,q,u,[4,2])[1]
    changed=x.clone();changed[3,0]+=9
    after=m([changed*mask],a,q,u,[4,2])[1]
    assert torch.equal(before[:3],after[:3])
    assert torch.count_nonzero(after[2:,1])==0
    full=u.T[...,None].expand(-1,-1,3)
    m.train();m([x],full,q,u,[4,2]);loss,n=m.write_state_loss(x,full,q,u)
    assert n==0 and loss.item()==0;loss.backward()


def test_supervision_matches_manual_missing_slot_average():
    m=make();x,a,u,q,mask=data();m([x*mask],a,q,u,[4,2])
    k,v=m.write_state.teacher(x,a,q,u);terms=[]
    for t,pk,pv in m.write_state.records:
        for b in range(2):
            for j in range(3):
                if u[b,t] and not a[t,b,j]:
                    terms.append(.5*(torch.nn.functional.smooth_l1_loss(pk[b,...,j],k[t,b,...,j])+
                                     torch.nn.functional.smooth_l1_loss(pv[b,...,j],v[t,b,...,j])))
    loss,count=m.write_state_loss(x,a,q,u)
    torch.testing.assert_close(loss,torch.stack(terms).mean());assert count==len(terms)
