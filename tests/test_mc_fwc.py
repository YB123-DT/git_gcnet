"""Counterfactual credit must be a restricted first-order task gradient."""
import copy
import math

import pytest
import torch

from gcnet_missing_m3.mc_fwc import pair_terms, source_writer_vjp, credit_backward, forward_view, writer_parameters
from gcnet_missing_m3.model import MissingM3GraphModel
from gcnet_missing_m3.fwc_sampling import CreditPair


def model_and_view():
    torch.manual_seed(42)
    model = MissingM3GraphModel('LSTM', 3, 4, 5, 4, 4, 2, 2, 2, 1,
        dropout=.2, latent_dim=8, projector_dropout=.1, predictor_dropout=0,
        backbone_type='osram', osram_num_heads=2, osram_key_dim=3,
        osram_value_dim=3, osram_output_dim=12, fusion_type='mean',
        osram_bidirectional=False, osram_write_step=.6,
        training_objective='emotion-only', disable_unused_aux_modules=True)
    a = torch.tensor([[[1,1,1],[1,1,0]],[[1,0,1],[0,1,1]],
                      [[1,0,1],[1,0,1]],[[1,1,0],[0,0,0]]]).float()
    u = torch.tensor([[1,1,1,1],[1,1,1,0]]).float()
    x = torch.randn(4,2,12)
    expanded = torch.repeat_interleave(a, torch.tensor([3,4,5]), dim=-1)
    view = dict(incomplete=x*expanded, complete=x, availability=a, umask=u,
        qmask=torch.zeros(2,4).long(), lengths=[4,3],
        labels=torch.tensor([[1.,-.7,.8,-1.2],[-.8,1.,-.9,0.]]),
        conversation_ids=['a','b'])
    pairs = [CreditPair(0,'a',0,1,2,1,'random'), CreditPair(1,'b',0,0,1,0,'random')]
    return model, view, pairs


def test_pair_softplus_detaches_omit_and_has_exact_weight():
    keep = torch.tensor([.2,3.], requires_grad=True)
    omit = torch.tensor([1.,1.], requires_grad=True)
    loss,w,utility = pair_terms(keep, omit, .7)
    dk, do = torch.autograd.grad(loss.sum(), (keep,omit), allow_unused=True)
    torch.testing.assert_close(dk,w)
    assert do is None
    torch.testing.assert_close(utility, omit.detach()-keep.detach())
    assert w[0] < .5 < w[1]
    equal, weight, _ = pair_terms(keep,keep,1.)
    torch.testing.assert_close(equal,torch.full_like(keep,math.log(2)))
    assert (weight == .5).all()
    for tau in (0.,-1.,float('nan')):
        with pytest.raises(ValueError): pair_terms(keep,omit,tau)


def test_source_vjp_exact_mask_and_no_second_order():
    p = torch.nn.Parameter(torch.tensor(2.))
    encoder = torch.nn.Parameter(torch.tensor(3.))
    k = (p*encoder*torch.ones(2,1,1,3)).clone()
    v = (p.square()*torch.ones(2,1,1,3)).clone()
    # Only (batch0,modality1) can contribute, despite loss depending on all.
    loss = k.square().sum()+v.square().sum()
    result, coverage = source_writer_vjp(loss,{0:(k,v)},[(0,0,1)],(p,))
    expected = 2*6*3+2*4*4
    assert result[0].item() == expected
    assert encoder.grad is None and p.grad is None
    assert not result[0].requires_grad
    assert coverage['connected_nodes'] == 2


def test_no_pairs_is_exact_normal_training_and_rng():
    model,view,_ = model_and_view()
    baseline = copy.deepcopy(model)
    rng = torch.get_rng_state()
    pred = baseline([view['incomplete']],view['availability'],view['qmask'],view['umask'],view['lengths'])[0]
    loss = ((pred.squeeze(-1)-view['labels'].T)[view['umask'].T.bool()]**2).mean()
    loss.backward(); after = torch.get_rng_state()
    torch.set_rng_state(rng)
    actual,task,stats = credit_backward(model,view,[],mode='E4')
    assert torch.equal(actual,pred) and torch.equal(task,loss)
    assert torch.equal(after,torch.get_rng_state())
    for (name,p),q in zip(model.named_parameters(),baseline.parameters()):
        assert (p.grad is None) == (q.grad is None),name
        if p.grad is not None: torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
    assert stats['omit_forwards'] == 0


def test_extra_gradients_only_writer_and_zero_init_honesty():
    model,view,pairs = model_and_view()
    baseline = copy.deepcopy(model)
    optim = torch.optim.Adam(model.parameters(),lr=.01)
    _,_,cold = credit_backward(model,view,pairs,mode='E4')
    assert cold['extra_gradient_norm'] == 0.
    assert cold['connected_nodes'] > 0
    optim.step();optim.zero_grad(set_to_none=True)
    baseline.load_state_dict(model.state_dict(),strict=True)
    rng = torch.get_rng_state()
    _,_,normal = credit_backward(baseline,view,[],mode='E0')
    after = torch.get_rng_state()
    torch.set_rng_state(rng)
    _,_,stats = credit_backward(model,view,pairs,mode='E4',lambda_fwc=.5)
    assert torch.equal(after,torch.get_rng_state())
    assert stats['extra_gradient_norm'] > 0
    changed = []
    for (name,p),q in zip(model.named_parameters(),baseline.parameters()):
        assert (p.grad is None) == (q.grad is None)
        if p.grad is None: continue
        assert torch.isfinite(p.grad).all()
        if not torch.equal(p.grad,q.grad): changed.append(name)
    assert changed and all(n.startswith(('osram.key_projectors.','osram.value_projectors.')) for n in changed)
    assert baseline.observed_set.projectors['audio'].fc1.weight.grad.abs().sum()>0
    assert baseline.smax_fc.weight.grad.abs().sum()>0
    assert stats['omit_forwards'] == 1


@pytest.mark.parametrize('mode',['E2','E3','E4','shuffled'])
def test_modes_share_counterfactual_budget_and_pair_identity(mode):
    model,view,pairs = model_and_view()
    _,_,stats = credit_backward(model,view,pairs,mode=mode)
    assert stats['pair_count']==2 and stats['omit_forwards']==1
    assert len(stats['pairs'])==2
    assert stats['omit_requires_grad'] is False


def test_multi_pair_per_conversation_separate_credit():
    model,view,pairs = model_and_view()
    pairs.append(CreditPair(0,'a',1,0,3,2,'random'))
    _,_,stats=credit_backward(model,view,pairs,mode='E4')
    assert stats['omit_forwards']==2
    assert [r['normalizer'] for r in stats['pairs']]==[.25,.5,.25]


def test_warmed_batched_vjp_equals_individual_source_credit():
    model,view,pairs=model_and_view()
    optimizer=torch.optim.SGD(model.parameters(),lr=.03)
    credit_backward(model,view,[],mode='E0')
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    nodes={}
    logits=forward_view(model,view,write_node_collector=nodes)
    losses=[(logits[p.future_j,p.batch,0]-view['labels'][p.batch,p.future_j]).square() for p in pairs]
    params=writer_parameters(model)
    grouped,_=source_writer_vjp(sum(losses),nodes,
        [(p.source_t,p.batch,p.source_modality) for p in pairs],params)
    individual=[]
    for p,loss in zip(pairs,losses):
        g,_=source_writer_vjp(loss,nodes,[(p.source_t,p.batch,p.source_modality)],params)
        individual.append(g)
    assert sum(float(g.abs().sum()) for g in grouped if g is not None)>0
    for p,actual,*expected in zip(params,grouped,*individual):
        total=sum((g for g in expected if g is not None),torch.zeros_like(p))
        torch.testing.assert_close(actual if actual is not None else torch.zeros_like(p),total,atol=1e-6,rtol=1e-5)


def test_full_gradient_control_reaches_classifier():
    model,view,pairs=model_and_view()
    baseline=copy.deepcopy(model)
    rng=torch.get_rng_state()
    credit_backward(baseline,view,[],mode='E0')
    torch.set_rng_state(rng)
    credit_backward(model,view,pairs,mode='E3')
    assert not torch.equal(model.smax_fc.weight.grad,baseline.smax_fc.weight.grad)
