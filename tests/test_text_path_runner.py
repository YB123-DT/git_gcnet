import importlib

import numpy as np
import pytest
import torch


def api():
    try:
        return importlib.import_module('experiments.osram_text_path_router_20261019.run')
    except ModuleNotFoundError as error:
        pytest.fail('Missing Router-only experiment runner: '+str(error))


def record(eligible):
    from experiments.osram_text_path_router_20261019.router import TextPathRouter
    router=TextPathRouter(local_dim=4,context_dim=6,projection_dim=3,hidden_dim=5)
    n=len(eligible)
    item=dict(local=torch.randn(n,4),base_text=torch.randn(n,6),gap_text=torch.randn(n,6),
        text_trajectory=torch.tensor([[2.,.5,1.,1.]]).expand(n,-1),
        candidates=torch.randn(n,3,6),eligible=torch.tensor(eligible),labels=torch.arange(n)%6,
        availability=torch.tensor([[1.,0.,0.]]).expand(n,-1),
        meta=dict(conversation_ids=np.array(['a']*n),utterance_indices=np.arange(n),speakers=np.zeros(n)))
    return router,item


def test_no_eligible_training_batch_does_not_apply_weight_decay_or_adam_step():
    module=api();router,item=record([False,False])
    opt=torch.optim.Adam(router.parameters(),lr=.01,weight_decay=.1)
    before={k:v.clone() for k,v in router.state_dict().items()}
    report=module.optimize_record(router,opt,item,1.)
    assert report['optimizer_steps']==0 and report['eligible']==0
    assert not opt.state
    assert all(torch.equal(v,router.state_dict()[k]) for k,v in before.items())


def test_training_loss_is_original_sample_mean_nll_and_router_updates():
    module=api();router,item=record([True,False,True])
    opt=torch.optim.Adam(router.parameters(),lr=.01,weight_decay=0)
    with torch.no_grad():
        expected=torch.nn.functional.nll_loss(router(**{k:item[k] for k in module.ROUTER_KEYS})[0],item['labels'])
    before=router.network[-1].weight.clone()
    # Discover the final layer through the state dict, not any training label input.
    report=module.optimize_record(router,opt,item,1.)
    assert report['loss']==pytest.approx(float(expected))
    assert report['optimizer_steps']==1 and report['eligible']==2
    assert report['nonzero_gradient_parameters']>0
    assert not torch.equal(before,router.network[-1].weight)


def test_predictions_keep_noneligible_classes_and_report_six_class_metrics():
    module=api();router,item=record([True,False,True])
    arrays=module.predict_records(router,[item])
    np.testing.assert_array_equal(arrays['log_probs_router'][1].argmax(),item['candidates'][1,0].argmax())
    result=module.summarize_predictions(arrays,6)
    assert result['eligible_N']==2 and result['full']['N']==3
    assert len(result['router']['class_recall'])==6
    pair=result['paired_vs_full']
    assert pair['corrections']-pair['harms']==int((arrays['log_probs_router'].argmax(-1)==arrays['labels']).sum())-int((arrays['candidates'][:,0].argmax(-1)==arrays['labels']).sum())


def test_validation_key_is_registered_acc_then_ce_not_test_ua():
    module=api()
    good={str(i):{'router':{'ACC':60.,'CE':1.}} for i in range(8)}
    other={str(i):{'router':{'ACC':59.,'CE':.1}} for i in range(8)}
    assert module.selection_key(good)>module.selection_key(other)


def test_report_labels_fixed_anchor_and_includes_all_rates(tmp_path):
    module=api();router,item=record([True,False,True])
    metric=module.summarize_predictions(module.predict_records(router,[item]),6)
    try:
        from experiments.osram_text_path_router_20261019.report import write_report
    except ModuleNotFoundError as error:
        pytest.fail('Missing complete-result report: '+str(error))
    rates={f'{r/10:.1f}':metric for r in range(8)}
    aggregate={tag:{scope:{m:metric[tag][m] for m in ('ACC','UA','WF1','CE')} for scope in ('mean','high')}
               for tag in ('full','router','uniform','candidate_B','candidate_G')}
    summary=dict(dataset='IEMOCAPSix',source_commit='test',selected_epoch=2,router_parameters=95439,
        seconds=1.,peak_memory_mib=1.,identity={'case':{'fold':5,'epoch':69}},aggregate=aggregate,evaluation=rates)
    write_report(tmp_path,summary)
    text=(tmp_path/'RESULT.md').read_text()
    assert 'INTERNAL' in text and 'fold5' in text and '0.7' in text
    assert (tmp_path/'METRICS.csv').exists()


def test_complete_router_recovery_reproduces_the_next_optimizer_step(tmp_path):
    import random
    module=api();router,item=record([True,True,False])
    optimizer=torch.optim.Adam(router.parameters(),lr=.01,weight_decay=.001)
    module.optimize_record(router,optimizer,item,1.)
    state=module.checkpoint(router,optimizer,1,{'id':'same'},[],(60.,-1.))
    torch.save(state,tmp_path/'recovery.pt')
    expected_random=(random.random(),float(np.random.rand()),torch.rand(3))
    module.optimize_record(router,optimizer,item,1.)
    expected={k:v.clone() for k,v in router.state_dict().items()}
    state=torch.load(tmp_path/'recovery.pt',weights_only=False)
    module.restore(router,optimizer,state,{'id':'same'})
    assert random.random()==expected_random[0]
    assert float(np.random.rand())==expected_random[1]
    assert torch.equal(torch.rand(3),expected_random[2])
    module.optimize_record(router,optimizer,item,1.)
    assert all(torch.equal(v,router.state_dict()[k]) for k,v in expected.items())
    with pytest.raises(ValueError):module.restore(router,optimizer,state,{'id':'wrong'})
