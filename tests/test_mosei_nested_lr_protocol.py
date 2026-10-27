import copy
import pytest

from experiments.osram_mosei_frozen_nested_lr_20261027 import protocol as p


def test_only_two_lrs_and_sixteen_own_rate_jobs():
    assert p.LRS == {'lr1e4':1e-4,'lr1e3':1e-3}
    jobs=p.jobs_for('train')
    assert len(jobs)==len({j['id'] for j in jobs})==16
    assert {(j['config'],j['rate']) for j in jobs}=={(c,r) for c in p.LRS for r in p.RATES}
    assert len(p.jobs_for('smoke'))==2
    for config,rate in [('unknown','0.3'),('lr1e4','0.8'),('lr1e3',True)]:
        with pytest.raises(ValueError):p.validate_scope(config,rate)


def reference():
    parent={'parent_id':'original'}
    cfg={'learning_rate':3e-4,'batch_size':32,'dataset':'CMUMOSEI'}
    ref=dict(status='complete',epoch=60,total_parameters=159235,parent_id='original',
        identity=dict(stage='frozen',mode='nested',method=p.METHOD,dataset='CMUMOSEI',seed=66,
            fold=1,evaluation_rate='0.3',smoke=False,stage2_epochs=60,cyclic_training=True,
            training_rates=list(p.RATES),mask_protocol='gcnet-random-mask-v1',parent=parent,
            test_mask_sha256='mask',effective_stage2_config=cfg))
    return ref,parent,dict(cfg,learning_rate=1e-4)


def test_reference_allows_only_branch_lr_to_change():
    ref,parent,cfg=reference()
    p.validate_reference(ref,parent,cfg,'mask','0.3')
    for key,value in [('batch_size',64),('dataset','CMUMOSI')]:
        changed=dict(cfg,**{key:value})
        with pytest.raises(ValueError):p.validate_reference(ref,parent,changed,'mask','0.3')
    with pytest.raises(ValueError):p.validate_reference(ref,{'parent_id':'wrong'},cfg,'mask','0.3')
    with pytest.raises(ValueError):p.validate_reference(ref,parent,cfg,'new-mask','0.3')
    bad=copy.deepcopy(ref);bad['identity']['seed']=67
    with pytest.raises(ValueError):p.validate_reference(bad,parent,cfg,'mask','0.3')


def test_trace_checks_sampling_separately_from_new_lr():
    old=dict(epoch=1,optimizer_steps=100,training_rates=[0.,.1,.2],mask_trace_sha256='same',N=999,learning_rate=3e-4)
    new=dict(old,learning_rate=1e-4)
    p.validate_epoch_trace(new,old,1e-4)
    with pytest.raises(ValueError):p.validate_epoch_trace(new,old,1e-3)
    for k,v in [('N',998),('mask_trace_sha256','different'),('training_rates',[0.,.1])]:
        with pytest.raises(ValueError):p.validate_epoch_trace(dict(new,**{k:v}),old,1e-4)
