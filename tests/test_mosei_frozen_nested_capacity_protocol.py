import copy
import pytest

from experiments.osram_mosei_frozen_nested_capacity_20261027 import protocol as p


def test_only_three_single_factor_structures_twentyfour_jobs():
    assert set(p.CONFIGS)=={'dim32','mlp128','depth2'}
    expected={'dim32':(32,3,64),'mlp128':(64,3,128),'depth2':(64,2,64)}
    for name,(dim,depth,mlp) in expected.items():
        c=p.CONFIGS[name]
        assert (c['dim'],c['depth'],c['mlp_hidden'],c['groups'])==(dim,depth,mlp,8)
    assert p.CONFIGS['dim32']['method']=='mosei_nested_d32_l3_g8_m64'
    jobs=p.jobs_for('train')
    assert len(jobs)==len({j['id'] for j in jobs})==24
    assert {(j['config'],j['rate']) for j in jobs}=={(c,r) for c in p.CONFIGS for r in p.RATES}
    assert len(p.jobs_for('smoke'))==3 and p.LR==3e-4 and p.EPOCHS==60
    with pytest.raises(ValueError):p.validate_scope('dim128','0.3')
    with pytest.raises(ValueError):p.validate_scope('dim32','0.8')


def test_exact_original_stage2_config_parent_and_mask_are_fixed():
    cfg={'dataset':'CMUMOSEI','learning_rate':3e-4,'batch_size':32}
    parent={'parent_id':'p'}
    ref=dict(status='complete',epoch=60,total_parameters=159235,parent_id='p',identity=dict(
        stage='frozen',mode='nested',method='nested_gnn_rooted_evidence',dataset='CMUMOSEI',seed=66,
        fold=1,evaluation_rate='0.3',smoke=False,stage2_epochs=60,cyclic_training=True,
        training_rates=list(p.RATES),mask_protocol='gcnet-random-mask-v1',parent=parent,
        test_mask_sha256='mask',effective_stage2_config=cfg))
    p.validate_reference(ref,parent,cfg,'mask','0.3')
    for changes in ({'learning_rate':1e-4},{'batch_size':64}):
        with pytest.raises(ValueError):p.validate_reference(ref,parent,dict(cfg,**changes),'mask','0.3')
    bad=copy.deepcopy(ref);bad['identity']['seed']=67
    with pytest.raises(ValueError):p.validate_reference(bad,parent,cfg,'mask','0.3')


def test_trace_lr_and_full_sample_order_are_enforced():
    old=dict(epoch=1,optimizer_steps=71,N=16326,mask_trace_sha256='x',training_rates=[0.,.1],learning_rate=3e-4)
    p.validate_epoch_trace(old,old,3e-4)
    with pytest.raises(ValueError):p.validate_epoch_trace(dict(old,learning_rate=1e-4),old,3e-4)
    with pytest.raises(ValueError):p.validate_epoch_trace(dict(old,mask_trace_sha256='y'),old,3e-4)
