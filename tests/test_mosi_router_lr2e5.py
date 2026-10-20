import copy
from types import SimpleNamespace

import pytest


def test_shared_lr_worker_accepts_2e5_without_changing_old_default():
    from experiments.osram_mosi_router_lr1e5_20261020.worker import set_epoch_lrs
    optimizer=SimpleNamespace(param_groups=[dict(name='backbone',lr=1.),dict(name='router',lr=1.)])
    for epoch in (0,99,100,149):
        set_epoch_lrs(optimizer,epoch,backbone_lr=2e-5)
        assert optimizer.param_groups[0]['lr']==2e-5
        assert optimizer.param_groups[1]['lr']==(3e-4 if epoch<100 else 3e-5)
    set_epoch_lrs(optimizer,0)
    assert optimizer.param_groups[0]['lr']==1e-5
    for invalid in (0.,-1.,float('nan'),float('inf')):
        with pytest.raises(ValueError):set_epoch_lrs(optimizer,0,backbone_lr=invalid)


def test_lr_reference_guard_allows_exactly_requested_2e5():
    from experiments.osram_mosi_router_lr1e5_20261020.worker import validate_reference
    from experiments.osram_mosi_gcnet_router_3seed_20261019.report import CORE
    identity=dict(config=dict(learning_rate=2e-5,seed=66),split={},mask_hashes={},seed_protocol={},
        source_hashes={n:'sha' for n in CORE})
    old=dict(status='complete',epoch=150,initial_backbone_sha256='init',identity=copy.deepcopy(identity))
    old['identity']['config']['learning_rate']=1e-3
    assert validate_reference(identity,old,'init',backbone_lr=2e-5)['only_backbone_lr_changed']
    with pytest.raises(ValueError,match='config'):validate_reference(identity,old,'init')
    identity['config']['seed']=67
    with pytest.raises(ValueError,match='config'):validate_reference(identity,old,'init',backbone_lr=2e-5)


def test_new_pair_routes_to_2e5_worker_not_old_run(tmp_path):
    from experiments.osram_mosi_router_lr2e5_20261020.run import plan_jobs
    from experiments.osram_mosi_router_lr2e5_20261020.worker import BACKBONE_LR
    assert BACKBONE_LR==2e-5
    jobs=plan_jobs(tmp_path,'sha',['GPU-healthy'])
    assert len(jobs)==2 and {(j['seed'],j['mode']) for j in jobs}=={(66,'flat'),(66,'joint')}
    assert all(j['command'][2]=='experiments.osram_mosi_router_lr2e5_20261020.worker' for j in jobs)
    assert all(j['checkpoint_selection']=='test-oracle-per-rate' for j in jobs)
