import copy
import importlib
from types import SimpleNamespace

import pytest


def worker():
    return importlib.import_module('experiments.osram_mosi_router_lr1e5_20261020.worker')


def test_backbone_stays_1e5_and_original_router_schedule_is_unchanged():
    module = worker()
    optimizer = SimpleNamespace(param_groups=[dict(name='backbone', lr=.001), dict(name='router', lr=.0003)])
    for epoch in (0, 99, 100, 149):
        module.set_epoch_lrs(optimizer, epoch)
        assert optimizer.param_groups[0]['lr'] == 1e-5
        assert optimizer.param_groups[1]['lr'] == (3e-4 if epoch < 100 else 3e-5)
    with pytest.raises(ValueError):
        module.set_epoch_lrs(optimizer, 150)
    with pytest.raises(ValueError):
        module.set_epoch_lrs(SimpleNamespace(param_groups=[dict(name='unknown', lr=1.)]), 0)


def test_reference_allows_only_lr_change_and_requires_original_initialization():
    module = worker()
    from experiments.osram_mosi_gcnet_router_3seed_20261019.report import CORE
    identity = dict(config=dict(learning_rate=1e-5, epochs=150, missing_mask_protocol='gcnet-random-mask-v1'),
                    split={'train': 'same'}, mask_hashes={'test': 'same'}, seed_protocol={'seed': 66},
                    source_hashes={n: 'sha' for n in CORE})
    reference = dict(status='complete', epoch=150, initial_backbone_sha256='initial',
                     identity=copy.deepcopy(identity))
    reference['identity']['config']['learning_rate'] = .001
    assert module.validate_reference(identity, reference, 'initial')['only_backbone_lr_changed']
    with pytest.raises(ValueError, match='initial'):
        module.validate_reference(identity, reference, 'other')
    changed = copy.deepcopy(identity)
    changed['config']['missing_mask_protocol'] = 'conversation-mask-v2'
    with pytest.raises(ValueError, match='config'):
        module.validate_reference(changed, reference, 'initial')
    changed = copy.deepcopy(identity)
    changed['mask_hashes'] = {'test': 'different'}
    with pytest.raises(ValueError, match='mask_hashes'):
        module.validate_reference(changed, reference, 'initial')


def test_training_mask_check_does_not_demand_same_losses_or_learning_rates():
    module = worker()
    old = dict(epoch=1, mask_trace_sha256='mask', optimizer_steps=4, task_MSE=2., learning_rates=[.001])
    actual = dict(old, task_MSE=3., learning_rates=[1e-5])
    assert module.check_mask_parity(actual, old)['training_masks_exact']
    actual['mask_trace_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='mask'):
        module.check_mask_parity(actual, old)


def test_plan_is_only_seed66_flat_and_router_on_one_healthy_gpu(tmp_path):
    from experiments.osram_mosi_router_lr1e5_20261020.run import plan_jobs
    jobs = plan_jobs(tmp_path, 'sha', ['GPU-healthy'])
    assert len(jobs) == 2
    assert {(j['seed'], j['mode']) for j in jobs} == {(66, 'flat'), (66, 'joint')}
    assert all(j['gpu_uuid'] == 'GPU-healthy' for j in jobs)
    assert all(j['checkpoint_selection'] == 'test-oracle-per-rate' for j in jobs)
    with pytest.raises(ValueError):
        plan_jobs(tmp_path, 'sha', ['GPU-46fb379f-cc90-dc82-9b5e-5d011f552264'])


def test_report_refuses_incomplete_or_expanded_scope(tmp_path):
    from experiments.osram_mosi_router_lr1e5_20261020.report import write_report
    with pytest.raises(ValueError, match='both completed'):
        write_report(tmp_path, {'jobs': []})
