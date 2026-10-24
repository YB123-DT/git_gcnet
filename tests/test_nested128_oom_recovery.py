import copy
import pytest


def fixture():
    return {'jobs': [
        {'status': 'complete', 'seed': 66, 'fold': 1},
        {'status': 'failed', 'seed': 68, 'fold': 5, 'output': '/failed',
         'command': ['python', 'worker'], 'pid': 123, 'error': 'exit 1'},
        {'status': 'blocked', 'seed': 68, 'fold': 1, 'output': '/pending',
         'command': ['python', 'worker'], 'error': 'Earlier child failed'},
    ]}


def recovery():
    from experiments.osram_iemocap6_nested128_3seed_20261024 import recover_oom
    return recover_oom


def test_only_failed_worker_resumes_and_completed_results_unchanged():
    state = fixture()
    original = copy.deepcopy(state)
    seen = []
    repaired = recovery().prepare(state, lambda job: seen.append(job['output']))
    assert state == original
    assert repaired['jobs'][0] == original['jobs'][0]
    assert seen == ['/failed']
    assert repaired['jobs'][1]['command'][-1] == '--resume'
    assert repaired['jobs'][1]['status'] == 'pending'
    assert repaired['jobs'][2]['status'] == 'pending'
    assert '--resume' not in repaired['jobs'][2]['command']
    assert repaired['oom_recovery']['maximum_parallel'] == 3


def test_recovery_is_bounded_to_one_known_failure():
    state = fixture()
    state['oom_recovery'] = {'attempt': 1}
    with pytest.raises(ValueError, match='already'):
        recovery().prepare(state, lambda job: None)
    state = fixture()
    state['jobs'][1]['seed'] = 67
    with pytest.raises(ValueError, match='Unexpected'):
        recovery().prepare(state, lambda job: None)


def test_invalid_checkpoint_never_requeues():
    state = fixture()
    def reject(job):
        raise ValueError('Missing CUDA RNG')
    with pytest.raises(ValueError, match='CUDA RNG'):
        recovery().prepare(state, reject)
    assert state['jobs'][1]['status'] == 'failed'


def test_resource_policy_caps_three_and_reserves_full_context():
    r = recovery()
    resources = dict(total_mib=32768, used_mib=14000, own_used_mib=14000,
                     active_count=2, ram_gib=128, disk_gib=100, parallel=6)
    assert r.can_admit(**resources)
    assert not r.can_admit(**dict(resources, active_count=3))
    assert not r.can_admit(**dict(resources, used_mib=25000))
    assert not r.can_admit(**dict(resources, ram_gib=63))
