"""Finite expansion of the fixed Nested128 candidate, without new tuning."""
import pytest


def runner():
    from experiments.osram_iemocap6_nested128_3seed_20261024 import run
    return run


def test_exact_fourteen_new_jobs_and_reused_fold(tmp_path):
    run = runner()
    jobs = run.plan_jobs(tmp_path, 'sha', run.GPU)
    expected = {(s, f) for s in (66, 67, 68) for f in range(1, 6)} - {(66, 5)}
    assert {(j['seed'], j['fold']) for j in jobs} == expected
    assert len(jobs) == 14
    assert [(j['seed'], j['fold']) for j in jobs[:6]] == [
        (66, 1), (67, 5), (68, 5), (66, 2), (67, 1), (68, 1)]
    for job in jobs:
        assert job['dataset'] == 'IEMOCAPSix'
        assert job['method'] == 'nested_dim128'
        assert job['epochs'] == 100 and job['learning_rate'] == 1e-3
        assert job['mask_protocol'] == 'gcnet-random-mask-v1'
        assert job['command'][0] == '/data2/yb/conda_envs/cmerc-repro/bin/python'
        assert '--resume' not in job['command']
        assert job['gpu_uuid'] == run.GPU


def test_reservations_allow_five_not_six_without_double_counting():
    run = runner()
    resources = dict(total_mib=32768, used_mib=4 * 5536, own_used_mib=4 * 5536,
                     active_count=4, ram_gib=128, disk_gib=100, parallel=6)
    assert run.can_admit(**resources)
    resources.update(used_mib=5 * 5536, own_used_mib=5 * 5536, active_count=5)
    assert not run.can_admit(**resources)
    resources.update(used_mib=32768 - 500, own_used_mib=4 * 5536, active_count=4)
    assert not run.can_admit(**resources)


def test_resource_bounds_and_gpu_authorization():
    run = runner()
    resources = dict(total_mib=32768, used_mib=0, own_used_mib=0,
                     active_count=0, ram_gib=64, disk_gib=25, parallel=6)
    assert run.can_admit(**resources)
    assert not run.can_admit(**dict(resources, ram_gib=63))
    assert not run.can_admit(**dict(resources, disk_gib=24))
    with pytest.raises(ValueError):
        run.can_admit(**dict(resources, parallel=7))
    with pytest.raises(ValueError):
        run.validate_gpu_uuid(run.BAD_UUID)
    with pytest.raises(ValueError):
        run.validate_gpu_uuid('GPU-unverified')


def test_actual_own_allocations_above_reservation_remain_charged():
    run = runner()
    assert not run.can_admit(total_mib=32768, used_mib=28000, own_used_mib=28000,
                             active_count=4, ram_gib=128, disk_gib=100, parallel=6)


def test_only_own_pid_allocations_are_reclaimed(monkeypatch):
    run = runner()
    monkeypatch.setattr(run.subprocess, 'check_output', lambda *args, **kwargs:
        f'101, {run.GPU}, 3000\n202, {run.GPU}, 10000\n101, GPU-other, 777\n')
    assert run.own_gpu_memory_mib(run.GPU, [101, 303]) == 3000


def test_recovery_rejects_live_workers_and_never_retries_failures(monkeypatch):
    run = runner()
    monkeypatch.setattr(run.os, 'kill', lambda pid, signal: None)
    with pytest.raises(RuntimeError, match='still alive'):
        run.recover_jobs({'jobs': [{'status': 'running', 'pid': 123}]})
    with pytest.raises(RuntimeError, match='automatically retried'):
        run.recover_jobs({'jobs': [{'status': 'failed'}]})
