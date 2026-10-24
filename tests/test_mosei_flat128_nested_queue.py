"""CPU-only contracts for the finite Flat128 / Nested screening queue."""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def runner():
    module = 'experiments.osram_mosei_flat128_nested_20261024.run'
    try:
        spec = importlib.util.find_spec(module)
    except ModuleNotFoundError:
        spec = None
    assert spec is not None, 'The finite Flat128 Nested queue is missing'
    return importlib.import_module(module)


def proof():
    return dict(status='passed', default_reserve_mib=4096,
                variants={'N01': {'reserve_mib': 4096}, 'N20': {'reserve_mib': 4096},
                          'N21': {'reserve_mib': 5120}})


def test_exact_fresh_plan_initial_placement_and_worker_protocol(tmp_path):
    run = runner()
    jobs = run.plan_jobs(tmp_path, 'sha', proof())
    assert len(jobs) == 23
    assert {j['slot_id'] for j in jobs} == {'F67', 'F68'} | {f'N{i:02d}' for i in range(1, 22)}
    assert [j['slot_id'] for j in jobs[:12]] == [
        'F67', 'F68', 'N19', 'N20', 'N01', 'N02',
        'N03', 'N04', 'N05', 'N06', 'N07', 'N08']
    assert [j['preferred_gpu_uuid'] for j in jobs[:12]] == [run.GPU0] * 6 + [run.GPU1] * 6
    assert len({j['output'] for j in jobs}) == 23
    for job in jobs:
        assert job['seed'] == (int(job['slot_id'][1:]) if job['slot_id'].startswith('F') else 66)
        assert job['epochs'] == 60 and job['learning_rate'] == 1e-4
        assert job['dataset'] == 'CMUMOSEI' and job['fold'] == 1
        assert job['command'][0] == run.PYTHON
        assert job['command'][job['command'].index('--variant') + 1] == job['slot_id']
        assert job['command'][job['command'].index('--gpu-uuid') + 1] == job['gpu_uuid']
        assert '--resume' not in job['command']
        assert job['reserve_mib'] == (5120 if job['slot_id'] == 'N21' else 4096)


def test_proof_requires_all_measured_anchors_and_immutable_bytes(tmp_path):
    run = runner()
    path = tmp_path / 'proof.json'
    path.write_text(json.dumps(proof()))
    loaded, digest = run.load_resource_proof(path)
    assert loaded == proof()
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    for change in ('status', 'missing_anchor', 'nonfinite', 'underreserve'):
        bad = proof()
        if change == 'status': bad['status'] = 'running'
        elif change == 'missing_anchor': del bad['variants']['N01']
        elif change == 'nonfinite': bad['variants']['N21']['reserve_mib'] = float('nan')
        else: bad['default_reserve_mib'] = 0
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError): run.load_resource_proof(path)


def test_six_workers_fit_measured_reservations_without_double_charging():
    run = runner()
    resources = dict(total_mib=32768, used_mib=5 * 3000, own_used_mib=5 * 3000,
                     active_reserve_mib=5 * 4096, proposed_mib=4096, active_count=5,
                     global_active_count=10, ram_gib=64, disk_gib=25)
    assert run.can_admit(**resources)
    for changed in ({'active_count': 6}, {'global_active_count': 12},
                    {'ram_gib': 63.99}, {'disk_gib': 24.99}, {'total_mib': float('nan')},
                    {'used_mib': 31000}, {'own_used_mib': 29000, 'used_mib': 29000},
                    {'active_reserve_mib': 5 * 6144}):
        assert not run.can_admit(**dict(resources, **changed))
    assert not run.can_admit(**dict(resources, proposed_mib=12000))


def test_uuid_authorization_and_dynamic_backfill_command_binding(tmp_path):
    run = runner()
    for uuid in (run.GPU0, run.GPU1): run.validate_gpu_uuid(uuid)
    for uuid in (run.BAD_UUID, '0', 'GPU-unknown'):
        with pytest.raises(ValueError): run.validate_gpu_uuid(uuid)
    job = run.plan_jobs(tmp_path, 'sha', proof())[12]
    run.bind_gpu(job, run.GPU1)
    assert job['gpu_uuid'] == run.GPU1
    assert job['command'][job['command'].index('--gpu-uuid') + 1] == run.GPU1


def test_old_capacity_workers_are_counted_until_they_finish(tmp_path, monkeypatch):
    run = runner()
    path = tmp_path / 'QUEUE.json'
    old = dict(status='running', jobs=[
        dict(status='running', pid=12, gpu_uuid=run.GPU1, reserve_mib=6144,
             output=str(tmp_path / 'old'), command=['python', '-m', 'old'])])
    path.write_text(json.dumps(old))
    monkeypatch.setattr(run, 'recorded_worker_alive', lambda job: True)
    active = run.external_workers(path)
    assert len(active) == 1 and active[0]['pid'] == 12 and active[0]['reserve_mib'] == 6144
    monkeypatch.setattr(run, 'recorded_worker_alive', lambda job: False)
    with pytest.raises(RuntimeError, match='without completed'):
        run.external_workers(path)
    output = tmp_path / 'old'; output.mkdir()
    (output / 'SUMMARY.json').write_text(json.dumps({'status': 'complete', 'epoch': 60}))
    assert run.external_workers(path) == []


def test_recovery_refuses_living_workers_and_failed_runs(monkeypatch):
    run = runner()
    monkeypatch.setattr(run.common.os, 'kill', lambda pid, signal: None)
    with pytest.raises(RuntimeError, match='still alive'):
        run.common.recover_jobs({'jobs': [{'status': 'running', 'pid': 123}]})
    with pytest.raises(RuntimeError, match='automatically retried'):
        run.common.recover_jobs({'jobs': [{'status': 'failed'}]})


def setup_controller(tmp_path, monkeypatch):
    run = runner()
    resource = tmp_path / 'proof.json'; resource.write_text(json.dumps(proof()))
    monkeypatch.setattr(run.common, 'verify_source_snapshot', lambda root, commit:
                        {'manifest_sha256': 'manifest', 'source_commit': commit})
    monkeypatch.setattr(run.config, 'verify_registry_extension', lambda root: {'status': 'passed'})
    monkeypatch.setattr(run.common.base, 'devices', lambda uuids: {
        uuid: dict(index=i, total_mib=32768, used_mib=0)
        for i, uuid in enumerate((run.GPU0, run.GPU1))})
    monkeypatch.setattr(run, 'external_workers', lambda: [])
    monkeypatch.setattr(run, 'own_gpu_memory_mib', lambda uuid, pids: 0)
    monkeypatch.setattr(run.common, 'available_gib', lambda: 128)
    monkeypatch.setattr(run.shutil, 'disk_usage', lambda path: SimpleNamespace(free=100 * 2**30))
    monkeypatch.setattr(run.time, 'sleep', lambda seconds: None)
    return run, resource, ['--output', str(tmp_path / 'queue'), '--source-commit', 'sha',
                           '--resource-proof', str(resource)]


def test_failure_stops_new_admissions_and_drains_without_signals(tmp_path, monkeypatch):
    run, resource, argv = setup_controller(tmp_path, monkeypatch)
    launched = []
    class Process:
        pid = 123
        def wait(self): return 1
    def launch(job):
        job.update(status='running', pid=123)
        launched.append(job['slot_id'])
        return Process()
    def poll(process, job):
        job['status'] = 'failed'
        return True
    monkeypatch.setattr(run.common.base, 'launch', launch)
    monkeypatch.setattr(run.common, 'poll', poll)
    with pytest.raises(RuntimeError, match='failed'):
        run.main(argv, reporter=lambda *args, **kwargs: None,
                 reused_validator=lambda path: {'status': 'passed'})
    state = json.loads((tmp_path / 'queue/QUEUE.json').read_text())
    assert len(launched) == 1
    assert state['status'] == 'failed'
    assert sum(j['status'] == 'blocked' for j in state['jobs']) == 22
    assert state['identity']['resource_proof_sha256'] == hashlib.sha256(resource.read_bytes()).hexdigest()
    assert state['reused']['slot_id'] == 'F66'


def test_complete_only_after_23_fresh_and_reused_slot_and_source_checks(tmp_path, monkeypatch):
    run, resource, argv = setup_controller(tmp_path, monkeypatch)
    launched = []
    final_calls = []
    class Process:
        def __init__(self, pid): self.pid = pid
    def launch(job):
        launched.append(job['slot_id'])
        job.update(status='running', pid=len(launched))
        return Process(len(launched))
    def poll(process, job):
        job['status'] = 'complete'
        return True
    def report(output, state, final=False):
        if final:
            assert sum(j['status'] == 'complete' for j in state['jobs']) == 23
            assert state['reused']['status'] == 'reused_complete'
            final_calls.append(True)
    monkeypatch.setattr(run.common.base, 'launch', launch)
    monkeypatch.setattr(run.common, 'poll', poll)
    run.main(argv, reporter=report, reused_validator=lambda path: {'status': 'passed'})
    state = json.loads((tmp_path / 'queue/QUEUE.json').read_text())
    assert len(launched) == 23 and len(set(launched)) == 23
    assert final_calls == [True] and state['status'] == 'complete'
    assert state['maximum_new_jobs'] == 23 and state['total_result_slots'] == 24
    with pytest.raises(FileExistsError):
        run.main(argv, reporter=report, reused_validator=lambda path: {'status': 'passed'})


def test_interrupted_controller_can_explicitly_resume_without_repeating_completed_jobs(tmp_path, monkeypatch):
    run, resource, argv = setup_controller(tmp_path, monkeypatch)
    launched = []
    class Process:
        pid = 123
        def wait(self): return 0
    def launch(job):
        launched.append(job['slot_id'])
        job.update(status='running', pid=123)
        return Process()
    def poll(process, job):
        job['status'] = 'complete'
        return True
    def interrupt(seconds): raise KeyboardInterrupt()
    monkeypatch.setattr(run.common.base, 'launch', launch)
    monkeypatch.setattr(run.common, 'poll', poll)
    monkeypatch.setattr(run.time, 'sleep', interrupt)
    kwargs = dict(reporter=lambda *args, **kwargs: None,
                  reused_validator=lambda path: {'status': 'passed'})
    with pytest.raises(KeyboardInterrupt): run.main(argv, **kwargs)
    state = json.loads((tmp_path / 'queue/QUEUE.json').read_text())
    assert state['status'] == 'interrupted'
    assert sum(j['status'] == 'pending' for j in state['jobs']) == 22
    monkeypatch.setattr(run.time, 'sleep', lambda seconds: None)
    recovered = []
    monkeypatch.setattr(run.common, 'recover_jobs', lambda state: recovered.append(True))
    run.main(argv + ['--resume'], **kwargs)
    assert recovered == [True]
    assert len(launched) == 23 and len(set(launched)) == 23


def test_changed_proof_cannot_resume_the_same_output_identity(tmp_path, monkeypatch):
    run, resource, argv = setup_controller(tmp_path, monkeypatch)
    class Process:
        pid = 123
        def wait(self): return 0
    def launch(job):
        job.update(status='running', pid=123)
        return Process()
    def interrupt(seconds): raise KeyboardInterrupt()
    def poll(process, job):
        job['status'] = 'complete'
        return True
    monkeypatch.setattr(run.common.base, 'launch', launch)
    monkeypatch.setattr(run.common, 'poll', poll)
    monkeypatch.setattr(run.time, 'sleep', interrupt)
    kwargs = dict(reporter=lambda *args, **kwargs: None,
                  reused_validator=lambda path: {'status': 'passed'})
    with pytest.raises(KeyboardInterrupt): run.main(argv, **kwargs)
    resource.write_text(json.dumps(dict(proof(), measurement_revision=2)))
    with pytest.raises(ValueError, match='identity'):
        run.main(argv + ['--resume'], **kwargs)


def test_source_mutation_stops_admission_before_launch(tmp_path, monkeypatch):
    run, resource, argv = setup_controller(tmp_path, monkeypatch)
    manifests = iter([{'manifest_sha256': 'manifest', 'source_commit': 'sha'},
                      {'manifest_sha256': 'different', 'source_commit': 'sha'}])
    monkeypatch.setattr(run.common, 'verify_source_snapshot', lambda *args: next(manifests))
    launched = []
    monkeypatch.setattr(run.common.base, 'launch', lambda job: launched.append(job))
    with pytest.raises(ValueError, match='Immutable source changed'):
        run.main(argv, reporter=lambda *args, **kwargs: None,
                 reused_validator=lambda path: {'status': 'passed'})
    assert launched == []


def test_three_existing_gpu1_workers_reduce_new_admission_to_nine(tmp_path, monkeypatch):
    run, resource, argv = setup_controller(tmp_path, monkeypatch)
    old = [dict(pid=i, gpu_uuid=run.GPU1, reserve_mib=6144) for i in range(90, 93)]
    monkeypatch.setattr(run, 'external_workers', lambda: old)
    launched = []
    draining = []
    class Process:
        def __init__(self, pid): self.pid = pid
        def wait(self): return 0
    def launch(job):
        job.update(status='running', pid=len(launched) + 1)
        launched.append(job['gpu_uuid'])
        return Process(len(launched))
    def poll(process, job):
        if not draining: return False
        job['status'] = 'complete'
        return True
    def stop_when_full(seconds):
        if seconds == 15:
            draining.append(True)
            raise RuntimeError('simulation reached admission ceiling')
    monkeypatch.setattr(run.common.base, 'launch', launch)
    monkeypatch.setattr(run.common, 'poll', poll)
    monkeypatch.setattr(run.time, 'sleep', stop_when_full)
    with pytest.raises(RuntimeError, match='admission ceiling'):
        run.main(argv, reporter=lambda *args, **kwargs: None,
                 reused_validator=lambda path: {'status': 'passed'})
    assert launched.count(run.GPU0) == 6 and launched.count(run.GPU1) == 3


def test_memory_spike_does_not_consume_another_loading_workers_reservation(tmp_path, monkeypatch):
    run, resource, argv = setup_controller(tmp_path, monkeypatch)
    allocations = {1: 7000, 2: 1000}
    monkeypatch.setattr(run, 'own_gpu_memory_mib', lambda uuid, pids: allocations[pids[0]])
    active = [dict(pid=pid, gpu_uuid=run.GPU0, reserve_mib=4096) for pid in (1, 2)]
    resources = run._resources(run.GPU0, active, tmp_path)
    assert resources['own_used_mib'] == 8000
    assert resources['active_reserve_mib'] == 7000 + 4096


def test_resume_validates_persisted_commands_and_allows_only_recorded_gpu_rebinding(tmp_path):
    run = runner()
    planned = run.plan_jobs(tmp_path, 'sha', proof())
    jobs = json.loads(json.dumps(planned))
    run.bind_gpu(jobs[12], run.GPU1)
    jobs[12]['command'].append('--resume')
    run.validate_persisted_plan(jobs, planned)
    for changed in ('output', 'reserve_mib', 'seed', 'command', 'missing_job'):
        wrong = json.loads(json.dumps(jobs))
        if changed == 'missing_job': wrong.pop()
        elif changed == 'command': wrong[0]['command'].append('--unsafe')
        elif changed == 'reserve_mib': wrong[0][changed] = 1
        elif changed == 'seed': wrong[0][changed] = 123
        else: wrong[0][changed] = '/different/output'
        with pytest.raises(ValueError, match='Persisted'):
            run.validate_persisted_plan(wrong, planned)
