"""The new Nested queue is finite, recoverable, and preserves other workers."""
import copy
import importlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

PACKAGE = 'experiments.osram_frozen_nested_3seed_20261026'


def scheduler():
    assert importlib.util.find_spec(PACKAGE + '.scheduler') is not None, 'new finite queue is missing'
    return importlib.import_module(PACKAGE + '.scheduler')


def arguments(root, **overrides):
    return SimpleNamespace(**dict(dict(root=root, source_commit='locked', resume=False,
                                      gpu_map={0: 'GPU-a'}, per_gpu=4, poll_seconds=0), **overrides))


def identity(job, smoke=False):
    return dict(source_commit='locked', method='nested_gnn_rooted_evidence',
                dataset=job['dataset'], seed=job['seed'], fold=job['fold'],
                evaluation_rate=job['rate'], smoke=smoke)


def checkpoint(job, smoke=False):
    return dict(version=1, block={'weight': 1},
                optimizer=dict(state={}, param_groups=[dict(params=[0])]),
                rng=dict(python=(), numpy=(), torch=(), cuda=[]),
                identity=identity(job, smoke), epoch=1,
                history=[dict(epoch=1)], best=dict(epoch=1))


def test_exact_new_scope_and_priority():
    queue = scheduler()
    train, smoke = queue.jobs_for('train'), queue.jobs_for('smoke')
    assert len(train) == len({job['id'] for job in train}) == 128
    assert {job['seed'] for job in train} == {67, 68}
    assert {dataset: sum(job['dataset'] == dataset for job in train)
            for dataset in queue.DATASETS} == dict(CMUMOSI=16, CMUMOSEI=16, MELD=16, IEMOCAPSix=80)
    assert {(job['seed'], job['fold'], job['rate']) for job in train
            if job['dataset'] == 'IEMOCAPSix'} == {
                (seed, fold, str(rate / 10)) for seed in (67, 68)
                for fold in range(1, 6) for rate in range(8)}
    assert len(smoke) == 8 and {job['rate'] for job in smoke} == {'0.7'}
    assert {(job['dataset'], job['seed']) for job in smoke} == {
        (dataset, seed) for dataset in queue.DATASETS for seed in (67, 68)}
    assert all(job['fold'] == (5 if job['dataset'] == 'IEMOCAPSix' else 1) for job in smoke)
    assert {job['dataset'] for job in train[:4]} == {'CMUMOSEI', 'MELD'}
    with pytest.raises(ValueError):
        queue.jobs_for('unbounded')


def test_launch_pins_numeric_seed_snapshot_uuid_and_resume(tmp_path, monkeypatch):
    queue = scheduler()
    launched = []
    monkeypatch.setattr(queue, 'source_root', lambda dataset, seed: Path('/numeric') / dataset / str(seed))
    monkeypatch.setattr(queue.subprocess, 'Popen', lambda command, **kw:
                        launched.append((command, kw)) or SimpleNamespace(pid=123))
    job = queue.jobs_for('smoke')[1]
    queue.launch(arguments(tmp_path), job, 'GPU-a', 'smoke')
    command, kw = launched[-1]
    for flag, value in (('--seed', str(job['seed'])), ('--dataset', job['dataset']),
                        ('--fold', str(job['fold'])), ('--rate', job['rate']),
                        ('--gpu-uuid', 'GPU-a'), ('--source-commit', 'locked')):
        assert command[command.index(flag) + 1] == value
    assert PACKAGE + '.run' in command and '--smoke' in command and '--resume' not in command
    assert kw['cwd'] == f'/numeric/{job["dataset"]}/{job["seed"]}'
    assert kw['env']['CUDA_VISIBLE_DEVICES'] == 'GPU-a'
    assert kw['env']['CUDA_DEVICE_ORDER'] == 'PCI_BUS_ID' and kw['start_new_session']
    with pytest.raises((ValueError, RuntimeError)):
        queue.launch(arguments(tmp_path, resume=True), job, 'GPU-a', 'smoke')
    (Path(job['output']) / 'last.pt').write_bytes(b'validated before launch')
    queue.launch(arguments(tmp_path, resume=True), job, 'GPU-a', 'smoke')
    assert '--resume' in launched[-1][0]


def test_launch_failure_is_recorded_without_erasing_existing_workers(tmp_path, monkeypatch):
    queue = scheduler()
    monkeypatch.setattr(queue, 'source_root', lambda dataset, seed: tmp_path)
    def failed_spawn(*args, **kwargs):
        raise OSError('spawn failed')
    monkeypatch.setattr(queue.subprocess, 'Popen', failed_spawn)
    job = queue.jobs_for('smoke')[0]
    with pytest.raises(OSError, match='spawn failed'):
        queue.launch(arguments(tmp_path), job, 'GPU-a', 'smoke')
    assert job['status'] == 'failed' and job['command'] and job['output']


def test_only_gpu012_and_maximum_four_workers_with_resource_reserve():
    queue = scheduler()
    assert queue.parse_gpu_map('0=GPU-a,1=GPU-b,2=GPU-c') == {0: 'GPU-a', 1: 'GPU-b', 2: 'GPU-c'}
    for value in ('3=GPU-a', '4=GPU-a', '0=GPU-a,2=GPU-a', '0=GPU-a,0=GPU-b',
                  '0=GPU-46fb379f-cc90-dc82-9b5e-5d011f552264'):
        with pytest.raises(ValueError):
            queue.parse_gpu_map(value)
    snapshot = dict(gpus={'GPU-a': dict(free_mib=20000)}, ram_available_gib=40, disk_free_gib=30)
    peaks = {'MELD': dict(gpu_mib=1000, rss_gib=2)}
    job = dict(dataset='MELD')
    assert queue.fits(snapshot, 'GPU-a', job, [job] * 3, peaks, 4)
    assert not queue.fits(snapshot, 'GPU-a', job, [job] * 4, peaks, 4)
    assert not queue.fits(dict(snapshot, ram_available_gib=25), 'GPU-a', job, [], peaks, 4)
    assert not queue.fits(dict(snapshot, gpus={'GPU-a': dict(free_mib=1000)}), 'GPU-a', job, [], peaks, 4)


def test_reconcile_adopts_exact_live_worker_without_reading_checkpoint(tmp_path, monkeypatch):
    queue = scheduler()
    job = dict(queue.jobs_for('train')[0], status='running', pid=321, gpu_uuid='GPU-a')
    monkeypatch.setattr(queue, 'process_matches', lambda value: True)
    queue.reconcile([job], 'train', 'locked', tmp_path)
    assert job['status'] == 'running' and job['pid'] == 321


@pytest.mark.parametrize('phase', ['smoke', 'train'])
def test_full_block_adam_rng_history_recovery_includes_pending_outputs(tmp_path, phase):
    import torch
    queue = scheduler()
    job = dict(queue.jobs_for(phase)[0], status='pending')
    output = tmp_path / phase / job['id']
    output.mkdir(parents=True)
    torch.save(checkpoint(job, phase == 'smoke'), output / 'last.pt')
    queue.reconcile([job], phase, 'locked', tmp_path)
    assert job['status'] == 'pending' and job['output'] == str(output)
    assert 'Adam' in job['recovery'] and 'RNG' in job['recovery']


@pytest.mark.parametrize('corruption', ['seed', 'source', 'optimizer', 'rng', 'epoch', 'history'])
def test_incompatible_partial_recovery_is_failed(tmp_path, corruption):
    import torch
    queue = scheduler()
    job = dict(queue.jobs_for('train')[0], status='failed', output=str(tmp_path))
    saved = checkpoint(job)
    if corruption == 'seed':
        saved['identity']['seed'] = 66
    elif corruption == 'source':
        saved['identity']['source_commit'] = 'different'
    elif corruption in ('optimizer', 'rng'):
        saved.pop(corruption)
    elif corruption == 'epoch':
        saved['epoch'] = 999
    else:
        saved['history'] = []
    torch.save(saved, tmp_path / 'last.pt')
    queue.reconcile([job], 'train', 'locked')
    assert job['status'] == 'failed'


def test_existing_partial_output_cannot_be_silently_restarted(tmp_path):
    queue = scheduler()
    job = queue.jobs_for('train')[0]
    (tmp_path / 'train' / job['id']).mkdir(parents=True)
    queue.reconcile([job], 'train', 'locked', tmp_path)
    assert job['status'] == 'failed'


def test_recovery_validates_completion_identity_not_only_status(tmp_path):
    queue = scheduler()
    job = dict(queue.jobs_for('train')[0], status='running', output=str(tmp_path), log='worker.log')
    wrong = identity(job)
    wrong['seed'] = 66
    (tmp_path / 'SUMMARY.json').write_text(json.dumps(dict(status='complete', epoch=60,
        peak_memory_mib=1000, peak_rss_gib=2, total_parameters=159235, identity=wrong)))
    queue.reconcile([job], 'train', 'locked')
    assert job['status'] == 'failed'


def test_pid_reuse_requires_full_command_output_source_uid_and_cwd(tmp_path, monkeypatch):
    queue = scheduler()
    monkeypatch.setattr(queue, 'PROC', tmp_path)
    proc = tmp_path / '123'
    proc.mkdir()
    cwd = tmp_path / 'source'
    cwd.mkdir()
    (proc / 'cwd').symlink_to(cwd, target_is_directory=True)
    command = ['python', '-m', queue.MODULE, '--source-commit', 'locked', '--output', '/run']
    job = dict(pid=123, command=command, output='/run', source_commit='locked', cwd=str(cwd))
    (proc / 'cmdline').write_bytes(b'\0'.join(value.encode() for value in command) + b'\0')
    assert queue.process_matches(job)
    (proc / 'cmdline').write_bytes(b'python\0-m\0unrelated\0')
    assert not queue.process_matches(job)


def test_smoke_measurements_keep_both_seed_maxima_and_new_cycle_warmup(tmp_path):
    queue = scheduler()
    jobs = []
    for index, seed in enumerate((67, 68)):
        output = tmp_path / str(seed)
        output.mkdir()
        (output / 'SUMMARY.json').write_text(json.dumps(dict(
            peak_memory_mib=900 + index * 100, peak_rss_gib=3 + index)))
        jobs.append(dict(dataset='MELD', seed=seed, status='smoke_passed', summary=str(output/'SUMMARY.json')))
    peaks = queue.smoke_measurements(jobs)
    assert peaks['MELD'] == dict(gpu_mib=1000., rss_gib=4.)
    output = tmp_path / 'train'
    output.mkdir()
    job = dict(dataset='MELD', seed=67, status='running', output=str(output))
    (output/'history.json').write_text(json.dumps([dict(training_rates=[0., .1])]))
    assert not queue.refresh_measurements([job], peaks)['MELD']
    (output/'history.json').write_text(json.dumps([dict(training_rates=[index/10 for index in range(8)])]))
    assert queue.refresh_measurements([job], peaks)['MELD']


def test_failed_worker_stops_admission_and_drains_surviving_live_worker(tmp_path, monkeypatch):
    queue = scheduler()
    jobs = queue.jobs_for('train')[:3]
    jobs[0].update(status='running', pid=1, gpu_uuid='GPU-a')
    jobs[1].update(status='running', pid=2, gpu_uuid='GPU-a')
    state = {'train': jobs}
    round_number = [0]
    monkeypatch.setattr(queue, 'process_matches', lambda job: job['pid'] == 2 and round_number[0] == 0)
    def finish(job, code, phase):
        job['status'] = 'failed' if job['pid'] == 1 else 'complete'
    monkeypatch.setattr(queue, 'finish', finish)
    monkeypatch.setattr(queue.time, 'sleep', lambda seconds: round_number.__setitem__(0, round_number[0]+1))
    monkeypatch.setattr(queue, 'resources', lambda *args: pytest.fail('admitted after failure'))
    with pytest.raises(RuntimeError, match='failed|blocked'):
        queue.run_phase(arguments(tmp_path), state, 'train', {})
    assert [job['status'] for job in jobs] == ['failed', 'complete', 'blocked']
    assert json.loads((tmp_path/'QUEUE.json').read_text())['train'][1]['pid'] == 2


def test_baseexception_persists_live_workers_and_blocks_new_admission(tmp_path, monkeypatch):
    queue = scheduler()
    jobs = queue.jobs_for('train')[:2]
    jobs[0].update(status='running', pid=123, gpu_uuid='GPU-a')
    state = {'train': jobs}
    monkeypatch.setattr(queue, 'process_matches', lambda job: True)
    def interrupted(*args):
        raise KeyboardInterrupt('controller interrupted')
    monkeypatch.setattr(queue, 'resources', interrupted)
    with pytest.raises(KeyboardInterrupt):
        queue.run_phase(arguments(tmp_path), state, 'train', {})
    saved = json.loads((tmp_path/'QUEUE.json').read_text())
    assert saved['train'][0]['status'] == 'running' and saved['train'][0]['pid'] == 123
    assert saved['train'][1]['status'] == 'blocked'
    assert saved['preserved_live_workers'] == [jobs[0]['id']]


def test_real_cyclic_warmup_allows_only_one_new_worker_per_dataset(tmp_path, monkeypatch):
    queue = scheduler()
    jobs = queue.jobs_for('train')[:4]
    launched = []
    state = {'train': jobs}
    peaks = {dataset: dict(gpu_mib=1000., rss_gib=2.) for dataset in queue.DATASETS}
    monkeypatch.setattr(queue, 'resources', lambda *args: dict(
        gpus={'GPU-a': dict(free_mib=40000)}, ram_available_gib=100, disk_free_gib=100))
    monkeypatch.setattr(queue, 'refresh_measurements', lambda *args: dict.fromkeys(queue.DATASETS, False))
    def launch(args, job, uuid, phase):
        launched.append((job['dataset'], job['seed']))
        job.update(status='running', gpu_uuid=uuid, pid=len(launched))
        return SimpleNamespace(poll=lambda: None)
    monkeypatch.setattr(queue, 'launch', launch)
    def tick(seconds):
        if len(launched) == 2:
            raise KeyboardInterrupt('enough observations')
    monkeypatch.setattr(queue.time, 'sleep', tick)
    with pytest.raises(KeyboardInterrupt):
        queue.run_phase(arguments(tmp_path), state, 'train', peaks)
    assert launched == [('CMUMOSEI', 67), ('MELD', 67)]
    assert all(job['status'] == 'blocked' for job in jobs if job['seed'] == 68)


def test_scheduler_lock_blocks_duplicate_controller_before_gpu_checks(tmp_path, monkeypatch):
    import fcntl
    queue = scheduler()
    monkeypatch.setattr(queue, 'guard_gpu', lambda *args: pytest.fail('duplicate controller passed lock'))
    with (tmp_path/'scheduler.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            queue.main(['--root', str(tmp_path), '--source-commit', 'locked',
                        '--gpu-map', '0=GPU-a', '--resume'])


def test_guard_baseexception_restores_environment_and_saves_controller(tmp_path, monkeypatch):
    queue = scheduler()
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'preserved-environment')
    def interrupt(uuid, index):
        assert queue.os.environ['CUDA_VISIBLE_DEVICES'] == uuid
        raise KeyboardInterrupt('guard interruption')
    monkeypatch.setattr(queue, 'guard_gpu', interrupt)
    root = tmp_path/'execution'
    with pytest.raises(KeyboardInterrupt):
        queue.main(['--root', str(root), '--source-commit', 'locked', '--gpu-map', '0=GPU-a'])
    assert queue.os.environ['CUDA_VISIBLE_DEVICES'] == 'preserved-environment'
    saved = json.loads((root/'QUEUE.json').read_text())
    assert saved['status'] == 'failed' and saved['preserved_live_workers'] == []


def test_persisted_queue_cannot_expand_scope_or_duplicate_jobs():
    queue = scheduler()
    state = dict(source_commit='locked', method='nested_gnn_rooted_evidence',
                 expected_train_jobs=128, seeds=[67, 68], train=queue.jobs_for('train'))
    queue.validate_state(state, 'locked')
    altered = copy.deepcopy(state)
    altered['train'][0]['seed'] = 66
    with pytest.raises(ValueError):
        queue.validate_state(altered, 'locked')
    altered = copy.deepcopy(state)
    altered['train'][0] = copy.deepcopy(altered['train'][1])
    with pytest.raises(ValueError):
        queue.validate_state(altered, 'locked')
    altered = copy.deepcopy(state)
    altered['train'][0]['source_commit'] = 'different-worker-source'
    with pytest.raises(ValueError):
        queue.validate_state(altered, 'locked')
