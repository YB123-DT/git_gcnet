"""Finite128-job frozen original Nested extension for seeds67/68 only."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.osram_frozen_nested_gcnet_20261025.scheduler import parse_gpu_map
from experiments.osram_signature_cyclic_multi_20261025.scheduler import fits
from experiments.osram_signature_stage2_extension_20261025.scheduler import finish as finish_worker
from experiments.osram_three_block_joint_mosi_20261025.scheduler import guard_gpu, now, resources, save
from .protocol import METHOD, PARAMETERS, RATES, SEEDS, budget, source_root, validate_scope

PACKAGE = 'experiments.osram_frozen_nested_3seed_20261026'
MODULE = PACKAGE + '.run'
DATASETS = ('CMUMOSEI', 'MELD', 'CMUMOSI', 'IEMOCAPSix')
PROC = Path('/proc')


def jobs_for(phase):
    if phase not in ('smoke', 'train'):
        raise ValueError('Only smoke/train phases are authorized')
    specs = ([(dataset, seed, 5 if dataset == 'IEMOCAPSix' else 1, '0.7')
              for dataset in DATASETS for seed in SEEDS] if phase == 'smoke' else
             [(dataset, seed, fold, rate) for rate in RATES for dataset in DATASETS
              for seed in SEEDS for fold in (range(1, 6) if dataset == 'IEMOCAPSix' else (1,))])
    return [dict(id=f'{dataset}_{seed}_fold{fold}_{rate}', dataset=dataset, seed=seed,
                 fold=fold, rate=rate, status='pending')
            for dataset, seed, fold, rate in specs]


def validate_state(state, source):
    if (state.get('source_commit') != source or state.get('method') != METHOD
            or state.get('expected_train_jobs') != 128 or state.get('seeds') != list(SEEDS)):
        raise ValueError('Controller source/method/finite seed scope recovery mismatch')
    for phase in ('smoke', 'train'):
        if phase not in state:
            continue
        expected = {job['id']: job for job in jobs_for(phase)}
        jobs = state[phase]
        if len(jobs) != len(expected) or len({job['id'] for job in jobs}) != len(expected):
            raise ValueError('Persisted queue has missing/duplicated/unapproved jobs')
        for job in jobs:
            approved = expected.get(job['id'])
            if approved is None or any(job.get(key) != approved[key]
                                       for key in ('dataset', 'seed', 'fold', 'rate')):
                raise ValueError('Persisted queue expands or changes the approved job scope')
            validate_scope(job['dataset'], job['seed'], job['fold'], job['rate'])
            if job.get('source_commit', source) != source:
                raise ValueError('Persisted worker source differs from the pinned controller source')
            if job['status'] not in ('pending', 'running', 'failed', 'blocked',
                                     'smoke_passed' if phase == 'smoke' else 'complete'):
                raise ValueError('Unknown persisted worker status')


def launch(args, job, uuid, phase):
    validate_scope(job['dataset'], job['seed'], job['fold'], job['rate'])
    output = args.root / phase / job['id']
    recovering = output.exists()
    if recovering and (not args.resume or not (output / 'last.pt').exists()):
        raise ValueError('Existing output requires explicit compatible full-state recovery')
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-u', '-m', MODULE, '--dataset', job['dataset'],
               '--seed', str(job['seed']), '--fold', str(job['fold']), '--rate', job['rate'],
               '--output', str(output), '--gpu-uuid', uuid, '--source-commit', args.source_commit]
    if phase == 'smoke':
        command.append('--smoke')
    if recovering:
        command.append('--resume')
    cwd = str(source_root(job['dataset'], job['seed']))
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    job.update(command=command, cwd=cwd, source_commit=args.source_commit, output=str(output),
               gpu_uuid=uuid, log=str(output / 'worker.log'), resume=recovering,
               environment={key: environment[key] for key in ('CUDA_VISIBLE_DEVICES',
                   'CUDA_DEVICE_ORDER', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')})
    try:
        with (output / 'worker.log').open('a', buffering=1) as stream:
            child = subprocess.Popen(command, cwd=cwd, env=environment, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        job.update(status='running', pid=child.pid, started_at=now())
        return child
    except BaseException as error:
        job.update(status='failed', error=repr(error), finished_at=now())
        raise


def process_matches(job):
    try:
        proc = PROC / str(job['pid'])
        values = [part.decode() for part in (proc / 'cmdline').read_bytes().split(b'\0') if part]
        return (proc.stat().st_uid == os.getuid() and values == job['command'] and MODULE in values
                and job['output'] in values and job['source_commit'] in values
                and (proc / 'cwd').resolve() == Path(job['cwd']).resolve())
    except (KeyError, OSError, UnicodeError):
        return False


def check_identity(identity, job, phase, source):
    validate_scope(identity['dataset'], identity['seed'], identity['fold'], identity['evaluation_rate'])
    expected = dict(source_commit=source, method=METHOD, dataset=job['dataset'], seed=job['seed'],
                    fold=job['fold'], evaluation_rate=job['rate'], smoke=(phase == 'smoke'))
    if type(identity.get('smoke')) is not bool or any(identity.get(k) != v for k, v in expected.items()):
        raise ValueError('Worker identity differs from the pinned source/dataset/seed/fold/rate/phase')


def finish(job, code, phase):
    finish_worker(job, code, phase)
    if job['status'] == 'failed':
        return
    try:
        summary = json.loads(Path(job['summary']).read_text())
        check_identity(summary['identity'], job, phase, job['source_commit'])
        limit = 2 if phase == 'smoke' else budget(job['dataset'])
        rss = float(summary['peak_rss_gib'])
        if (type(summary['epoch']) is not int or summary['epoch'] != limit
                or summary['total_parameters'] != PARAMETERS or not math.isfinite(rss) or rss <= 0):
            raise ValueError('Completion needs the full budget, original Nested, and positive measured RAM')
    except Exception as error:
        job.update(status='failed', error=str(error))


def check_recovery(path, job, phase, source):
    import torch
    saved = torch.load(path, map_location='cpu', weights_only=False)
    required = ('block', 'optimizer', 'rng', 'identity', 'history', 'epoch', 'best')
    if saved.get('version') != 1 or not all(key in saved for key in required):
        raise ValueError('Full versioned block/Adam/RNG/history checkpoint required')
    check_identity(saved['identity'], job, phase, source)
    epoch = saved['epoch']
    limit = 2 if phase == 'smoke' else budget(job['dataset'])
    if type(epoch) is not int or not 0 < epoch <= limit:
        raise ValueError('Checkpoint epoch outside the finite stage2 budget')
    if (not isinstance(saved['history'], list)
            or [row['epoch'] for row in saved['history']] != list(range(1, epoch + 1))):
        raise ValueError('Checkpoint history is incomplete')
    if (not saved['block'] or not isinstance(saved['optimizer']['state'], dict)
            or not saved['optimizer']['param_groups']
            or not all(key in saved['rng'] for key in ('python', 'numpy', 'torch', 'cuda'))
            or not 1 <= saved['best']['epoch'] <= epoch):
        raise ValueError('Block/Adam/RNG/selected epoch recovery state is incomplete')


def reconcile(jobs, phase, source, root=None):
    for job in jobs:
        validate_scope(job['dataset'], job['seed'], job['fold'], job['rate'])
        if job.get('pid') is not None and process_matches(job):
            job.update(status='running')
            continue
        if 'output' not in job and root is not None:
            output = root / phase / job['id']
            if output.exists():
                job['output'] = str(output)
        if 'output' not in job:
            job.update(status='pending')
            job.pop('error', None)
            continue
        job['source_commit'] = source
        if (Path(job['output']) / 'SUMMARY.json').exists():
            finish(job, 0, phase)
            if job['status'] != 'failed':
                continue
        try:
            check_recovery(Path(job['output']) / 'last.pt', job, phase, source)
            job.update(status='pending', recovery='compatible block/Adam/RNG/complete history')
            job.pop('error', None)
        except Exception as error:
            job.update(status='failed', error='No validated completion/full recovery state: ' + str(error))


def smoke_measurements(jobs):
    peaks = {}
    for job in jobs:
        if job['status'] != 'smoke_passed':
            raise ValueError('Every new seed/dataset smoke must pass before formal admission')
        summary = json.loads(Path(job['summary']).read_text())
        measured = dict(gpu_mib=float(summary['peak_memory_mib']), rss_gib=float(summary['peak_rss_gib']))
        if not all(math.isfinite(value) and value > 0 for value in measured.values()):
            raise ValueError('Positive measured smoke VRAM/RAM required')
        peak = peaks.setdefault(job['dataset'], dict(gpu_mib=0., rss_gib=0.))
        for name, value in measured.items():
            peak[name] = max(peak[name], value)
    return peaks


def refresh_measurements(jobs, peaks):
    """A new real full missing-rate cycle gates concurrency beyond one/dataset."""
    coverage = {dataset: set() for dataset in DATASETS}
    for job in jobs:
        if job['status'] not in ('running', 'complete') or 'output' not in job:
            continue
        output, dataset = Path(job['output']), job['dataset']
        summary_path = output / 'SUMMARY.json'
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            for field, name in (('peak_memory_mib', 'gpu_mib'), ('peak_rss_gib', 'rss_gib')):
                measured = float(summary.get(field, 0.))
                if not math.isfinite(measured) or measured < 0:
                    raise ValueError('Nonfinite/negative actual resource measurement')
                peaks[dataset][name] = max(peaks[dataset][name], measured)
        history_path = output / 'history.json'
        if history_path.exists():
            for row in json.loads(history_path.read_text()):
                rates = row['training_rates']
                if any(isinstance(rate, bool) or rate not in tuple(map(float, RATES)) for rate in rates):
                    raise ValueError('Unexpected cyclic training rate in resource warmup')
                coverage[dataset].update(rates)
    return {dataset: rates == set(map(float, RATES)) for dataset, rates in coverage.items()}


def preserved_live_workers(state):
    return [job['id'] for phase in ('smoke', 'train') for job in state.get(phase, [])
            if job['status'] == 'running']


def run_phase(args, state, phase, peaks=None):
    jobs = state.setdefault(phase, jobs_for(phase))
    children = {}
    try:
        if not 1 <= args.per_gpu <= 4:
            raise ValueError('Maximum four task workers per healthy GPU')
        if args.resume:
            reconcile(jobs, phase, args.source_commit, args.root)
        if any(job['status'] == 'running' and job.get('gpu_uuid') not in args.gpu_map.values() for job in jobs):
            raise ValueError('Live worker outside requested GPU map; preserve it for recovery')
        while any(job['status'] in ('running', 'pending') for job in jobs):
            for job in jobs:
                if job['status'] != 'running':
                    continue
                child = children.get(job['id'])
                code = child.poll() if child is not None else (None if process_matches(job) else 0)
                if code is not None:
                    finish(job, code, phase)
                    children.pop(job['id'], None)
            if any(job['status'] == 'failed' for job in jobs):
                for job in jobs:
                    if job['status'] == 'pending':
                        job.update(status='blocked', error='Worker failure; other live workers preserved')
            else:
                snapshot = resources(args.root, list(args.gpu_map.values()))
                state['latest_resources'] = snapshot
                warmed = refresh_measurements(jobs, peaks) if phase == 'train' else {}
                state['real_cycle_capacity_verified'] = warmed
                pending = next((job for job in jobs if job['status'] == 'pending' and
                    (phase == 'smoke' or warmed[job['dataset']] or not any(
                        peer['dataset'] == job['dataset'] and peer['status'] in ('running', 'complete')
                        for peer in jobs))), None)
                if pending is not None:
                    for uuid in sorted(args.gpu_map.values(), key=lambda value: sum(
                            job['status'] == 'running' and job.get('gpu_uuid') == value for job in jobs)):
                        peers = [job for job in jobs if job['status'] == 'running' and job.get('gpu_uuid') == uuid]
                        allowed = (not peers and snapshot['gpus'][uuid]['free_mib'] >= 8192
                            and snapshot['ram_available_gib'] >= 32 and snapshot['disk_free_gib'] >= 24
                            if phase == 'smoke' else fits(snapshot, uuid, pending, peers, peaks, args.per_gpu))
                        if allowed:
                            children[pending['id']] = launch(args, pending, uuid, phase)
                            break
            state['updated_at'] = now()
            save(args.root / 'QUEUE.json', state)
            if any(job['status'] in ('pending', 'running') for job in jobs):
                time.sleep(getattr(args, 'poll_seconds', 5))
        expected = 'smoke_passed' if phase == 'smoke' else 'complete'
        if any(job['status'] != expected for job in jobs):
            raise RuntimeError(phase + ' contains failed/blocked jobs')
    except BaseException as error:
        for job in jobs:
            if job['status'] == 'pending':
                job.update(status='blocked', error='Controller admission stopped: ' + repr(error))
        state.update(admission_error=repr(error), updated_at=now(),
                     preserved_live_workers=preserved_live_workers(state))
        save(args.root / 'QUEUE.json', state)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--gpu-map', type=parse_gpu_map, required=True)
    parser.add_argument('--per-gpu', type=int, default=4)
    parser.add_argument('--poll-seconds', type=float, default=5)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= args.per_gpu <= 4 or not math.isfinite(args.poll_seconds) or not 0 < args.poll_seconds <= 60:
        parser.error('Maximum four workers/card; poll interval must be in (0,60] seconds')
    args.root = args.root.resolve()
    args.root.mkdir(parents=True, exist_ok=args.resume)
    with (args.root / 'scheduler.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = args.root / 'QUEUE.json'
        state = json.loads(path.read_text()) if args.resume and path.exists() else dict(
            source_commit=args.source_commit, status='preflight', started_at=now(),
            expected_train_jobs=128, seeds=list(SEEDS), method=METHOD)
        validate_state(state, args.source_commit)
        state.update(pid=os.getpid(), command=[sys.executable] + sys.argv, cwd=os.getcwd(),
                     python=sys.executable, gpu_map=args.gpu_map, max_per_gpu=args.per_gpu)
        try:
            previous_device = os.environ.get('CUDA_VISIBLE_DEVICES')
            try:
                for index, uuid in args.gpu_map.items():
                    os.environ['CUDA_VISIBLE_DEVICES'] = uuid
                    guard_gpu(uuid, index)
            finally:
                if previous_device is None:
                    os.environ.pop('CUDA_VISIBLE_DEVICES', None)
                else:
                    os.environ['CUDA_VISIBLE_DEVICES'] = previous_device
            state['status'] = 'smoke'
            run_phase(args, state, 'smoke')
            peaks = smoke_measurements(state['smoke'])
            state.update(status='train', admission_measurements=peaks)
            run_phase(args, state, 'train', peaks)
            from .summarize import summarize
            result = summarize(args.root, args.source_commit, state['train'])
            if result['status'] != 'complete':
                raise RuntimeError('Final saved artifact validation failed')
            state.update(status='complete', completed_at=now())
            state.pop('admission_error', None)
            state.pop('preserved_live_workers', None)
            save(path, state)
        except BaseException as error:
            state.update(status='failed', error=repr(error), finished_at=now(),
                         preserved_live_workers=preserved_live_workers(state))
            save(path, state)
            raise


if __name__ == '__main__':
    main()
