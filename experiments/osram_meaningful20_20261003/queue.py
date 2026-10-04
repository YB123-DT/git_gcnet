"""Durable single-host queue; dispatches only individually verified snapshots."""
from __future__ import annotations
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

from .manifest import digest_json, now, read, sha, validate_round, verify_snapshot, write
from .preflight import admission, query_gpus, validate_gpu, validate_readiness
from .run import completion_status
from .summarize import rank_promotions, replication_decision, scores


def process_identity(pid):
    try:
        fields = Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z': return None
        return {'pid': int(pid), 'start_ticks': fields[19],
                'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    except (OSError, ValueError, IndexError):
        return None


def process_matches(record):
    actual = process_identity(record.get('pid', -1))
    return actual is not None and all(record.get(k) == v for k, v in actual.items())


def reconcile_job(job):
    job = dict(job)
    if process_matches(job): return dict(job, status='running')
    identity_file = Path(job['output']) / 'RUN_IDENTITY.json'
    if identity_file.exists():
        child = read(identity_file)
        if child.get('run_id') == job.get('run_id') and process_matches(child):
            return dict(job, **{k: child[k] for k in ('pid', 'start_ticks', 'boot_id')}, status='running')
    failure_path = Path(job['output']).with_name(Path(job['output']).name + '.launch-failure.json')
    failure = None
    if failure_path.exists():
        candidate = read(failure_path)
        if (candidate.get('run_id') == job.get('run_id')
                and candidate.get('at', '') >= job.get('launched_at', '')):
            failure = candidate
    recoverable = (failure and failure.get('failure_category') == 'interrupted'
                   and (Path(job['output']) / 'last_training.pt').is_file())
    if job.get('process_exit_code') not in (None, 0):
        return dict(job, status='interrupted' if recoverable else 'failed',
                    detail=f'observed process exit {job["process_exit_code"]}',
                    failure_category='interrupted' if recoverable else 'nonzero_exit')
    status, detail = completion_status(job['output'])
    if status == 'complete':
        if job.get('process_exit_code') == 0: return dict(job, status='complete', detail=None)
        return dict(job, status='interrupted', detail='Exit unobserved after coordinator restart; resume committed state to verify clean exit')
    if failure:
        return dict(job, status='interrupted' if recoverable else 'failed', detail=failure.get('error'),
                    failure_category=failure.get('failure_category'))
    if status == 'failed': return dict(job, status='failed', detail=detail)
    return dict(job, status='inspection_pending' if job.get('status') == 'launch_intent' else 'interrupted', detail=detail)


def _epoch_count(job):
    try: return len(read(Path(job['output']) / 'history.json'))
    except (OSError, ValueError): return 0


def resource_rejected_before_start(job):
    """Only retry admission races that created no training output/state."""
    if not job or job.get('status') != 'failed' or Path(job['output']).exists():
        return False
    marker = Path(job['output']).with_name(Path(job['output']).name + '.launch-failure.json')
    if not marker.is_file():
        return False
    failure = read(marker)
    return (failure.get('run_id') == job.get('run_id')
            and 'Admission changed before child start:' in failure.get('error', ''))


def reserved_disk_gib(active):
    reserved = 0.
    for job in active:
        used = sum(p.stat().st_size for p in Path(job['output']).rglob('*') if p.is_file()) / 1024 ** 3
        reserved += max(0., job.get('profile', {}).get('artifact_gib', 0.) - used)
    return reserved


def update_throughput(state, active):
    """Measure completed epochs per wall-second for stable GPU cohorts."""
    windows = state.setdefault('throughput_windows', {})
    measurements = state.setdefault('throughput', {})
    timestamp = time.time()
    uuids = {job['gpu_uuid'] for job in active}
    for gpu_uuid in uuids:
        cohort = sorted(key for key, job in state['jobs'].items()
                        if job['status'] == 'running' and job['gpu_uuid'] == gpu_uuid)
        count = sum(_epoch_count(state['jobs'][key]) for key in cohort)
        window = windows.get(gpu_uuid)
        if window:
            old_count = sum(_epoch_count(state['jobs'][key]) for key in window['cohort'])
            elapsed = timestamp - window['started']
            if elapsed >= 60 and old_count > window['epochs']:
                throughput = (old_count - window['epochs']) / elapsed
                previous = measurements.get(gpu_uuid, {})
                baseline = previous.get('baseline_epochs_per_second', throughput)
                ratio = throughput / baseline if baseline else 1.
                cap = previous.get('concurrency_cap', 6)
                if ratio < .9:
                    cap = min(cap, max(1, len(window['cohort']) - 1))
                measurements[gpu_uuid] = {'epochs_per_second': throughput,
                    'baseline_epochs_per_second': max(baseline, throughput),
                    'throughput_ratio': ratio, 'concurrency_cap': cap,
                    'window_seconds': elapsed, 'measured_at': now()}
                window = None
            elif window['cohort'] != cohort:
                window = None
        if window is None:
            windows[gpu_uuid] = {'cohort': cohort, 'epochs': count, 'started': timestamp}


def _phase(state, ids, baseline):
    jobs = state['jobs']
    if not all(jobs.get(f'{name}:66', {}).get('status') == 'complete' for name in ids): return
    rows = {name: dict(scores(read(Path(jobs[f'{name}:66']['output']) / 'metrics.json')), status='complete') for name in ids}
    promotions = rank_promotions(ids, rows, baseline[66])
    if 'promotions' in state and state['promotions'] != promotions:
        raise ValueError('Frozen promotion decision changed')
    state['promotions'] = promotions
    if not promotions:
        state.update(phase='awaiting_new_round', decision={'next': 'new_round', 'reason': 'no seed66 improvement'})
    elif all(jobs.get(f'{name}:{seed}', {}).get('status') == 'complete' for name in promotions for seed in (67, 68)):
        values = {name: {seed: scores(read(Path(jobs[f'{name}:{seed}']['output']) / 'metrics.json'))['mean8']
                         for seed in (66, 67, 68)} for name in promotions}
        decision = replication_decision(values, baseline)
        state.update(phase='verified_improvement' if decision['best'] else 'awaiting_new_round', decision=decision)
    else:
        state['phase'] = 'replication'


def coordinate(args):
    manifest = read(args.manifest)
    ids = validate_round(manifest)
    cards = {card['id']: card for card in manifest['cards']}
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / 'logs').mkdir(exist_ok=True)
    baseline = {row['seed']: row['mean8'] for row in read(args.baseline_audit)['runs']}
    spec = {'manifest_sha256': sha(args.manifest), 'baseline_audit_sha256': sha(args.baseline_audit),
            'data_manifest_sha256': sha(args.data_manifest), 'reference_root': str(args.reference_root.resolve())}
    with (args.root / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = args.root / 'QUEUE.json'
        state = read(path) if path.exists() else {'spec': spec, 'phase': 'screening', 'jobs': {}, 'created_at': now()}
        if state['spec'] != spec: raise ValueError('Queue source/design/protocol identity changed')
        handles = {}
        while True:
            for name in ('manifest', 'baseline_audit', 'data_manifest'):
                if sha(getattr(args, name)) != spec[name + '_sha256']:
                    raise ValueError(f'Frozen queue input changed: {name}')
            for key, job in list(state['jobs'].items()):
                if job['status'] == 'complete': continue  # Fully audited once; final summarizer rechecks.
                if key in handles:
                    code = handles[key].poll()
                    if code is not None: job['process_exit_code'] = code
                state['jobs'][key] = reconcile_job(job)
            _phase(state, ids, baseline)
            state.update(updated_at=now(), coordinator=process_identity(os.getpid()))
            if state['phase'] in ('awaiting_new_round', 'verified_improvement'):
                write(path, state)
                return
            expected = ([(name, 66) for name in ids] if state['phase'] == 'screening' else
                        [(name, seed) for name in state['promotions'] for seed in (67, 68)])
            active = [job for job in state['jobs'].values() if job['status'] == 'running']
            update_throughput(state, active)
            state['waiting_reason'] = 'awaiting implementation/CPU+CUDA readiness'
            for candidate, seed in expected:
                key = f'{candidate}:{seed}'
                previous = state['jobs'].get(key)
                resumable = previous and previous['status'] == 'interrupted' and (Path(previous['output']) / 'last_training.pt').is_file()
                if previous and not (resumable or resource_rejected_before_start(previous)): continue
                if len(active) >= args.max_concurrent:
                    state['waiting_reason'] = 'concurrency cap'; break
                if any(_epoch_count(job) < 1 for job in active):
                    state['waiting_reason'] = 'first complete epoch before increasing concurrency'; break
                ready_path = args.readiness_root / candidate / 'READY.json'
                if not ready_path.exists(): continue
                ready = read(ready_path)
                snapshot = Path(ready['snapshot_root']).resolve()
                source = verify_snapshot(snapshot)
                profile = validate_readiness(ready, candidate=candidate,
                    design_sha256=cards[candidate]['design_sha256'], source_sha256=source['source_sha256'])
                if previous and previous.get('snapshot_sha256') != digest_json(source['source_sha256']):
                    raise ValueError('Cannot resume under a changed source snapshot')
                if os.getloadavg()[0] > max(1, os.cpu_count() or 1):
                    state['waiting_reason'] = 'CPU load admission'; break
                resources = query_gpus()
                mapping = {i: row['uuid'] for i, row in resources.items()}
                if 'gpu_mapping' in state and state['gpu_mapping'] != mapping:
                    raise ValueError('GPU index/UUID mapping changed; re-audit before dispatch')
                state['gpu_mapping'] = mapping
                active_uuids = {job['gpu_uuid'] for job in active}
                available = []
                disk = shutil.disk_usage(args.root).free / 1024 ** 3 - reserved_disk_gib(active)
                for index, gpu in resources.items():
                    if index == '4': continue
                    validate_gpu(index, gpu['uuid'], mapping)
                    measured = state.get('throughput', {}).get(gpu['uuid'], {})
                    gpu_jobs = sum(job['gpu_uuid'] == gpu['uuid'] for job in active)
                    if gpu_jobs >= measured.get('concurrency_cap', args.max_concurrent):
                        state['waiting_reason'] = 'measured throughput concurrency cap'; continue
                    ok, reason = admission(gpu, profile, disk_free_gib=disk)
                    if ok: available.append((gpu['uuid'] not in active_uuids, gpu['utilization'], -gpu['free_mib'], index))
                    else: state['waiting_reason'] = reason
                if not available: break
                gpu_index = min(available)[-1]; gpu_uuid = mapping[gpu_index]
                output = Path(previous['output']) if previous else args.root / 'runs' / candidate / f'seed_{seed}'
                output.parent.mkdir(parents=True, exist_ok=True)
                if output.exists() and not resumable:
                    state['jobs'][key] = {'output': str(output), 'status': 'inspection_pending', 'detail': 'orphan output'}
                    continue
                run_id = previous['run_id'] if previous else uuid.uuid4().hex
                command = [args.python, '-u', '-m', 'experiments.osram_meaningful20_20261003.run',
                    '--candidate', candidate, '--seed', str(seed), '--output', str(output.resolve()),
                    '--manifest', str(args.manifest.resolve()), '--manifest-sha256', spec['manifest_sha256'],
                    '--snapshot', str(snapshot), '--readiness', str(ready_path.resolve()),
                    '--reference-root', str(args.reference_root.resolve()),
                    '--baseline-audit', str(args.baseline_audit.resolve()), '--data-manifest', str(args.data_manifest.resolve()),
                    '--baseline-audit-sha256', spec['baseline_audit_sha256'],
                    '--data-manifest-sha256', spec['data_manifest_sha256'], '--readiness-sha256', sha(ready_path),
                    '--gpu', gpu_index, '--gpu-uuid', gpu_uuid, '--run-id', run_id]
                if resumable: command.append('--resume')
                attempt = (previous.get('attempt', 1) + 1) if previous else 1
                job = {'status': 'launch_intent', 'output': str(output.resolve()), 'run_id': run_id,
                       'candidate': candidate, 'seed': seed, 'gpu_index': gpu_index, 'gpu_uuid': gpu_uuid,
                       'command': command, 'attempt': attempt, 'launched_at': now(),
                       'snapshot_sha256': digest_json(source['source_sha256']), 'profile': profile}
                state['jobs'][key] = job
                write(path, state)
                environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid, PYTHONPATH=str(snapshot),
                    OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
                log_path = args.root / 'logs' / f'{candidate}_seed{seed}_attempt{attempt}.log'
                try:
                    with log_path.open('x') as log:
                        process = subprocess.Popen(command, cwd=snapshot, env=environment,
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    handles[key] = process
                    job.update(process_identity(process.pid) or {'pid': process.pid})
                    job.update(status='running', log=str(log_path), profile=profile)
                    active.append(job)
                except BaseException as error:
                    job.update(status='inspection_pending', detail=repr(error))
                    write(path, state)
                    raise
                write(path, state)
                break  # Stagger launches; remeasure live resources next iteration.
            write(path, state)
            if args.once: return
            time.sleep(args.poll_seconds)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'manifest', 'readiness-root', 'reference-root', 'baseline-audit', 'data-manifest'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--max-concurrent', type=int, default=3)
    p.add_argument('--poll-seconds', type=float, default=15)
    p.add_argument('--once', action='store_true')
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    if not 1 <= args.max_concurrent <= 6 or args.poll_seconds < 1:
        raise ValueError('Require 1..6 jobs and positive polling interval')
    coordinate(args)
