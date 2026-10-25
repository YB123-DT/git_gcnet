"""Finite18-job, four-configuration NoArea wiring/stage comparison."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.osram_three_block_joint_mosi_20261025.scheduler import guard_gpu, now, resources, save
from experiments.osram_signature_stage2_extension_20261025.scheduler import finish
from experiments.osram_signature_cyclic_multi_20261025.scheduler import fits as measured_fits

PACKAGE = 'experiments.osram_recent_noarea_serial_20261025'
RATES = tuple(f'{i/10:.1f}' for i in range(8))


def parse_gpu_map(value):
    output = {}
    for entry in value.split(','):
        name, uuid = entry.split('=', 1)
        index = int(name)
        if (index not in (0, 1, 2) or index in output or uuid in output.values()
            or not uuid.startswith('GPU-') or uuid == 'GPU-46fb379f-cc90-dc82-9b5e-5d011f552264'):
            raise ValueError('Use distinct healthy physicalGPU0/1/2 UUIDs only')
        output[index] = uuid
    if not output:
        raise ValueError('At least one healthy GPU is required')
    return output


def jobs_for(phase):
    specs = [('stage1', mode, None) for mode in ('parallel', 'serial')]
    specs += [('stage2', mode, rate) for rate in (('0.7',) if phase == 'smoke' else RATES)
              for mode in ('parallel', 'serial')]
    return [dict(id=f'{stage}_{mode}'+('' if rate is None else '_'+rate),
        stage=stage, mode=mode, rate=rate, dataset='CMUMOSI', seed=66, status='pending')
        for stage, mode, rate in specs]


def key(job):
    return job['stage']+':'+job['mode']


def fits(snapshot, uuid, job, peers, peaks, limit):
    return measured_fits(snapshot, uuid, dict(dataset=key(job)),
        [dict(dataset=key(p)) for p in peers], peaks, limit)


def process_matches(job):
    try:
        proc = Path(f'/proc/{job["pid"]}')
        cmd = [part.decode() for part in (proc/'cmdline').read_bytes().split(b'\0') if part]
        return (proc.stat().st_uid == os.getuid() and cmd == job['command']
            and PACKAGE+'.'+job['stage'] in cmd and job['output'] in cmd
            and job['source_commit'] in cmd and (proc/'cwd').resolve() == Path(job['cwd']).resolve())
    except (OSError, KeyError):
        return False


def recoverable(job):
    import torch
    try:
        state = torch.load(Path(job['output'])/'last.pt', map_location='cpu', weights_only=False)
        identity = state['identity']
        required = ('model', 'optimizer', 'rng', 'noise_rng', 'history', 'epoch', 'best_by_rate') if job['stage'] == 'stage1' else (
            'block', 'optimizer', 'rng', 'identity', 'history', 'epoch', 'best')
        return (all(k in state for k in required) and identity['source_commit'] == job['source_commit']
            and identity['mode'] == job['mode'] and identity['seed'] == 66)
    except (OSError, KeyError, RuntimeError, EOFError, ValueError):
        return False


def reconcile(jobs, phase):
    for job in jobs:
        if job['status'] == 'pending' or (job['status'] == 'running' and process_matches(job)):
            continue
        if 'output' not in job:
            if job['status'] == 'blocked':
                job.update(status='pending')
                job.pop('error', None)
            else:
                job.update(status='failed', error='No launch/output record; recovery cannot be verified')
            continue
        if (Path(job['output'])/'SUMMARY.json').exists():
            finish(job, 0, phase)
            if job['status'] != 'failed':
                continue
        if phase == 'train' and recoverable(job):
            job.update(status='pending', recovery='full compatible checkpoint')
        else:
            job.update(status='failed', error='No validated completion/full checkpoint; preserve artifacts')


def launch(args, job, uuid, phase):
    output = args.root/phase/job['id']
    recovering = output.exists()
    if recovering and (not args.resume or not (output/'last.pt').exists()):
        raise ValueError('Existing artifacts require explicit full-state resume')
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-u', '-m', PACKAGE+'.'+job['stage'], '--mode', job['mode'],
        '--output', str(output), '--gpu-uuid', uuid, '--source-commit', args.source_commit]
    if job['stage'] == 'stage2':
        command += ['--rate', job['rate']]
    if phase == 'smoke':
        command += ['--smoke']
    if recovering:
        command += ['--resume']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    log_path = output/'worker.log'
    with log_path.open('a', buffering=1) as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    job.update(status='running', command=command, pid=process.pid, source_commit=args.source_commit,
        cwd=os.getcwd(), output=str(output), gpu_uuid=uuid, log=str(log_path), started_at=now(), resume=recovering)
    return process


def run_phase(args, state, phase, peaks=None):
    jobs = state.setdefault(phase, jobs_for(phase))
    if args.resume:
        reconcile(jobs, phase)
    if any(j['status'] == 'running' and j.get('gpu_uuid') not in args.gpu_map.values() for j in jobs):
        raise ValueError('Live worker outside requested map; no duplicate admission')
    processes = {}
    while any(j['status'] in ('pending', 'running') for j in jobs):
        for job in jobs:
            if job['status'] != 'running':
                continue
            p = processes.get(job['id'])
            code = p.poll() if p is not None else (None if process_matches(job) else 0)
            if code is not None:
                finish(job, code, phase)
                processes.pop(job['id'], None)
        if any(j['status'] == 'failed' for j in jobs):
            for job in jobs:
                if job['status'] == 'pending':
                    job.update(status='blocked', error='Worker failure; live jobs preserved, no further admissions')
        else:
            snapshot = resources(args.root, list(args.gpu_map.values()))
            state['latest_resources'] = snapshot
            pending = next((j for j in jobs if j['status'] == 'pending'), None)
            if pending:
                for uuid in sorted(args.gpu_map.values(), key=lambda u:sum(
                    j['status'] == 'running' and j.get('gpu_uuid') == u for j in jobs)):
                    peers = [j for j in jobs if j['status'] == 'running' and j.get('gpu_uuid') == uuid]
                    allowed = (not peers and snapshot['ram_available_gib'] >= 32
                        and snapshot['disk_free_gib'] >= 24 and snapshot['gpus'][uuid]['free_mib'] >= 8192
                        if phase == 'smoke' else fits(snapshot, uuid, pending, peers, peaks, args.per_gpu))
                    if allowed:
                        processes[pending['id']] = launch(args, pending, uuid, phase)
                        break
        state['updated_at'] = now()
        save(args.root/'QUEUE.json', state)
        if any(j['status'] in ('pending', 'running') for j in jobs):
            time.sleep(args.poll_seconds)
    if any(j['status'] != ('smoke_passed' if phase == 'smoke' else 'complete') for j in jobs):
        raise RuntimeError(phase+' contains failed/blocked jobs')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--gpu-map', type=parse_gpu_map, required=True)
    parser.add_argument('--per-gpu', type=int, default=6)
    parser.add_argument('--poll-seconds', type=float, default=5)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= args.per_gpu <= 6 or args.poll_seconds < 1:
        parser.error('per-gpu1..6 and poll>=1second required')
    args.root = args.root.resolve()
    args.root.mkdir(parents=True, exist_ok=args.resume)
    with (args.root/'scheduler.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = args.root/'QUEUE.json'
        state = json.loads(path.read_text()) if args.resume and path.exists() else dict(
            source_commit=args.source_commit, status='preflight', started_at=now(),
            dataset='CMUMOSI', seed=66, descriptor='S1+symS2;NoArea', expected_train_jobs=18)
        if state['source_commit'] != args.source_commit or state['descriptor'] != 'S1+symS2;NoArea':
            raise ValueError('Immutable source/descriptor resume mismatch')
        state.update(pid=os.getpid(), command=[sys.executable]+sys.argv, cwd=os.getcwd(),
            python=sys.executable, gpu_map=args.gpu_map, max_per_gpu=args.per_gpu)
        try:
            old = os.environ.get('CUDA_VISIBLE_DEVICES')
            try:
                for index, uuid in args.gpu_map.items():
                    os.environ['CUDA_VISIBLE_DEVICES'] = uuid
                    guard_gpu(uuid, index)
            finally:
                if old is None:
                    os.environ.pop('CUDA_VISIBLE_DEVICES', None)
                else:
                    os.environ['CUDA_VISIBLE_DEVICES'] = old
            state['status'] = 'smoke'
            run_phase(args, state, 'smoke')
            peaks = {}
            for job in state['smoke']:
                s = json.loads(Path(job['summary']).read_text())
                gpu, ram = (float(s[k]) for k in ('peak_memory_mib', 'peak_rss_gib'))
                if not all(math.isfinite(v) and v > 0 for v in (gpu, ram)):
                    raise ValueError('Require positive actual VRAM/RAM smoke measurements')
                peaks[key(job)] = dict(gpu_mib=gpu, rss_gib=ram)
            state.update(status='train', admission_measurements=peaks)
            run_phase(args, state, 'train', peaks)
            from .summarize import summarize
            result = summarize(args.root, args.source_commit, state['train'])
            if result['status'] != 'complete':
                raise RuntimeError('Final selected-artifact validation failed')
            state.update(status='complete', completed_at=now())
            save(path, state)
            return 0
        except Exception as error:
            state.update(status='failed', error=repr(error), finished_at=now())
            save(path, state)
            print(str(error), file=sys.stderr)
            return 1


if __name__ == '__main__':
    raise SystemExit(main())
