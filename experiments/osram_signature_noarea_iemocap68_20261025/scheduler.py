"""Finite seed68 NoArea comparison; 40 independent frozen-parent branches."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.osram_signature_cyclic_multi_20261025.scheduler import fits
from experiments.osram_signature_stage2_extension_20261025.scheduler import finish
from experiments.osram_three_block_joint_mosi_20261025.scheduler import guard_gpu, now, resources, save

RATES = tuple(f'{i/10:.1f}' for i in range(8))
MODULE = 'experiments.osram_signature_noarea_iemocap68_20261025.run'
BAD_UUID = 'GPU-46fb379f-cc90-dc82-9b5e-5d011f552264'


def parse_gpu_map(value):
    result = {}
    for item in value.split(','):
        index_text, uuid = item.strip().split('=', 1)
        index = int(index_text)
        if (index not in (0, 1, 2) or not uuid.startswith('GPU-') or uuid == BAD_UUID
                or index in result or uuid in result.values()):
            raise ValueError('Use distinct healthy physical GPU0/1/2 UUIDs only')
        result[index] = uuid
    if not result:
        raise ValueError('At least one healthy GPU is required')
    return result


def jobs_for(phase):
    specs = [(5, '0.7')] if phase == 'smoke' else [(fold, rate) for rate in RATES for fold in (5, 1, 2, 3, 4)]
    return [dict(id=f'IEMOCAPSix_68_fold{fold}_{rate}', dataset='IEMOCAPSix', seed=68,
                 fold=fold, rate=rate, status='pending') for fold, rate in specs]


def process_matches(job):
    """A recycled PID, foreign UID, different source or cwd is never adopted."""
    try:
        process = Path(f'/proc/{job["pid"]}')
        values = [v.decode() for v in (process/'cmdline').read_bytes().split(b'\0') if v]
        return (process.stat().st_uid == os.getuid() and values == job['command']
                and MODULE in values and job['output'] in values
                and job['source_commit'] in values
                and (process/'cwd').resolve() == Path(job['worker_cwd']).resolve())
    except (FileNotFoundError, KeyError, PermissionError, OSError):
        return False


def recoverable(job):
    """Check full-state structure; the worker additionally verifies exact identity."""
    import torch
    try:
        saved = torch.load(Path(job['output'])/'last.pt', map_location='cpu', weights_only=False)
        identity = saved['identity']
        return (all(key in saved for key in ('block', 'optimizer', 'rng', 'epoch', 'history', 'best'))
                and identity['source_commit'] == job['source_commit']
                and identity['ablation'] == 'degree2_no_area'
                and identity['seed'] == 68 and identity['fold'] == job['fold']
                and identity['evaluation_rate'] == job['rate'])
    except (OSError, KeyError, ValueError, RuntimeError, EOFError):
        return False


def reconcile(jobs, phase):
    for job in jobs:
        if job['status'] == 'pending' or (job['status'] == 'running' and process_matches(job)):
            continue
        output = Path(job.get('output', '/nonexistent'))
        if (output/'SUMMARY.json').exists():
            finish(job, 0, phase)
            if job['status'] != 'failed':
                continue
        if phase == 'train' and recoverable(job):
            job.update(status='pending', recovery='full block/optimizer/RNG checkpoint')
        else:
            job.update(status='failed', error='No validated completed output or full recovery checkpoint')


def launch(args, job, uuid, phase):
    output = args.root/phase/job['id']
    recovering = output.exists()
    if recovering and (not args.resume or not (output/'last.pt').exists()):
        raise RuntimeError(f'Existing output requires --resume and full last.pt: {output}')
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-u', '-m', MODULE, '--seed', '68', '--fold', str(job['fold']),
               '--rate', job['rate'], '--output', str(output), '--gpu-uuid', uuid,
               '--source-commit', args.source_commit]
    if phase == 'smoke':
        command.append('--smoke')
    if recovering:
        command.append('--resume')
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    log_path = output/'worker.log'
    with log_path.open('a', buffering=1) as log:
        process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    job.update(status='running', pid=process.pid, command=command, gpu_uuid=uuid,
               source_commit=args.source_commit, worker_cwd=os.getcwd(), output=str(output),
               log=str(log_path), started_at=now(), resume=recovering,
               environment={key:environment[key] for key in ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER',
                            'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')})
    return process


def run_phase(args, state, phase, peaks=None):
    jobs = state.setdefault(phase, jobs_for(phase))
    if args.resume:
        reconcile(jobs, phase)
    if any(j['status'] == 'running' and j.get('gpu_uuid') not in args.gpu_map.values() for j in jobs):
        raise ValueError('A surviving worker is outside requested GPU map; preserve it without duplicate admission')
    processes = {}
    expected = 'smoke_passed' if phase == 'smoke' else 'complete'
    while any(j['status'] in ('pending', 'running') for j in jobs):
        for job in jobs:
            if job['status'] != 'running':
                continue
            process = processes.get(job['id'])
            code = process.poll() if process else (None if process_matches(job) else 0)
            if code is not None:
                finish(job, code, phase)
                processes.pop(job['id'], None)
        if any(j['status'] == 'failed' for j in jobs):
            for job in jobs:
                if job['status'] == 'pending':
                    job.update(status='blocked', error='Admission stopped after worker failure; live workers preserved')
        else:
            snapshot = resources(args.root, list(args.gpu_map.values()))
            state['latest_resources'] = snapshot
            for job in jobs:
                if job['status'] != 'pending':
                    continue
                admitted = False
                for uuid in sorted(args.gpu_map.values(), key=lambda u:sum(
                        j['status'] == 'running' and j.get('gpu_uuid') == u for j in jobs)):
                    peers = [j for j in jobs if j['status'] == 'running' and j.get('gpu_uuid') == uuid]
                    allowed = ((not peers and snapshot['ram_available_gib'] >= 32
                                and snapshot['disk_free_gib'] >= 16
                                and snapshot['gpus'][uuid]['free_mib'] >= args.smoke_reserve_mib)
                               if phase == 'smoke' else fits(snapshot, uuid, job, peers, peaks, args.per_gpu))
                    if allowed:
                        processes[job['id']] = launch(args, job, uuid, phase)
                        admitted = True
                        break
                if admitted:
                    break  # At most one import/cache load per polling interval.
        state['updated_at'] = now()
        save(args.root/'QUEUE.json', state)
        if any(j['status'] in ('pending', 'running') for j in jobs):
            time.sleep(args.poll_seconds)
    if any(j['status'] != expected for j in jobs):
        raise RuntimeError(f'{phase} has failed/blocked jobs; inspect QUEUE.json')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--gpu-map', type=parse_gpu_map, required=True)
    parser.add_argument('--per-gpu', type=int, default=6)
    parser.add_argument('--smoke-reserve-mib', type=int, default=8192)
    parser.add_argument('--poll-seconds', type=float, default=5)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= args.per_gpu <= 12 or args.poll_seconds < 1:
        parser.error('per-gpu must be 1..12 and poll-seconds >=1')
    args.root = args.root.resolve()
    args.root.mkdir(parents=True, exist_ok=args.resume)
    with (args.root/'scheduler.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        queue = args.root/'QUEUE.json'
        state = json.loads(queue.read_text()) if args.resume and queue.exists() else dict(
            status='preflight', source_commit=args.source_commit, started_at=now(), dataset='IEMOCAPSix',
            seeds=[68], folds=[1, 2, 3, 4, 5], ablation='degree2_no_area', python=sys.executable, cwd=os.getcwd())
        if state['source_commit'] != args.source_commit or state.get('ablation') != 'degree2_no_area':
            raise ValueError('Resume requires identical immutable source and ablation')
        state.update(pid=os.getpid(), command=[sys.executable]+sys.argv, max_per_gpu=args.per_gpu, gpu_map=args.gpu_map)
        try:
            previous = os.environ.get('CUDA_VISIBLE_DEVICES')
            try:
                for index, uuid in args.gpu_map.items():
                    os.environ['CUDA_VISIBLE_DEVICES'] = uuid
                    guard_gpu(uuid, index)
            finally:
                if previous is None:
                    os.environ.pop('CUDA_VISIBLE_DEVICES', None)
                else:
                    os.environ['CUDA_VISIBLE_DEVICES'] = previous
            state['status'] = 'smoke'
            save(queue, state)
            run_phase(args, state, 'smoke')
            measured = json.loads(Path(state['smoke'][0]['summary']).read_text())
            gpu, rss = float(measured['peak_memory_mib']), float(measured.get('peak_rss_gib', 6.))
            if not all(math.isfinite(v) and v > 0 for v in (gpu, rss)):
                raise ValueError('Real smoke must report positive finite VRAM/RAM peaks')
            peaks = {'IEMOCAPSix':dict(gpu_mib=gpu, rss_gib=rss)}
            state.update(status='train', admission_measurements=peaks)
            run_phase(args, state, 'train', peaks)
            from .summarize import summarize
            result = summarize(args.root, args.source_commit, state['train'])
            if result['status'] != 'complete':
                raise RuntimeError('Final paired integrity/complete-fivefold validation failed')
            state.update(status='complete', completed_at=now())
            save(queue, state)
            return 0
        except Exception as error:
            state.update(status='failed', error=str(error), finished_at=now())
            save(queue, state)
            if state.get('train'):
                from .summarize import summarize
                summarize(args.root, args.source_commit, state['train'])
            print(f'Queue failed (no worker signalled): {error}', file=sys.stderr)
            return 1


if __name__ == '__main__':
    raise SystemExit(main())
