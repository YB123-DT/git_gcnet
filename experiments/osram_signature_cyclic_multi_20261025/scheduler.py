"""Persistent, finite 48-job frozen Signature cyclic stage-two queue."""
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

RATES = tuple(f'{i / 10:.1f}' for i in range(8))
SCOPE = (('CMUMOSI', 66), ('CMUMOSI', 67), ('CMUMOSEI', 66),
         ('CMUMOSEI', 67), ('CMUMOSEI', 68), ('MELD', 66))
MODULE = 'experiments.osram_signature_cyclic_multi_20261025.run'
BAD_UUID = 'GPU-46fb379f-cc90-dc82-9b5e-5d011f552264'


def jobs_for(phase):
    specs = [('CMUMOSI', 66), ('CMUMOSEI', 66), ('MELD', 66)] if phase == 'smoke' else SCOPE
    rates = ('0.7',) if phase == 'smoke' else RATES
    # Round-robin conditions avoids delaying all of one dataset behind another.
    return [dict(id=f'{d}_{s}_{r}', dataset=d, seed=s, rate=r, status='pending')
            for r in rates for d, s in specs]


def parse_gpu_map(value):
    result = {}
    for entry in value.split(','):
        index_text, uuid = entry.strip().split('=', 1)
        index = int(index_text)
        if (index not in (0, 1, 2, 3) or not uuid.startswith('GPU-') or uuid == BAD_UUID
                or index in result or uuid in result.values()):
            raise ValueError('Use distinct healthy physical GPU0/1/2/3 UUIDs only')
        result[index] = uuid
    if not result:
        raise ValueError('At least one GPU UUID is required')
    return result


def fits(snapshot, uuid, job, peers, peaks, limit):
    """Reserve measured peaks even while workers are still importing/loading."""
    if len(peers) >= limit or snapshot['disk_free_gib'] < 16:
        return False
    measurement = peaks[job['dataset']]
    required = 4096 + 1.15 * (measurement['gpu_mib'] + sum(
        peaks[p['dataset']]['gpu_mib'] for p in peers))
    # Available RAM already excludes resident workers; retain a 24-GiB system reserve.
    return (snapshot['gpus'][uuid]['free_mib'] >= required
            and snapshot['ram_available_gib'] >= 24 + measurement['rss_gib'])


def launch(args, job, uuid, phase):
    output = args.root / phase / job['id']
    recovering = output.exists()
    if recovering and not args.resume:
        raise FileExistsError(f'Output exists without explicit resume: {output}')
    if recovering and not (output/'last.pt').exists():
        # Do not overwrite failed/partial work without a recoverable full state.
        raise RuntimeError(f'Cannot resume without full last.pt: {output}')
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-u', '-m', MODULE, '--dataset', job['dataset'],
               '--seed', str(job['seed']), '--rate', job['rate'], '--output', str(output),
               '--gpu-uuid', uuid, '--source-commit', args.source_commit]
    if phase == 'smoke':
        command.append('--smoke')
    if recovering:
        command.append('--resume')
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    log_path = output/'worker.log'
    with log_path.open('a', buffering=1) as log:
        process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
    job.update(status='running', pid=process.pid, command=command, gpu_uuid=uuid,
               output=str(output), log=str(log_path), started_at=now(), resume=recovering,
               environment={k: environment[k] for k in ('CUDA_VISIBLE_DEVICES',
                   'CUDA_DEVICE_ORDER', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')})
    return process


def process_matches(job):
    """Avoid adopting a recycled PID or a different experiment's process."""
    try:
        command = Path(f'/proc/{job["pid"]}/cmdline').read_bytes().split(b'\0')
        values = [part.decode() for part in command if part]
        return MODULE in values and job['output'] in values
    except (FileNotFoundError, KeyError, PermissionError):
        return False


def reconcile(jobs, phase):
    for job in jobs:
        if job['status'] == 'pending':
            continue
        if job['status'] == 'running' and process_matches(job):
            continue  # Retain/adopt; never duplicate a surviving worker.
        output = Path(job.get('output', '/nonexistent'))
        if (output/'SUMMARY.json').exists():
            finish(job, 0, phase)
            if job['status'] != 'failed':
                continue
        if phase == 'train' and (output/'last.pt').exists():
            job.update(status='pending', recovery='full worker checkpoint')
        else:
            job.update(status='failed', error='No completed output or recoverable full last.pt')


def run_phase(args, state, phase, peaks=None):
    jobs = state.setdefault(phase, jobs_for(phase))
    if args.resume:
        reconcile(jobs, phase)
    local = {}
    expected = 'smoke_passed' if phase == 'smoke' else 'complete'
    while any(j['status'] in ('pending', 'running') for j in jobs):
        for job in jobs:
            if job['status'] != 'running':
                continue
            process = local.get(job['id'])
            code = process.poll() if process is not None else (None if process_matches(job) else 0)
            if code is not None:
                finish(job, code, phase)
                local.pop(job['id'], None)
        failed = [j for j in jobs if j['status'] == 'failed']
        if failed:
            for job in jobs:
                if job['status'] == 'pending':
                    job.update(status='blocked', error='Queue admission stopped after failed worker')
        else:
            snapshot = resources(args.root, list(args.gpu_map.values()))
            state['latest_resources'] = snapshot
            # One admission per poll avoids loading 24 feature caches simultaneously.
            for job in jobs:
                if job['status'] != 'pending':
                    continue
                admitted = False
                for uuid in sorted(args.gpu_map.values(), key=lambda u: sum(
                        j['status'] == 'running' and j.get('gpu_uuid') == u for j in jobs)):
                    peers = [j for j in jobs if j['status'] == 'running' and j.get('gpu_uuid') == uuid]
                    if phase == 'smoke':
                        allowed = (not peers and snapshot['ram_available_gib'] >= 32
                                   and snapshot['disk_free_gib'] >= 16
                                   and snapshot['gpus'][uuid]['free_mib'] >= args.smoke_reserve_mib)
                    else:
                        allowed = fits(snapshot, uuid, job, peers, peaks, args.per_gpu)
                    if allowed:
                        local[job['id']] = launch(args, job, uuid, phase)
                        admitted = True
                        break
                if admitted:
                    break
        state['updated_at'] = now()
        save(args.root/'QUEUE.json', state)
        if any(j['status'] == 'running' for j in jobs) or any(j['status'] == 'pending' for j in jobs):
            # Resource contention can resolve; do not treat occupied GPUs as a failed experiment.
            time.sleep(args.poll_seconds)
    if any(j['status'] != expected for j in jobs):
        raise RuntimeError(f'{phase}: incomplete/failed jobs retained in QUEUE.json')


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
    if not 1 <= args.per_gpu <= 6 or args.poll_seconds < 1:
        parser.error('per-gpu must be 1..6 and poll-seconds >= 1')
    args.root = args.root.resolve()
    args.root.mkdir(parents=True, exist_ok=args.resume)
    with (args.root/'scheduler.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        queue = args.root/'QUEUE.json'
        state = json.loads(queue.read_text()) if args.resume and queue.exists() else dict(
            status='preflight', source_commit=args.source_commit, started_at=now(),
            scope=list(SCOPE), max_per_gpu=args.per_gpu, gpu_map=args.gpu_map,
            python=sys.executable, cwd=os.getcwd())
        if state['source_commit'] != args.source_commit:
            raise ValueError('Cannot resume a queue with a different immutable source')
        state.update(pid=os.getpid(), command=[sys.executable]+sys.argv)
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
            save(queue, state)
            state['status'] = 'smoke'
            run_phase(args, state, 'smoke')
            peaks = {}
            for job in state['smoke']:
                summary = json.loads(Path(job['summary']).read_text())
                gpu = float(summary['peak_memory_mib'])
                rss = float(summary.get('peak_rss_gib', 6.))
                if not math.isfinite(gpu) or not math.isfinite(rss) or min(gpu, rss) <= 0:
                    raise ValueError('Smoke must provide finite positive memory measurements')
                peaks[job['dataset']] = dict(gpu_mib=gpu, rss_gib=rss)
            state.update(status='train', admission_measurements=peaks)
            run_phase(args, state, 'train', peaks)
            from .summarize import summarize
            result = summarize(args.root, args.source_commit, state['train'])
            if result['status'] != 'complete':
                raise RuntimeError('Final results failed integrity/complete-group validation; inspect RESULT.json')
            state.update(status='complete', completed_at=now())
            save(queue, state)
            return 0
        except Exception as error:
            state.update(status='failed', error=str(error), finished_at=now())
            save(queue, state)
            print(f'Queue failure (worker PIDs retained, none signalled): {error}', file=sys.stderr)
            return 1


if __name__ == '__main__':
    raise SystemExit(main())
