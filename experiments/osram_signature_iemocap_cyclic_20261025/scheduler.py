"""Finite frozen Signature IEMOCAP-6 queue: 3 seeds x 5 folds x 8 rates."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.osram_signature_cyclic_multi_20261025.scheduler import fits, parse_gpu_map
from experiments.osram_signature_stage2_extension_20261025.scheduler import finish
from experiments.osram_three_block_joint_mosi_20261025.scheduler import guard_gpu, now, resources, save

RATES = tuple(f'{i / 10:.1f}' for i in range(8))
MODULE = 'experiments.osram_signature_iemocap_cyclic_20261025.run'


def jobs_for(phase):
    specs = [(66, 5, '0.7')] if phase == 'smoke' else [
        (seed, fold, rate) for rate in RATES for fold in (5, 1, 2, 3, 4)
        for seed in (66, 67, 68)]
    return [dict(id=f'IEMOCAPSix_{seed}_fold{fold}_{rate}', dataset='IEMOCAPSix',
                 seed=seed, fold=fold, rate=rate, status='pending')
            for seed, fold, rate in specs]


def process_matches(job):
    try:
        values = [v.decode() for v in Path(f'/proc/{job["pid"]}/cmdline').read_bytes().split(b'\0') if v]
        return MODULE in values and job['output'] in values
    except (FileNotFoundError, KeyError, PermissionError):
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
        if phase == 'train' and (output/'last.pt').exists():
            job.update(status='pending', recovery='full worker checkpoint')
        else:
            job.update(status='failed', error='No completed output or full last.pt')


def launch(args, job, uuid, phase):
    output = args.root/phase/job['id']
    recovering = output.exists()
    if recovering and (not args.resume or not (output/'last.pt').exists()):
        raise RuntimeError(f'Existing output requires --resume and full last.pt: {output}')
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-u', '-m', MODULE, '--seed', str(job['seed']),
               '--fold', str(job['fold']), '--rate', job['rate'], '--output', str(output),
               '--gpu-uuid', uuid, '--source-commit', args.source_commit]
    if phase == 'smoke':
        command.append('--smoke')
    if recovering:
        command.append('--resume')
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    log_path = output/'worker.log'
    with log_path.open('a', buffering=1) as stream:
        process = subprocess.Popen(command, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    job.update(status='running', pid=process.pid, command=command, gpu_uuid=uuid,
               output=str(output), log=str(log_path), started_at=now(), resume=recovering,
               environment={k: environment[k] for k in ('CUDA_VISIBLE_DEVICES',
                   'CUDA_DEVICE_ORDER', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')})
    return process


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
            code = process.poll() if process else (None if process_matches(job) else 0)
            if code is not None:
                finish(job, code, phase)
                local.pop(job['id'], None)
        if any(j['status'] == 'failed' for j in jobs):
            for job in jobs:
                if job['status'] == 'pending':
                    job.update(status='blocked', error='Admission stopped after worker failure')
        else:
            snapshot = resources(args.root, list(args.gpu_map.values()))
            state['latest_resources'] = snapshot
            for job in jobs:
                if job['status'] != 'pending':
                    continue
                admitted = False
                for uuid in sorted(args.gpu_map.values(), key=lambda u: sum(
                        j['status'] == 'running' and j.get('gpu_uuid') == u for j in jobs)):
                    peers = [j for j in jobs if j['status'] == 'running' and j.get('gpu_uuid') == uuid]
                    allowed = ((not peers and snapshot['ram_available_gib'] >= 32
                                and snapshot['disk_free_gib'] >= 16
                                and snapshot['gpus'][uuid]['free_mib'] >= args.smoke_reserve_mib)
                               if phase == 'smoke' else
                               fits(snapshot, uuid, job, peers, peaks, args.per_gpu))
                    if allowed:
                        local[job['id']] = launch(args, job, uuid, phase)
                        admitted = True
                        break
                if admitted:
                    break  # stagger feature-cache imports, not 24 simultaneous loads
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
    if not 1 <= args.per_gpu <= 6 or args.poll_seconds < 1:
        parser.error('per-gpu must be 1..6 and poll-seconds >=1')
    args.root = args.root.resolve()
    args.root.mkdir(parents=True, exist_ok=args.resume)
    with (args.root/'scheduler.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        queue = args.root/'QUEUE.json'
        state = json.loads(queue.read_text()) if args.resume and queue.exists() else dict(
            status='preflight', source_commit=args.source_commit, started_at=now(),
            dataset='IEMOCAPSix', seeds=[66, 67, 68], folds=[1, 2, 3, 4, 5],
            max_per_gpu=args.per_gpu, gpu_map=args.gpu_map, python=sys.executable,
            cwd=os.getcwd())
        if state['source_commit'] != args.source_commit:
            raise ValueError('Queue resume requires identical immutable source')
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
            state['status'] = 'smoke'
            save(queue, state)
            run_phase(args, state, 'smoke')
            measurement = json.loads(Path(state['smoke'][0]['summary']).read_text())
            gpu, rss = float(measurement['peak_memory_mib']), float(measurement.get('peak_rss_gib', 6.))
            if not all(math.isfinite(v) and v > 0 for v in (gpu, rss)):
                raise ValueError('Smoke must measure finite positive memory peaks')
            peaks = {'IEMOCAPSix': dict(gpu_mib=gpu, rss_gib=rss)}
            state.update(status='train', admission_measurements=peaks)
            run_phase(args, state, 'train', peaks)
            from .summarize import summarize
            result = summarize(args.root, args.source_commit, state['train'])
            if result['status'] != 'complete':
                raise RuntimeError('Final integrity or full-fivefold aggregation failed')
            state.update(status='complete', completed_at=now())
            save(queue, state)
            return 0
        except Exception as error:
            state.update(status='failed', error=str(error), finished_at=now())
            save(queue, state)
            print(f'Queue failed (no worker signalled): {error}', file=sys.stderr)
            return 1


if __name__ == '__main__':
    raise SystemExit(main())
