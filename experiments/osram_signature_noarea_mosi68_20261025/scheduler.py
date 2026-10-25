"""Eight matched MOSI68 NoArea branches; smoke, finite queue and paired report."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.osram_signature_noarea_iemocap68_20261025.scheduler import parse_gpu_map
from experiments.osram_signature_cyclic_multi_20261025.scheduler import fits
from experiments.osram_signature_stage2_extension_20261025.scheduler import finish
from experiments.osram_three_block_joint_mosi_20261025.scheduler import guard_gpu, now, resources, save

MODULE = 'experiments.osram_signature_noarea_mosi68_20261025.run'
RATES = tuple(f'{i/10:.1f}' for i in range(8))


def jobs_for(phase):
    return [dict(id=f'CMUMOSI_68_{rate}', dataset='CMUMOSI', seed=68, fold=1,
                 rate=rate, status='pending') for rate in (('0.7',) if phase == 'smoke' else RATES)]


def process_matches(job):
    try:
        proc = Path(f'/proc/{job["pid"]}')
        cmd = [p.decode() for p in (proc/'cmdline').read_bytes().split(b'\0') if p]
        return (proc.stat().st_uid == os.getuid() and cmd == job['command']
            and MODULE in cmd and job['output'] in cmd and job['source_commit'] in cmd
            and (proc/'cwd').resolve() == Path(job['worker_cwd']).resolve())
    except (OSError, KeyError):
        return False


def reconcile(jobs, phase):
    from experiments.osram_signature_noarea_iemocap68_20261025.scheduler import recoverable
    for job in jobs:
        if job['status'] == 'pending' or (job['status'] == 'running' and process_matches(job)):
            continue
        if (Path(job['output'])/'SUMMARY.json').exists():
            finish(job, 0, phase)
            if job['status'] != 'failed':
                continue
        if phase == 'train' and recoverable(job):
            job.update(status='pending', recovery='full block/Adam/RNG checkpoint')
        else:
            job.update(status='failed', error='No validated completion or full recovery checkpoint')


def launch(args, job, uuid, phase):
    output = args.root/phase/job['id']
    recovering = output.exists()
    if recovering and (not args.resume or not (output/'last.pt').exists()):
        raise ValueError('Existing output requires explicit full-state resume')
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-u', '-m', MODULE, '--rate', job['rate'],
        '--output', str(output), '--gpu-uuid', uuid, '--source-commit', args.source_commit]
    if phase == 'smoke':
        command.append('--smoke')
    if recovering:
        command.append('--resume')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    log_path = output/'worker.log'
    with log_path.open('a', buffering=1) as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    job.update(status='running', command=command, pid=process.pid, gpu_uuid=uuid,
        source_commit=args.source_commit, worker_cwd=os.getcwd(), output=str(output),
        log=str(log_path), started_at=now(), resume=recovering)
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
            process = processes.get(job['id'])
            code = process.poll() if process is not None else (None if process_matches(job) else 0)
            if code is not None:
                finish(job, code, phase)
                processes.pop(job['id'], None)
        if any(j['status'] == 'failed' for j in jobs):
            for job in jobs:
                if job['status'] == 'pending':
                    job.update(status='blocked', error='Admission stopped after failure; live workers preserved')
        else:
            snapshot = resources(args.root, list(args.gpu_map.values()))
            state['latest_resources'] = snapshot
            pending = next((j for j in jobs if j['status'] == 'pending'), None)
            if pending is not None:
                for uuid in sorted(args.gpu_map.values(), key=lambda u:sum(
                        j['status'] == 'running' and j.get('gpu_uuid') == u for j in jobs)):
                    peers = [j for j in jobs if j['status'] == 'running' and j.get('gpu_uuid') == uuid]
                    allowed = (snapshot['ram_available_gib'] >= 32 and snapshot['disk_free_gib'] >= 16
                        and snapshot['gpus'][uuid]['free_mib'] >= 8192) if phase == 'smoke' else fits(
                            snapshot, uuid, pending, peers, peaks, args.per_gpu)
                    if allowed:
                        processes[pending['id']] = launch(args, pending, uuid, phase)
                        break
        state['updated_at'] = now()
        save(args.root/'QUEUE.json', state)
        if any(j['status'] in ('pending', 'running') for j in jobs):
            time.sleep(args.poll_seconds)
    expected = 'smoke_passed' if phase == 'smoke' else 'complete'
    if any(j['status'] != expected for j in jobs):
        raise RuntimeError(f'{phase} incomplete; inspect preserved logs')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--gpu-map', type=parse_gpu_map, required=True)
    parser.add_argument('--per-gpu', type=int, default=3)
    parser.add_argument('--poll-seconds', type=float, default=5)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= args.per_gpu <= 4 or args.poll_seconds < 1:
        parser.error('Require 1..4 workers/card and poll >=1s')
    args.root = args.root.resolve()
    args.root.mkdir(parents=True, exist_ok=args.resume)
    with (args.root/'scheduler.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = args.root/'QUEUE.json'
        state = json.loads(path.read_text()) if args.resume and path.exists() else dict(
            source_commit=args.source_commit, dataset='CMUMOSI', seeds=[68], folds=[1],
            ablation='degree2_no_area', status='preflight', started_at=now())
        if state['source_commit'] != args.source_commit or state['ablation'] != 'degree2_no_area':
            raise ValueError('Resume source/ablation mismatch')
        state.update(command=[sys.executable]+sys.argv, pid=os.getpid(), cwd=os.getcwd(),
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
            smoke = json.loads(Path(state['smoke'][0]['summary']).read_text())
            values = [float(smoke[k]) for k in ('peak_memory_mib', 'peak_rss_gib')]
            if not all(math.isfinite(v) and v > 0 for v in values):
                raise ValueError('Require real finite VRAM/RAM smoke measurements')
            peaks = {'CMUMOSI':dict(gpu_mib=values[0], rss_gib=values[1])}
            state.update(status='train', admission_measurements=peaks)
            run_phase(args, state, 'train', peaks)
            from .summarize import summarize
            result = summarize(args.root, args.source_commit, state['train'])
            if result['status'] != 'complete':
                raise RuntimeError('Strict paired result validation failed')
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
