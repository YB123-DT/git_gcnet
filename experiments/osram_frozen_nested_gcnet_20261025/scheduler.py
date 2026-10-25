"""One measured GPU smoke, then exactly eight frozen Nested jobs."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.osram_recent_noarea_serial_20261025.scheduler import parse_gpu_map
from experiments.osram_three_block_joint_mosi_20261025.scheduler import guard_gpu, now, resources, save
from experiments.osram_signature_cyclic_multi_20261025.scheduler import fits
from experiments.osram_signature_stage2_extension_20261025.scheduler import finish

PACKAGE = 'experiments.osram_frozen_nested_gcnet_20261025'
MODULE = PACKAGE+'.run'


def jobs_for(phase):
    if phase not in ('smoke', 'train'): raise ValueError('Only smoke/train phases')
    rates = ('0.7',) if phase == 'smoke' else tuple(str(i/10) for i in range(8))
    return [dict(id='nested_'+r, dataset='CMUMOSI', seed=66, rate=r,
        stage='stage2', mode='nested', status='pending') for r in rates]


def launch(args, job, uuid, phase):
    output = args.root/phase/job['id']
    recovering = output.exists()
    if recovering and (not args.resume or not (output/'last.pt').exists()):
        raise ValueError('Existing run requires explicit full-state recovery')
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-u', '-m', MODULE, '--rate', job['rate'],
        '--output', str(output), '--gpu-uuid', uuid, '--source-commit', args.source_commit]
    if phase == 'smoke': command.append('--smoke')
    if recovering: command.append('--resume')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    with (output/'worker.log').open('a', buffering=1) as stream:
        process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True)
    job.update(status='running', pid=process.pid, command=command, source_commit=args.source_commit,
        cwd=os.getcwd(), gpu_uuid=uuid, output=str(output), log=str(output/'worker.log'),
        started_at=now(), resume=recovering)
    return process


def process_matches(job):
    try:
        proc = Path(f'/proc/{job["pid"]}')
        values = [p.decode() for p in (proc/'cmdline').read_bytes().split(b'\0') if p]
        return (proc.stat().st_uid == os.getuid() and values == job['command'] and MODULE in values
            and job['source_commit'] in values and (proc/'cwd').resolve() == Path(job['cwd']).resolve())
    except (KeyError, OSError):
        return False


def reconcile(jobs, phase, source):
    import torch
    for job in jobs:
        if job['status'] == 'pending' or (job['status'] == 'running' and process_matches(job)): continue
        if 'output' not in job:
            job.update(status='pending')
            continue
        if (Path(job['output'])/'SUMMARY.json').exists():
            finish(job, 0, phase)
            if job['status'] != 'failed': continue
        try:
            state = torch.load(Path(job['output'])/'last.pt', map_location='cpu', weights_only=False)
            identity = state['identity']
            assert identity['source_commit'] == source and identity['method'] == 'nested_gnn_rooted_evidence'
            assert identity['seed'] == 66 and identity['evaluation_rate'] == job['rate']
            assert identity['smoke'] == (phase == 'smoke') and 0 < state['epoch'] <= (2 if phase == 'smoke' else 150)
            assert all(k in state for k in ('block','optimizer','rng','identity','epoch','history','best'))
            job.update(status='pending', recovery='compatible block/Adam/RNG/history')
        except (OSError, KeyError, RuntimeError, ValueError, EOFError, AssertionError):
            job.update(status='failed', error='No validated completion/full recovery state')


def run_phase(args, state, phase, peaks=None):
    jobs = state.setdefault(phase, jobs_for(phase))
    if args.resume: reconcile(jobs, phase, args.source_commit)
    if any(j['status'] == 'running' and j.get('gpu_uuid') not in args.gpu_map.values() for j in jobs):
        raise ValueError('Cannot adopt live jobs outside the requested UUID map')
    processes = {}
    while any(j['status'] in ('running','pending') for j in jobs):
        for job in jobs:
            if job['status'] != 'running': continue
            process = processes.get(job['id'])
            code = process.poll() if process else (None if process_matches(job) else 0)
            if code is not None:
                finish(job, code, phase)
                processes.pop(job['id'], None)
        if any(j['status'] == 'failed' for j in jobs):
            for job in jobs:
                if job['status'] == 'pending': job.update(status='blocked', error='Worker failed; preserve all live jobs')
        else:
            snapshot = resources(args.root, list(args.gpu_map.values()))
            state['latest_resources'] = snapshot
            pending = next((j for j in jobs if j['status'] == 'pending'), None)
            if pending:
                for uuid in sorted(args.gpu_map.values(), key=lambda u:sum(
                        j['status'] == 'running' and j.get('gpu_uuid') == u for j in jobs)):
                    peers = [j for j in jobs if j['status'] == 'running' and j.get('gpu_uuid') == uuid]
                    allowed = (not peers and snapshot['gpus'][uuid]['free_mib'] >= 8192
                        and snapshot['ram_available_gib'] >= 32 and snapshot['disk_free_gib'] >= 24
                        if phase == 'smoke' else fits(snapshot, uuid, pending, peers, peaks, args.per_gpu))
                    if allowed:
                        processes[pending['id']] = launch(args, pending, uuid, phase)
                        break
        state['updated_at'] = now()
        save(args.root/'QUEUE.json', state)
        if any(j['status'] in ('pending','running') for j in jobs): time.sleep(5)
    if any(j['status'] != ('smoke_passed' if phase == 'smoke' else 'complete') for j in jobs):
        raise RuntimeError(phase+' has failed/blocked jobs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--gpu-map', type=parse_gpu_map, required=True)
    parser.add_argument('--per-gpu', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.per_gpu <= 4: parser.error('Per-GPU maximum4')
    args.root = args.root.resolve()
    args.root.mkdir(parents=True, exist_ok=args.resume)
    with (args.root/'scheduler.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = args.root/'QUEUE.json'
        state = json.loads(path.read_text()) if args.resume and path.exists() else dict(
            source_commit=args.source_commit, status='preflight', started_at=now(), expected_train_jobs=8,
            dataset='CMUMOSI', seed=66, method='nested_gnn_rooted_evidence')
        if state['source_commit'] != args.source_commit or state['method'] != 'nested_gnn_rooted_evidence':
            raise ValueError('Controller recovery source/method mismatch')
        state.update(pid=os.getpid(), command=[sys.executable]+sys.argv, cwd=os.getcwd(),
            python=sys.executable, gpu_map=args.gpu_map, max_per_gpu=args.per_gpu)
        try:
            old = os.environ.get('CUDA_VISIBLE_DEVICES')
            try:
                for index, uuid in args.gpu_map.items():
                    os.environ['CUDA_VISIBLE_DEVICES'] = uuid
                    guard_gpu(uuid, index)
            finally:
                if old is None: os.environ.pop('CUDA_VISIBLE_DEVICES', None)
                else: os.environ['CUDA_VISIBLE_DEVICES'] = old
            state['status'] = 'smoke'
            run_phase(args, state, 'smoke')
            summary = json.loads(Path(state['smoke'][0]['summary']).read_text())
            measurement = dict(gpu_mib=float(summary['peak_memory_mib']), rss_gib=float(summary['peak_rss_gib']))
            if not all(math.isfinite(v) and v > 0 for v in measurement.values()):
                raise ValueError('Positive measured VRAM and RAM required')
            state.update(status='train', admission_measurements={'CMUMOSI':measurement})
            run_phase(args, state, 'train', state['admission_measurements'])
            from .summarize import summarize
            result = summarize(args.root, args.source_commit, state['train'])
            if result['status'] != 'complete': raise RuntimeError('Final artifact validation failed')
            state.update(status='complete', completed_at=now())
            save(path, state)
        except BaseException as error:
            state.update(status='failed', error=repr(error), finished_at=now())
            save(path, state)
            raise


if __name__ == '__main__':
    main()
