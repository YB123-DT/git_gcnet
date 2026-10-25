"""Finite NeuMiss MOSI67/68 completion; never rerun accepted seed66."""
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
from experiments.osram_neumiss_fixed_rate_stage2_20261025.scheduler import capacity

RATES = tuple(f'{i/10:.1f}' for i in range(8))
MODULE = 'experiments.osram_neumiss_stage2_3seed_20261025.run'
OLD_ROOT = Path('/data2/yb/remote_experiments/osram_neumiss_fixed_rate_stage2_20261025/execution_v1')
PROC = Path('/proc')


def jobs_for(phase):
    pairs = [(67,'0.7')] if phase == 'smoke' else [(s,r) for s in (67,68) for r in RATES]
    return [dict(id=f'CMUMOSI_{s}_{r}',seed=s,rate=r,mode='neumiss',status='pending') for s,r in pairs]


def parse_gpu_map(text):
    result = {}
    for item in text.split(','):
        index, uuid = item.strip().replace(':','=',1).split('=',1)
        index = int(index)
        if index not in (0,1,2) or index in result or uuid in result.values() or not uuid.startswith('GPU-'):
            raise ValueError('Only unique healthy physical GPU0/1/2 UUIDs are allowed')
        result[index] = uuid
    return result


def can_admit(snapshot, uuid, peers, peak, limit):
    return (snapshot['ram_available_gib'] >= 32 and snapshot['disk_free_gib'] >= 16
            and peers < limit and snapshot['gpus'][uuid]['free_mib'] >= peak*(peers+1)+4096)


def process_alive(job):
    directory = PROC/str(job['pid'])
    try:
        actual = [p.decode() for p in (directory/'cmdline').read_bytes().split(b'\0') if p]
    except FileNotFoundError:
        return False
    if not actual:
        return False
    if actual != job['command']:
        raise RuntimeError(f'Recovered PID {job["pid"]} no longer matches owned worker command')
    return True


def launch(args, job, uuid, phase):
    output = args.root/phase/job['id']
    exists = output.exists()
    if exists and not args.resume:
        raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=exists)
    command = [sys.executable,'-u','-m',MODULE,'--seed',str(job['seed']),'--rate',job['rate'],
        '--gpu-uuid',uuid,'--output',str(output),'--source-commit',args.source_commit]
    if phase == 'smoke':
        command.append('--smoke')
    if exists:
        if not (output/'last.pt').exists():
            raise RuntimeError(f'Cannot resume {output} without full last.pt')
        command.append('--resume')
    environment = dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,CUDA_DEVICE_ORDER='PCI_BUS_ID',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    log_path = output/'worker.log'
    with log_path.open('a',buffering=1) as log:
        process = subprocess.Popen(command,env=environment,stdout=log,stderr=subprocess.STDOUT)
    job.update(status='running',pid=process.pid,command=command,gpu_uuid=uuid,output=str(output),
        log=str(log_path),started_at=now(),environment={k:environment[k] for k in
        ('CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')})
    return process


def finish(job, phase, code=None):
    job.update(finished_at=now(),returncode=code)
    try:
        if code not in (None,0):
            raise RuntimeError(f'Worker exited {code}; see {job["log"]}')
        path = Path(job['output'])/'SUMMARY.json'
        summary = json.loads(path.read_text())
        expected = 'smoke_passed' if phase == 'smoke' else 'complete'
        if summary.get('status') != expected:
            raise ValueError(f'Expected worker status {expected}')
        peak = float(summary['peak_memory_mib'])
        if not math.isfinite(peak) or peak <= 0:
            raise ValueError('Missing positive measured GPU peak')
        job.update(status=expected,summary=str(path),peak_memory_mib=peak)
    except Exception as error:
        job.update(status='failed',error=str(error))


def run_phase(args,state,phase,peak=2048):
    jobs = state.setdefault(phase,jobs_for(phase))
    handles = {}
    # Adopt only exact recorded workers after a controller restart. No duplicate launch.
    for job in jobs:
        if job['status'] == 'running' and not process_alive(job):
            finish(job,phase)
            if job['status'] == 'failed' and args.resume and (Path(job['output'])/'last.pt').exists():
                job.update(status='pending',resume_reason=job.pop('error'))
    while any(j['status'] in ('pending','running') for j in jobs):
        for job in jobs:
            if job['status'] != 'running':
                continue
            handle = handles.get(job['id'])
            code = handle.poll() if handle else None
            alive = code is None if handle else process_alive(job)
            if not alive:
                finish(job,phase,code)
        if any(j['status'] == 'failed' for j in jobs):
            for job in jobs:
                if job['status'] == 'pending':
                    job.update(status='blocked',error='Earlier finite queue worker failed')
        else:
            snapshot = resources(args.root,list(args.gpus.values()))
            state['latest_resources'] = snapshot
            for job in jobs:
                if job['status'] != 'pending':
                    continue
                for uuid in args.gpus.values():
                    peers = sum(j['status']=='running' and j['gpu_uuid']==uuid for j in jobs)
                    if not can_admit(snapshot,uuid,peers,peak,1 if phase=='smoke' else args.per_gpu):
                        continue
                    handles[job['id']] = launch(args,job,uuid,phase)
                    snapshot['gpus'][uuid]['free_mib'] -= peak
                    save(args.root/'QUEUE.json',state)
                    break
        state.update(updated_at=now())
        save(args.root/'QUEUE.json',state)
        if any(j['status'] in ('pending','running') for j in jobs):
            time.sleep(5)  # Busy resources wait without changing protocol or terminating others.
    expected = 'smoke_passed' if phase=='smoke' else 'complete'
    if any(j['status'] != expected for j in jobs):
        raise RuntimeError(f'{phase} contains failed/blocked jobs; see QUEUE.json')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--gpu-map',required=True,help='Physical index=UUID, comma separated; only 0/1/2')
    parser.add_argument('--per-gpu',type=int,default=4)
    parser.add_argument('--mosi66-root',type=Path,default=OLD_ROOT)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args(argv)
    args.gpus = parse_gpu_map(args.gpu_map)
    if not 1 <= args.per_gpu <= 6:
        parser.error('--per-gpu must be 1..6, with smoke-measured memory admission')
    args.root = args.root.resolve()
    args.root.mkdir(parents=True,exist_ok=args.resume)
    lock = (args.root/'QUEUE.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.resume:
        state = json.loads((args.root/'QUEUE.json').read_text())
        if (state['source_commit'] != args.source_commit or state['gpu_map'] != args.gpu_map
                or state['max_per_gpu'] != args.per_gpu):
            raise ValueError('Resume must preserve immutable source/GPU/concurrency identity')
    else:
        state = dict(status='preflight',source_commit=args.source_commit,started_at=now(),
            gpu_map=args.gpu_map,max_per_gpu=args.per_gpu,mosi66_root=str(args.mosi66_root),
            command=[sys.executable]+sys.argv)
    state.update(pid=os.getpid())
    save(args.root/'QUEUE.json',state)
    try:
        previous = os.environ.get('CUDA_VISIBLE_DEVICES')
        try:
            for index,uuid in args.gpus.items():
                os.environ['CUDA_VISIBLE_DEVICES'] = uuid
                guard_gpu(uuid,index)
        finally:
            if previous is None:
                os.environ.pop('CUDA_VISIBLE_DEVICES',None)
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = previous
        capacity(resources(args.root,list(args.gpus.values())))
        from .summarize import collect, validated_row
        for rate in RATES:
            validated_row(args.mosi66_root/'train'/f'neumiss_{rate}'/'SUMMARY.json',66,rate)
        state.update(status='smoke')
        run_phase(args,state,'smoke')
        peak = max(j['peak_memory_mib'] for j in state['smoke'])
        state.update(status='train',admission_smoke_peak_mib=peak)
        run_phase(args,state,'train',peak)
        collect(args,state)
        state.update(status='complete',completed_at=now())
        save(args.root/'QUEUE.json',state)
        return 0
    except Exception as error:
        # Retain all child PIDs and complete checkpoints. Resume adopts exact processes.
        state.update(status='failed',error=repr(error),finished_at=now())
        save(args.root/'QUEUE.json',state)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
