"""Persistent finite controller: two real-batch smokes, then two concurrent runs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .worker import REFERENCES, LAYOUTS
from .report import summarize


def gpu_free(uuid):
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free', '--format=csv,noheader,nounits'], text=True)
    for line in raw.splitlines():
        index, found, free = [v.strip() for v in line.split(',')]
        if found == uuid:
            if index == '4' or found == 'GPU-46fb379f-cc90-dc82-9b5e-5d011f552264':
                raise ValueError('Forbidden failed host GPU4')
            return int(index), int(free)
    raise ValueError('Unknown GPU UUID')


def worker_command(args, dataset, phase):
    return [sys.executable, '-u', '-m', 'experiments.osram_source_relation_attention_20261021.worker',
        '--dataset', dataset, '--phase', phase, '--seed', '66', '--gpu-uuid', args.gpu_uuid,
        '--source-commit', args.source_commit, '--layout', args.layout,
        '--output', str(args.output/('smoke' if phase == 'smoke' else 'runs')/dataset)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--layout', choices=LAYOUTS, default='source-mean')
    args = parser.parse_args()
    from experiments.osram_text_path_router_3seed_20261019.resource_gate import available_gib
    if available_gib() < 64: raise RuntimeError('Need64GiB host RAM headroom for two data loaders')
    if args.output.exists(): raise FileExistsError('Fresh finite queue required; workers support explicit full-state resume')
    if shutil.disk_usage(args.output.parent).free < 15*2**30: raise RuntimeError('Need15GiB output headroom')
    index, free = gpu_free(args.gpu_uuid)
    if free < 6000: raise RuntimeError('Insufficient free GPU memory for real-batch smoke')
    args.output.mkdir(parents=True)
    lock = (args.output/'QUEUE.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = args.output/'QUEUE.json'; start = time.monotonic()
    state = dict(status='smoke', source_commit=args.source_commit, pid=os.getpid(),
        gpu_uuid=args.gpu_uuid, physical_gpu=index, layout=args.layout, jobs=[], smoke=[], max_parallel=2)
    def save():
        temporary = path.with_suffix('.tmp'); temporary.write_text(json.dumps(state, indent=2)+'\n'); temporary.replace(path)
    def command(dataset, phase):
        return worker_command(args, dataset, phase)
    save(); processes = []
    try:
        for dataset in REFERENCES:
            _, free = gpu_free(args.gpu_uuid)
            if available_gib() < 64 or free < 6000:
                raise RuntimeError('RAM/GPU admission changed before smoke: '+dataset)
            log = args.output/('smoke_'+dataset+'.log')
            with log.open('w') as stream:
                result = subprocess.run(command(dataset, 'smoke'), stdout=stream, stderr=subprocess.STDOUT)
            p = args.output/'smoke'/dataset/'SUMMARY.json'
            summary = json.loads(p.read_text()) if p.exists() else {}
            state['smoke'].append(dict(dataset=dataset, returncode=result.returncode, status=summary.get('status'),
                peak_memory_mib=summary.get('peak_memory_mib'), log=str(log)))
            save()
            if result.returncode or summary.get('status') != 'complete': raise RuntimeError('Real-batch smoke failed: '+dataset)
        index, free = gpu_free(args.gpu_uuid)
        required = 2000 + 1.4*sum(s['peak_memory_mib'] for s in state['smoke'])
        if free < required: raise RuntimeError(f'Concurrent memory headroom insufficient: {free}<{required}; no training launched')
        state.update(status='running', free_memory_before_train_mib=free, estimated_required_mib=required)
        for dataset in REFERENCES:
            _, free = gpu_free(args.gpu_uuid)
            if available_gib() < 64 or free < required:
                raise RuntimeError('RAM/GPU admission changed before train: '+dataset)
            log = args.output/('train_'+dataset+'.log'); stream = log.open('w')
            proc = subprocess.Popen(command(dataset, 'train'), stdout=stream, stderr=subprocess.STDOUT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu_uuid, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
            stream.close(); processes.append(proc)
            state['jobs'].append(dict(dataset=dataset, pid=proc.pid, command=command(dataset, 'train'),
                log=str(log), status='running', output=str(args.output/'runs'/dataset)))
            save()
        while True:
            active = False
            for proc, job in zip(processes, state['jobs']):
                code = proc.poll()
                if code is None: active = True; continue
                p = Path(job['output'])/'SUMMARY.json'; summary = json.loads(p.read_text()) if p.exists() else {}
                job.update(returncode=code, status='complete' if code==0 and summary.get('status')=='complete' else 'failed')
            state['seconds'] = time.monotonic()-start; save()
            if not active: break
            time.sleep(15)
        summarize(args.output)
        state['status'] = 'complete' if all(j['status']=='complete' for j in state['jobs']) else 'failed'
    except BaseException as error:
        state.update(status='failed', error=repr(error)); raise
    finally:
        state['seconds'] = time.monotonic()-start; save()


if __name__ == '__main__': main()
