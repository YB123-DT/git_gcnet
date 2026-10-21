"""Two real-batch smokes, then exactly two concurrent seed66 V3 runs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiments.osram_source_relation_attention_20261021.run import gpu_free
from .worker import REFERENCES
from .report import summarize


def worker_command(args, dataset, phase):
    if dataset not in REFERENCES or phase not in ('smoke', 'train'):
        raise ValueError('Only approved V3 jobs')
    return [sys.executable, '-u', '-m', 'experiments.osram_source_role_v3_20261021.worker',
        '--dataset', dataset, '--phase', phase, '--seed', '66', '--gpu-uuid', args.gpu_uuid,
        '--source-commit', args.source_commit,
        '--output', str(args.output/('smoke' if phase == 'smoke' else 'runs')/dataset)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    from experiments.osram_text_path_router_3seed_20261019.resource_gate import available_gib
    if args.output.exists():
        raise FileExistsError('Fresh V3 output required; workers support explicit full-state resume')
    if not args.output.parent.is_dir() or shutil.disk_usage(args.output.parent).free < 15*2**30:
        raise RuntimeError('Need existing output parent and15GiB free space')
    index, free = gpu_free(args.gpu_uuid)
    if available_gib() < 64 or free < 6000:
        raise RuntimeError('Need64GiB host RAM and6GiB GPU headroom for smoke')
    args.output.mkdir()
    lock = (args.output/'QUEUE.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = args.output/'QUEUE.json'; start = time.monotonic()
    state = dict(status='smoke', source_commit=args.source_commit, pid=os.getpid(),
        gpu_uuid=args.gpu_uuid, physical_gpu=index, jobs=[], smoke=[], max_parallel=2)
    def save():
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2)+'\n'); temporary.replace(path)
    save(); processes = []
    try:
        for dataset in REFERENCES:
            _, free = gpu_free(args.gpu_uuid)
            if available_gib() < 64 or free < 6000:
                raise RuntimeError('Resource admission changed before smoke: ' + dataset)
            log = args.output/('smoke_'+dataset+'.log')
            with log.open('w') as stream:
                result = subprocess.run(worker_command(args, dataset, 'smoke'),
                    stdout=stream, stderr=subprocess.STDOUT)
            record = args.output/'smoke'/dataset/'SUMMARY.json'
            summary = json.loads(record.read_text()) if record.exists() else {}
            state['smoke'].append(dict(dataset=dataset, returncode=result.returncode,
                status=summary.get('status'), peak_memory_mib=summary.get('peak_memory_mib'), log=str(log)))
            save()
            if result.returncode or summary.get('status') != 'complete':
                raise RuntimeError('Real-batch smoke failed: ' + dataset)
        _, free = gpu_free(args.gpu_uuid)
        required = 2000 + 1.4*sum(s['peak_memory_mib'] for s in state['smoke'])
        if free < required:
            raise RuntimeError('Measured GPU headroom cannot admit both jobs')
        state.update(status='running', free_memory_before_train_mib=free, estimated_required_mib=required)
        for dataset in REFERENCES:
            _, free = gpu_free(args.gpu_uuid)
            if available_gib() < 64 or free < required:
                raise RuntimeError('Resource admission changed before train: ' + dataset)
            log = args.output/('train_'+dataset+'.log')
            with log.open('w') as stream:
                process = subprocess.Popen(worker_command(args, dataset, 'train'),
                    stdout=stream, stderr=subprocess.STDOUT, env=dict(os.environ,
                        CUDA_VISIBLE_DEVICES=args.gpu_uuid, OMP_NUM_THREADS='1',
                        MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
            processes.append(process)
            state['jobs'].append(dict(dataset=dataset, pid=process.pid, status='running',
                output=str(args.output/'runs'/dataset), log=str(log),
                command=worker_command(args, dataset, 'train')))
            save()
        while True:
            active = False
            for process, job in zip(processes, state['jobs']):
                code = process.poll()
                if code is None:
                    active = True
                    continue
                record = Path(job['output'])/'SUMMARY.json'
                summary = json.loads(record.read_text()) if record.exists() else {}
                job.update(returncode=code,
                    status='complete' if code == 0 and summary.get('status') == 'complete' else 'failed')
            state['seconds'] = time.monotonic()-start; save()
            if not active:
                break
            time.sleep(15)
        summarize(args.output)
        state['status'] = 'complete' if all(j['status'] == 'complete' for j in state['jobs']) else 'failed'
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        state['seconds'] = time.monotonic()-start; save()


if __name__ == '__main__':
    main()
