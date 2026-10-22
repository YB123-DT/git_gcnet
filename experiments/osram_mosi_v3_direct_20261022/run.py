"""One real smoke followed by one fresh MOSI66 direct training, without signals."""
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
from .worker import MODE
from .report import summarize


def worker_command(args, phase):
    if phase not in ('smoke', 'train'):
        raise ValueError('Only one smoke then one train is approved')
    return [sys.executable, '-u', '-m', 'experiments.osram_mosi_v3_direct_20261022.worker',
            '--mode', MODE, '--phase', phase, '--seed', '66', '--gpu-uuid', args.gpu_uuid,
            '--source-commit', args.source_commit,
            '--output', str(args.output/('smoke' if phase == 'smoke' else 'runs')/MODE)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args(argv)
    from experiments.osram_text_path_router_3seed_20261019.resource_gate import available_gib
    index, free = gpu_free(args.gpu_uuid)
    if args.output.exists():
        raise FileExistsError('New independent output required')
    if free < 6000 or available_gib() < 32 or shutil.disk_usage(args.output.parent).free < 10*2**30:
        raise RuntimeError('Need6GB GPU,32GiB RAM,10GiB disk headroom')
    args.output.mkdir()
    lock = (args.output/'QUEUE.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    tick = time.monotonic()
    state = dict(status='smoke', pid=os.getpid(), source_commit=args.source_commit,
                 gpu_uuid=args.gpu_uuid, physical_gpu=index, max_jobs=1, jobs=[], smoke=[])
    def save():
        state['seconds'] = time.monotonic()-tick
        tmp = args.output/'QUEUE.tmp'
        tmp.write_text(json.dumps(state, indent=2)+'\n'); tmp.replace(args.output/'QUEUE.json')
    save()
    try:
        for phase in ('smoke', 'train'):
            if phase == 'train':
                peak = state['smoke'][0]['peak_memory_mib']
                if gpu_free(args.gpu_uuid)[1] < 2000+1.5*peak or available_gib() < 32:
                    raise RuntimeError('Resource changed after measured smoke; no hidden reconfiguration')
            command = worker_command(args, phase)
            log = args.output/(phase+'.log')
            with log.open('w') as f:
                proc = subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu_uuid, OMP_NUM_THREADS='1',
                             MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
            job = dict(pid=proc.pid, command=command, status='running', log=str(log))
            state['smoke' if phase == 'smoke' else 'jobs'].append(job)
            state['status'] = phase
            save()
            file = args.output/('smoke' if phase == 'smoke' else 'runs')/MODE/'SUMMARY.json'
            while True:
                summary = json.loads(file.read_text()) if file.exists() else {}
                job.update(epoch=summary.get('epoch'), worker_status=summary.get('status'))
                code = proc.poll()
                if code is not None:
                    job.update(returncode=code, peak_memory_mib=summary.get('peak_memory_mib'),
                        seconds=summary.get('seconds'), status='complete' if code == 0
                        and summary.get('status') == 'complete' else 'failed')
                    save()
                    if job['status'] != 'complete':
                        raise RuntimeError('Worker failed: '+phase)
                    break
                save()
                time.sleep(10)
            if phase == 'train':
                summarize(args.output)
        state['status'] = 'complete'
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
