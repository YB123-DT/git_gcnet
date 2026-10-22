"""Reuse66; smoke then train only67/68 concurrently on one healthy GPU."""
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
from experiments.osram_mosi_v3_capacity_20261022.run import admission_required, resident_memory
from .report import summarize


def worker_command(args, seed, phase):
    if seed not in (67, 68) or phase not in ('smoke', 'train'):
        raise ValueError('Supplement only67/68 smoke/train; never repeat66')
    return [sys.executable, '-u', '-m', 'experiments.osram_mosi_v3_capacity_3seed_20261022.worker',
            '--mode', 'f512_v128', '--phase', phase, '--seed', str(seed),
            '--gpu-uuid', args.gpu_uuid, '--source-commit', args.source_commit,
            '--output', str(args.output / ('smoke' if phase == 'smoke' else 'runs') / f'seed_{seed}')]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu-uuid', required=True)
    p.add_argument('--source-commit', required=True)
    args = p.parse_args()
    from experiments.osram_text_path_router_3seed_20261019.resource_gate import available_gib
    index, free = gpu_free(args.gpu_uuid)
    if args.output.exists():
        raise FileExistsError('Fresh supplement required')
    if free < 12000 or available_gib() < 64 or shutil.disk_usage(args.output.parent).free < 20 * 2**30:
        raise RuntimeError('Need12GB GPU,64GiB RAM and20GiB disk headroom')
    args.output.mkdir()
    lock = (args.output / 'QUEUE.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = dict(status='smoke', source_commit=args.source_commit, pid=os.getpid(),
                 physical_gpu=index, gpu_uuid=args.gpu_uuid, max_parallel=2, smokes=[], jobs=[])
    tick = time.monotonic()
    def save():
        state['seconds'] = time.monotonic() - tick
        tmp = args.output / 'QUEUE.tmp'; tmp.write_text(json.dumps(state, indent=2) + '\n')
        tmp.replace(args.output / 'QUEUE.json')
    save()
    try:
        for seed in (67, 68):
            if gpu_free(args.gpu_uuid)[1] < 12000 or available_gib() < 64:
                raise RuntimeError('Resources changed before smoke')
            log = args.output / f'smoke_seed_{seed}.log'
            with log.open('w') as stream:
                done = subprocess.run(worker_command(args, seed, 'smoke'), stdout=stream, stderr=subprocess.STDOUT)
            f = args.output / 'smoke' / f'seed_{seed}' / 'SUMMARY.json'
            s = json.loads(f.read_text()) if f.exists() else {}
            state['smokes'].append(dict(seed=seed, returncode=done.returncode, status=s.get('status'),
                peak_memory_mib=s.get('peak_memory_mib'), seconds=s.get('seconds'), log=str(log)))
            save()
            if done.returncode or s.get('status') != 'complete':
                raise RuntimeError('Real MOSI smoke failed: ' + str(seed))
        peaks = {s['seed']: s['peak_memory_mib'] for s in state['smokes']}
        need = 2000 + 1.5 * sum(peaks.values())
        if gpu_free(args.gpu_uuid)[1] < need or available_gib() < 64:
            raise RuntimeError('Measured headroom cannot admit both seeds')
        state.update(status='running', estimated_required_mib=need); save()
        procs = []
        for seed in (67, 68):
            resident = resident_memory()
            needed = admission_required(peaks[seed], [(peaks[j['seed']], resident.get(j['pid'], 0)) for j in state['jobs']])
            if gpu_free(args.gpu_uuid)[1] < needed or available_gib() < 64:
                raise RuntimeError('Resource changed; do not kill started peer')
            log = args.output / f'train_seed_{seed}.log'; cmd = worker_command(args, seed, 'train')
            with log.open('w') as stream:
                proc = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu_uuid, OMP_NUM_THREADS='1',
                             MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
            procs.append(proc)
            state['jobs'].append(dict(seed=seed, pid=proc.pid, status='running', command=cmd,
                log=str(log), output=str(args.output / 'runs' / f'seed_{seed}'))); save()
        while True:
            active = False
            for proc, job in zip(procs, state['jobs']):
                f = Path(job['output']) / 'SUMMARY.json'; s = json.loads(f.read_text()) if f.exists() else {}
                code = proc.poll(); job.update(epoch=s.get('epoch'), worker_status=s.get('status'))
                if code is None:
                    active = True
                else:
                    job.update(returncode=code, status='complete' if code == 0 and s.get('status') == 'complete' else 'failed')
            save(); summarize(args.output)
            if not active:
                break
            time.sleep(15)
        state['status'] = 'complete' if all(j['status'] == 'complete' for j in state['jobs']) else 'failed'
    except BaseException as error:
        state.update(status='failed', error=repr(error)); raise
    finally:
        save()


if __name__ == '__main__':
    main()
