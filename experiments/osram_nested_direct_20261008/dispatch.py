"""One direct-replacement confirmation, existing protocol, persistent single-owner process."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiments.osram_core20_20261005.run import LABEL, now, write
from experiments.osram_nested_sweep_20261005.dispatch import scores, REFERENCE
from experiments.osram_readout_top3_3seed_20261005.dispatch import read, ORIGINALS

GPU = 'GPU-e4cafb17-818e-216a-b94a-7440063a9153'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--data-manifest', type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'DISPATCH.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / 'DISPATCH.json').exists():
        raise FileExistsError('Inspect prior process/outputs; do not duplicate or silently resume')
    source = Path(__file__).resolve().parents[2]
    state = dict(label=LABEL, status='pending', dispatcher_pid=os.getpid(), source=str(source),
                 seed=66, method='nested_gnn_direct_evidence', gpu=6, gpu_uuid=GPU)
    while True:
        raw = subprocess.check_output(['nvidia-smi', '--id=6',
             '--query-gpu=index,uuid,memory.free', '--format=csv,noheader,nounits'], text=True)
        index, identifier, free = [v.strip() for v in raw.split(',')]
        if index != '6' or identifier != GPU:
            raise ValueError('Healthy GPU6 UUID mismatch; GPU4 forbidden')
        write(root / 'DISPATCH.json', state)
        if float(free) >= 8048 and shutil.disk_usage(root).free / 2**30 >= 26:
            break
        time.sleep(15)
    output = root / 'seed_66'
    output.mkdir(exist_ok=False)
    command = [sys.executable, '-u', '-m', 'experiments.osram_core20_20261005.run',
               '--method', state['method'], '--seed', '66', '--reference', str(REFERENCE / 'config.json'),
               '--data-manifest', str(args.data_manifest), '--output', str(output), '--gpu-uuid', GPU]
    log = output / 'train.log'
    with log.open('x') as stream:
        child = subprocess.Popen(command, cwd=source, stdout=stream, stderr=subprocess.STDOUT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=GPU,
                     GCNET_DATASET_ROOT=read(args.data_manifest)['dataset_root'],
                     OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'))
        state.update(status='running', pid=child.pid, command=command, output=str(output),
                     log=str(log), started_utc=now(), free_mib_at_launch=float(free))
        write(root / 'DISPATCH.json', state)
        exit_code = child.wait()
    provenance = output / 'PROVENANCE.json'
    prov = read(provenance) if provenance.exists() else {}
    state.update(status='complete' if exit_code == 0 and prov.get('outputs_verified') else 'failed',
                 exit_code=exit_code, finished_utc=now(), error=prov.get('error'))
    write(root / 'DISPATCH.json', state)
    comparisons = {}
    if state['status'] == 'complete':
        for name, path in (('flat', REFERENCE),
                           ('old_nested', ORIGINALS['nested_gnn_rooted_evidence']), ('direct', output)):
            try:
                comparisons[name] = dict(status='available', output=str(path), **scores(path))
            except (OSError, ValueError, KeyError) as error:
                comparisons[name] = dict(status='unavailable', error=repr(error))
    write(root / 'SUMMARY.json', dict(label=LABEL, run=state, comparisons=comparisons))


if __name__ == '__main__':
    main()
