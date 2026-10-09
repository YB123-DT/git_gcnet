"""Persistent single-job launcher; healthy GPU0, independent immutable snapshot."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from experiments.osram_core20_20261005.run import write, now
from experiments.osram_frozen_nested_20261009.run import LABEL

GPU = 'GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45'
REFERENCE = '/data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full/seed_66/config.json'
DATA = '/data2/yb/remote_experiments/osram_meaningful20_round2_20261004/DATA.json'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(exist_ok=True, parents=True)
    lock = (root / 'DISPATCH.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / 'DISPATCH.json').exists():
        raise FileExistsError('inspect previous run; no duplicate or silent restart')
    row = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=index,uuid,memory.free',
                                   '--format=csv,noheader,nounits'], text=True).strip().split(',')
    assert row[0].strip() == '0' and row[1].strip() == GPU and int(row[2]) >= 8048
    command = [sys.executable, '-u', '-m', 'experiments.osram_frozen_nested_20261009.run',
               '--reference', REFERENCE, '--data-manifest', DATA, '--output', str(root / 'seed_66'), '--gpu-uuid', GPU]
    source = Path(__file__).resolve().parents[2]
    with (root / 'train.log').open('x') as stream:
        child = subprocess.Popen(command, cwd=source, stdout=stream, stderr=subprocess.STDOUT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'))
        state = dict(status='running', label=LABEL, host_gpu=0, gpu_uuid=GPU, pid=child.pid,
                     dispatcher_pid=os.getpid(), command=command, source=str(source), started_utc=now())
        write(root / 'DISPATCH.json', state)
        code = child.wait()
    provpath = root / 'seed_66/PROVENANCE.json'
    prov = json.loads(provpath.read_text()) if provpath.exists() else {}
    state.update(exit_code=code, finished_utc=now(), status='complete' if code == 0 and prov.get('outputs_verified') else 'failed')
    write(root / 'DISPATCH.json', state)


if __name__ == '__main__':
    main()
