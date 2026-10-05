"""Finite four-run queue. GPU6 only, admission from live peak reservations."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiments.osram_core20_20261005.dispatch import gpus
from experiments.osram_core20_20261005.run import now, sha, write

METHODS = ('R02', 'R03', 'R12', 'R18')
ESTIMATES = {'R02': 4096, 'R03': 12288, 'R12': 4096, 'R18': 4096}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--data-manifest', type=Path, required=True)
    p.add_argument('--seal-commit')
    args = p.parse_args()
    source = Path(__file__).resolve().parents[2]
    if args.seal_commit:
        files = [str(f.relative_to(source)) for f in source.rglob('*') if f.is_file()
                 and '__pycache__' not in f.parts and '.pytest_cache' not in f.parts
                 and f.name != 'SNAPSHOT.json']
        write(source / 'SNAPSHOT.json', dict(code_commit=args.seal_commit,
              source_sha256={f: sha(source / f) for f in files if (source / f).is_file()}))
        return
    args.root.mkdir(parents=True, exist_ok=True)
    lock = (args.root / 'DISPATCH.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (args.root / 'DISPATCH.json').exists():
        raise FileExistsError('Inspect existing queue; do not blindly restart')
    rows = [dict(method=m, seed=66, status='pending') for m in METHODS]
    jobs = {}
    while True:
        for method, (child, stream) in list(jobs.items()):
            if child.poll() is None:
                continue
            stream.close()
            path = args.root / method / 'PROVENANCE.json'
            record = json.loads(path.read_text()) if path.exists() else {}
            row = next(r for r in rows if r['method'] == method)
            row.update(status='complete' if child.returncode == 0 and record.get('outputs_verified') else 'failed',
                       exit_code=child.returncode, finished_utc=now(), error=record.get('error'))
            del jobs[method]
        uuid, used, total = gpus()[6]
        reserves = {}
        for method in jobs:
            path = args.root / method / 'RESOURCE.json'
            record = json.loads(path.read_text()) if path.exists() else {}
            peak = record.get('peak_reserved_mib', 0)
            reserves[method] = max(ESTIMATES[method], int(peak * 1.2 + 512))
        processes = subprocess.check_output(['nvidia-smi',
            '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'], text=True)
        own_pids = {child.pid for child, _ in jobs.values()}
        allocations = sum(int(memory.strip()) for line in processes.splitlines()
                          for pid, memory in [line.split(',')]
                          if int(pid.strip()) in own_pids and memory.strip().isdigit())
        # Preserve capacity for all our eventual peaks, plus currently visible
        # external jobs. Never terminate an external process or change batch.
        accounted = max(used, max(0, used - allocations) + sum(reserves.values()))
        if shutil.disk_usage(args.root).free < 40 * 2**30:
            raise RuntimeError('disk reserve below40GiB')
        for row in rows:
            if row['status'] != 'pending' or accounted + ESTIMATES[row['method']] + 2048 > total:
                continue
            output = args.root / row['method']
            output.mkdir(exist_ok=False)
            command = [sys.executable, '-u', '-m', 'experiments.osram_core20_20261005.run',
                '--method', row['method'], '--reference', str(args.reference),
                '--data-manifest', str(args.data_manifest), '--output', str(output), '--gpu-uuid', uuid]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            stream = (output / 'train.log').open('x')
            child = subprocess.Popen(command, cwd=source, env=env, stdout=stream, stderr=subprocess.STDOUT)
            jobs[row['method']] = child, stream
            row.update(status='running', pid=child.pid, command=command, gpu=6,
                       gpu_uuid=uuid, started_utc=now())
            print(now(), row['method'], child.pid, flush=True)
            break
        write(args.root / 'DISPATCH.json', dict(at=now(), pid=os.getpid(), runs=rows,
              label='INTERNAL DIAGNOSTIC ONLY', initial_reserve_mib=ESTIMATES))
        if not jobs and all(r['status'] in ('complete', 'failed') for r in rows):
            return
        time.sleep(15)


if __name__ == '__main__':
    main()
