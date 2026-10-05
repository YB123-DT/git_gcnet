"""Locked top-three confirmation: reuse seed66, run six additional jobs on GPU6."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .run import now, sha, write
from .dispatch import gpus

SELECTED = ('C20', 'C07', 'C08')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--reference-root', type=Path, required=True)
    p.add_argument('--data-manifest', type=Path, required=True)
    p.add_argument('--original-runs', type=Path, required=True)
    p.add_argument('--seal-commit')
    args = p.parse_args()
    source = Path(__file__).resolve().parents[2]
    if args.seal_commit:
        snapshot = json.loads((source / 'SNAPSHOT.json').read_text())
        snapshot.setdefault('original_model_commit', snapshot['code_commit'])
        snapshot['code_commit'] = args.seal_commit
        for name in ('experiments/osram_core20_20261005/run.py', 'experiments/osram_core20_20261005/multiseed.py'):
            snapshot['source_sha256'][name] = sha(source / name)
        write(source / 'SNAPSHOT.json', snapshot)
        return
    args.root.mkdir(parents=True, exist_ok=True)
    lock = (args.root / 'DISPATCH.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (args.root / 'DISPATCH.json').exists():
        raise FileExistsError('Inspect existing runs before restarting')
    rows = [dict(method=m, seed=s, status='pending') for m in SELECTED for s in (67, 68)]
    jobs = {}
    while True:
        for key, (child, stream, row) in list(jobs.items()):
            if child.poll() is None:
                continue
            stream.close()
            output = args.root / row['method'] / f"seed_{row['seed']}"
            prov = json.loads((output / 'PROVENANCE.json').read_text()) if (output / 'PROVENANCE.json').exists() else {}
            row.update(status='complete' if child.returncode == 0 and prov.get('outputs_verified') else 'failed',
                       exit_code=child.returncode, finished_utc=now(), error=prov.get('error'))
            del jobs[key]
        uuid, used, total = gpus()[6]
        original = json.loads((args.original_runs / 'DISPATCH.json').read_text())
        original_active = [r for r in original['runs'] if r['status'] == 'running']
        original_remaining = any(r['status'] in ('running', 'pending') for r in original['runs'])
        original_reserve = 0
        for r in original_active:
            resource = args.original_runs / r['method'] / 'RESOURCE.json'
            peak = json.loads(resource.read_text())['peak_reserved_mib'] if resource.exists() else 2000
            original_reserve += int(peak * 1.2 + 512)
        own_reserve = 0
        for _, (_, _, row) in jobs.items():
            resource = args.root / row['method'] / f"seed_{row['seed']}" / 'RESOURCE.json'
            original_resource = args.original_runs / row['method'] / 'RESOURCE.json'
            peak = json.loads(resource.read_text())['peak_reserved_mib'] if resource.exists() else json.loads(original_resource.read_text())['peak_reserved_mib']
            own_reserve += max(3000, int(peak * 1.2 + 512))
        if not any(row['method'] == 'C08' for _, _, row in jobs.values()):
            for row in rows:
                if row['status'] != 'pending':
                    continue
                heavy = row['method'] == 'C08'
                if heavy and (jobs or original_remaining or used > 1000):
                    continue
                original_peak = json.loads((args.original_runs / row['method'] / 'RESOURCE.json').read_text())['peak_reserved_mib']
                reserve = max(18000 if heavy else 3000, int(original_peak * 1.2 + 512))
                if len(jobs) >= 4 or max(used, original_reserve + own_reserve) + reserve + 2048 > total:
                    continue
                output = args.root / row['method'] / f"seed_{row['seed']}"
                output.mkdir(parents=True, exist_ok=False)
                command = [sys.executable, '-m', 'experiments.osram_core20_20261005.run',
                           '--method', row['method'], '--seed', str(row['seed']),
                           '--reference', str(args.reference_root / f"seed_{row['seed']}" / 'config.json'),
                           '--data-manifest', str(args.data_manifest), '--output', str(output), '--gpu-uuid', uuid]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
                stream = (output / 'train.log').open('x')
                child = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT, cwd=source)
                row.update(status='running', pid=child.pid, command=command, started_utc=now(), gpu=6)
                jobs[(row['method'], row['seed'])] = child, stream, row
                print(now(), row['method'], row['seed'], child.pid, flush=True)
                # One admission per iteration: observe real allocation before adding more.
                break
        write(args.root / 'DISPATCH.json', dict(at=now(), pid=os.getpid(), selected=SELECTED,
              seed66_reused=True, original_seed66_runs=str(args.original_runs), runs=rows,
              selection='completed C01-C20 method mean8 descending; controls excluded; locked before confirmation'))
        if not jobs and all(r['status'] in ('complete', 'failed') for r in rows):
            break
        time.sleep(15)


if __name__ == '__main__':
    main()
