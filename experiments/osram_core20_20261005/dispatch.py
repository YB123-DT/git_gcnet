"""Bounded persistent queue, independent output roots, healthy UUIDs only.

Resource reservations are conservative estimates, not measured runtime claims.
Heavy algorithms run alone until real peaks are available; never change batch.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .run import METHODS, CONTROLS, now, write

AUTHORIZED = {6: 'GPU-e4cafb17-818e-216a-b94a-7440063a9153'}
HEAVY = {'C06', 'C08', 'C09', 'C15', 'C16', 'C19'}
MAX_CONCURRENT = 11
RESERVE_MIB = 2000  # initial estimate only; live training records override it


def gpus():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.total',
                                     '--format=csv,noheader,nounits'], text=True)
    result = {}
    for line in output.splitlines():
        index, uuid, used, total = [x.strip() for x in line.split(',')]
        index = int(index)
        if index in AUTHORIZED:
            if uuid != AUTHORIZED[index] or index == 4:
                raise ValueError('physical GPU whitelist mismatch')
            result[index] = (uuid, int(used), int(total))
    if set(result) != set(AUTHORIZED):
        raise ValueError('authorized devices missing')
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--data-manifest', type=Path, required=True)
    args = p.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    reservations = {}
    lock = (args.root / 'DISPATCH.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    jobs = {}
    rows = {method: {'method': method, 'status': 'pending'} for method in METHODS + CONTROLS}
    if (args.root / 'DISPATCH.json').exists():
        prior = json.loads((args.root / 'DISPATCH.json').read_text())
        for row in prior['runs']:
            if row['status'] in ('running', 'failed'):
                # No blind relaunch after dispatcher crash. Record/inspect PID
                # and compatible last_training state first.
                raise RuntimeError('existing interrupted/failed queue requires audit before restart')
            rows[row['method']] = row
    while True:
        for method, (process, gpu, stream) in list(jobs.items()):
            code = process.poll()
            if code is None:
                continue
            stream.close()
            record_path = args.root / method / 'PROVENANCE.json'
            record = json.loads(record_path.read_text()) if record_path.exists() else {}
            complete = code == 0 and record.get('status') == 'complete' and record.get('outputs_verified')
            rows[method].update(status='complete' if complete else 'failed', exit_code=code,
                                finished_utc=now(), error=record.get('error'))
            del jobs[method]
        devices = gpus()
        for method in jobs:
            resource = args.root / method / 'RESOURCE.json'
            if resource.exists():
                peak = json.loads(resource.read_text())['peak_reserved_mib']
                reservations[method] = max(RESERVE_MIB, int(peak * 1.2 + 512))
        if shutil.disk_usage(args.root).free < 40 * 2**30:
            raise RuntimeError('disk reserve below40GiB; no new runs submitted')
        for method, row in rows.items():
            if row['status'] != 'pending':
                continue
            for gpu, (uuid, used, total) in devices.items():
                owned = [m for m, (_, g, _) in jobs.items() if g == gpu]
                if len(owned) >= MAX_CONCURRENT or any(m in HEAVY for m in owned):
                    continue
                if method in HEAVY:
                    if owned or used > 1000:
                        continue
                else:
                    # Reserve eventual peak of every already launched run too;
                    # do not fill based on just-started process memory usage.
                    accounted = max(used, sum(reservations.get(m, RESERVE_MIB) for m in owned))
                    if accounted + reservations.get(method, RESERVE_MIB) + 2048 > total:
                        continue
                output = args.root / method
                output.mkdir(exist_ok=True)
                stream = (output / 'train.log').open('a')
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                           OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')
                command = [sys.executable, '-m', 'experiments.osram_core20_20261005.run',
                           '--method', method, '--reference', str(args.reference),
                           '--data-manifest', str(args.data_manifest), '--output', str(output), '--gpu-uuid', uuid]
                process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
                jobs[method] = (process, gpu, stream)
                row.update(status='running', pid=process.pid, gpu=gpu, gpu_uuid=uuid,
                           started_utc=now(), command=command, resource_reservation_mib=reservations.get(method, RESERVE_MIB))
                print(f'{now()} launched {method} hostGPU{gpu} pid={process.pid}', flush=True)
                break
        write(args.root / 'DISPATCH.json', {'at': now(), 'pid': os.getpid(),
            'runs': list(rows.values()), 'all_terminal': all(r['status'] in ('complete', 'failed') for r in rows.values()),
            'resource_policy': 'GPU6 only; max11 light; initial 2GiB estimate then live training peaks+20%+512MiB; heavy exclusive; no extra smoke',
            'authorized_gpus': AUTHORIZED})
        if not jobs and all(r['status'] in ('complete', 'failed') for r in rows.values()):
            break
        time.sleep(15)


if __name__ == '__main__':
    main()
