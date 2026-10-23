"""Finite five-job cmerc sentiment queue; no automatic retry or peer termination."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import time

from experiments.osram_mosi_gcnet_router_3seed_20261019 import run as base
from experiments.osram_text_path_router_3seed_20261019.resource_gate import available_gib, write

ROOT = Path(__file__).resolve().parents[2]
PYTHON = '/data2/yb/conda_envs/cmerc-repro/bin/python'
SOURCE = '0da38b3326fdad09da0195e49b8b7d6424b4f613'
GPU = 'GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e'
REFERENCE = '/data2/yb/remote_experiments/osram_mosi_cmerc_20261023/execution_v2'
MODULE = 'experiments.osram_cmerc_sentiment_3seed_20261023.worker'


def plan_jobs(output, source_commit, gpu_uuid):
    jobs = []
    for dataset, seeds in (('CMUMOSI', (67, 68)), ('CMUMOSEI', (66, 67, 68))):
        for seed in seeds:
            path = Path(output) / 'runs' / dataset / f'seed_{seed}' / 'flat'
            job = base.make_job(MODULE, path, source_commit, gpu_uuid, 'flat', seed, dataset)
            job['command'][0] = PYTHON
            job['command'] += ['--dataset', dataset]
            jobs.append(job)
    return jobs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--gpu-uuid', default=GPU)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--poll-seconds', type=float, default=20)
    args = parser.parse_args(argv)
    if args.poll_seconds <= 0:
        raise ValueError('poll-seconds must be positive')
    args.output = args.output.resolve()
    jobs = plan_jobs(args.output, args.source_commit, args.gpu_uuid)
    if args.output.exists() and not args.resume:
        raise FileExistsError('Fresh output required; --resume explicitly restores this queue')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    lock = (args.output / 'QUEUE.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = args.output / 'QUEUE.json'
    if args.resume:
        state = json.loads(path.read_text())
        if (state['source_commit'] != args.source_commit or state['gpu_uuid'] != args.gpu_uuid
                or state['reference'] != REFERENCE):
            raise ValueError('Resume identity differs')
        for old, new in zip(state['jobs'], jobs):
            if any(old[k] != new[k] for k in ('dataset', 'seed', 'mode', 'output', 'source_commit', 'gpu_uuid')):
                raise ValueError('Resume plan differs')
            if [v for v in old['command'] if v != '--resume'] != new['command']:
                raise ValueError('Resume command differs')
        if len(state['jobs']) != 5 or state['status'] == 'complete':
            raise ValueError('Wrong plan or already completed queue')
        base.recover_jobs(state)
    else:
        state = dict(status='running', source_commit=args.source_commit, numerical_commit=SOURCE, gpu_uuid=args.gpu_uuid,
                     reference=REFERENCE, jobs=jobs, started_at=time.time(), pid=os.getpid(),
                     label='INTERNAL TEST-ORACLE SCREEN', maximum_parallel=5,
                     minimum_ram_gib=64, minimum_disk_gib=20,
                     admission_gpu_gib=6, stagger_seconds=4)
    children = {}
    base.ROOT = ROOT

    def record():
        state['updated_at'] = time.time()
        write(path, state)

    try:
        record()
        while True:
            for index, process in list(children.items()):
                if base.poll(process, state['jobs'][index]):
                    del children[index]
            failed = any(j['status'] == 'failed' for j in state['jobs'])
            pending = [i for i, j in enumerate(state['jobs']) if j['status'] == 'pending']
            if failed:
                state['status'] = 'draining_failed'
                if not children:
                    state['status'] = 'failed'
                    record()
                    raise RuntimeError('Child failed; pending jobs stopped, no automatic retries')
            elif not children and not pending:
                from .report import write_report
                write_report(args.output, state)
                state.update(status='complete', finished_at=time.time())
                record()
                return
            else:
                found = base.devices([args.gpu_uuid])[args.gpu_uuid]
                free_gpu = (found['total_mib'] - found['used_mib']) / 1024
                ram = available_gib()
                disk = shutil.disk_usage(args.output).free / 2**30
                state.update(available_ram_gib=ram, disk_free_gib=disk, gpu_free_gib=free_gpu)
                if pending and len(children) < 5 and free_gpu >= 6 and ram >= 64 and disk >= 20:
                    index = pending[0]
                    children[index] = base.launch(state['jobs'][index])
                    state['status'] = 'running'
                    record()
                    time.sleep(4)
                    continue
                state['status'] = 'running' if children else 'waiting_for_resources'
            record()
            time.sleep(args.poll_seconds)
    except BaseException as error:
        state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=repr(error))
        record()
        raise


if __name__ == '__main__':
    main()
