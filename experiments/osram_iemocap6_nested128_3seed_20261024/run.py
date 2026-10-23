"""Finite IEMOCAP-6 Nested128 five-fold / three-seed expansion.

Reuse the completed seed66/fold5 run; train precisely fourteen new folds.
The requested six-process ceiling is subject to measured full-fold memory
reservations, not optimistic startup allocations. No kills or automatic retries.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

from experiments.osram_mosi_gcnet_router_3seed_20261019 import run as base
from experiments.osram_nested_cmerc_tuning_20261023.run import (
    poll, recover_jobs, verify_source_snapshot,
)
from experiments.osram_text_path_router_3seed_20261019.resource_gate import available_gib, write

ROOT = Path(__file__).resolve().parents[2]
PYTHON = '/data2/yb/conda_envs/cmerc-repro/bin/python'
MODULE = 'experiments.osram_nested_cmerc_tuning_20261023.worker'
GPU = 'GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45'
BAD_UUID = base.BAD_UUID
REUSED = '/data2/yb/remote_experiments/osram_nested_cmerc_tuning_20261023/execution_v1/runs/IEMOCAPSix/fold_5/seed_66/nested_dim128'
RESERVE_MIB = 6000
MARGIN_MIB = 1250


def validate_gpu_uuid(uuid):
    if uuid != GPU:
        raise ValueError('This queue only authorizes physical GPU0; GPU4 is forbidden')


def plan_jobs(output, source_commit, gpu_uuid):
    validate_gpu_uuid(gpu_uuid)
    first = [(66, 1), (67, 5), (68, 5), (66, 2), (67, 1), (68, 1)]
    pairs = first + [(seed, fold) for seed in (66, 67, 68) for fold in range(1, 6)
                     if (seed, fold) not in first and (seed, fold) != (66, 5)]
    jobs = []
    for seed, fold in pairs:
        output_path = Path(output) / 'runs' / 'IEMOCAPSix' / f'fold_{fold}' / f'seed_{seed}' / 'nested_dim128'
        command = [PYTHON, '-u', '-m', MODULE, '--dataset', 'IEMOCAPSix',
                   '--method', 'nested_dim128', '--seed', str(seed), '--fold', str(fold),
                   '--phase', 'train', '--source-commit', source_commit, '--gpu-uuid', gpu_uuid,
                   '--output', str(output_path)]
        jobs.append(dict(dataset='IEMOCAPSix', method='nested_dim128', seed=seed, fold=fold,
                         epochs=100, learning_rate=1e-3, phase='train', source_commit=source_commit,
                         gpu_uuid=gpu_uuid, output=str(output_path), log=str(output_path) + '.log',
                         status='pending', command=command, mask_protocol='gcnet-random-mask-v1'))
    return jobs


def can_admit(*, total_mib, used_mib, own_used_mib, active_count, ram_gib, disk_gib, parallel=6):
    if not 1 <= parallel <= 6:
        raise ValueError('Parallel ceiling must be between one and six')
    if active_count < 0 or min(total_mib, used_mib, own_used_mib) < 0:
        raise ValueError('GPU accounting must be nonnegative')
    if not all(math.isfinite(float(v)) for v in (total_mib, used_mib, own_used_mib, ram_gib, disk_gib)):
        return False
    # Already allocated memory belonging to our workers is replaced by their
    # full reservations, not charged twice. Unknown/unrelated usage stays charged.
    external_mib = max(0, used_mib - min(own_used_mib, used_mib))
    reserved_after_admission = max(active_count * RESERVE_MIB, own_used_mib) + RESERVE_MIB
    return (active_count < parallel and ram_gib >= 64 and disk_gib >= 25
            and external_mib + reserved_after_admission + MARGIN_MIB <= total_mib)


def own_gpu_memory_mib(uuid, pids):
    """Only subtract allocations of this controller's recorded living children."""
    if not pids:
        return 0
    text = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory',
         '--format=csv,noheader,nounits'], text=True, timeout=20)
    own_pids = set(pids)
    allocated = 0
    for line in text.strip().splitlines():
        pid, process_uuid, used = [part.strip() for part in line.split(',')]
        if process_uuid == uuid and int(pid) in own_pids:
            # Unavailable process accounting fails closed instead of guessing.
            allocated += int(used)
    return allocated


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--gpu-uuid', default=GPU)
    parser.add_argument('--parallel', type=int, default=6)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--poll-seconds', type=float, default=20)
    args = parser.parse_args(argv)
    if not 1 <= args.parallel <= 6 or not math.isfinite(args.poll_seconds) or args.poll_seconds <= 0:
        raise ValueError('Require parallel 1–6 and a finite positive polling interval')
    validate_gpu_uuid(args.gpu_uuid)
    args.output = args.output.resolve()
    manifest = verify_source_snapshot(ROOT, args.source_commit)
    from .report import validate_reused, write_report
    validate_reused(REUSED)
    reused = dict(dataset='IEMOCAPSix', method='nested_dim128', seed=66, fold=5,
                  epochs=100, status='reused_complete', output=REUSED)
    jobs = plan_jobs(args.output, args.source_commit, args.gpu_uuid)
    devices = base.devices([args.gpu_uuid])
    if args.output.exists() and not args.resume:
        raise FileExistsError('Existing output requires explicit --resume; no duplicate queue submission')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    lock = (args.output / 'QUEUE.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    queue_path = args.output / 'QUEUE.json'
    identity = dict(source_commit=args.source_commit, gpu_uuid=args.gpu_uuid,
                    maximum_parallel=args.parallel, reserve_mib=RESERVE_MIB, margin_mib=MARGIN_MIB,
                    source_manifest_sha256=manifest['manifest_sha256'], reused=reused,
                    planned_jobs=copy.deepcopy(jobs))
    if args.resume:
        state = json.loads(queue_path.read_text())
        if state.get('identity') != identity or state['status'] == 'complete':
            raise ValueError('Resume identity differs or this finite queue is already complete')
        recover_jobs(state)
        state.update(pid=os.getpid(), recovery_started_at=time.time(), status='resuming')
        state.pop('error', None)
    else:
        state = dict(status='preflight', identity=identity, source_commit=args.source_commit,
                     gpu_uuid=args.gpu_uuid, maximum_parallel=args.parallel, pid=os.getpid(),
                     jobs=jobs, reused=reused, maximum_new_jobs=14, total_seed_fold_results=15,
                     started_at=time.time(), label='INTERNAL TEST-ORACLE SCREEN',
                     minimum_ram_gib=64, minimum_disk_gib=25, reserve_mib=RESERVE_MIB,
                     admission_margin_mib=MARGIN_MIB, gpu_preflight=devices)
        write(args.output / 'SOURCE_MANIFEST.json', manifest)
    children = {}
    base.ROOT = ROOT

    def record():
        state['updated_at'] = time.time()
        write(queue_path, state)

    def drain():
        for index, process in list(children.items()):
            process.wait()
            poll(process, state['jobs'][index])
            del children[index]
            record()

    try:
        record()
        while True:
            for index, process in list(children.items()):
                if poll(process, state['jobs'][index]):
                    del children[index]
            failed = any(job['status'] == 'failed' for job in state['jobs'])
            pending = [i for i, job in enumerate(state['jobs']) if job['status'] == 'pending']
            if failed:
                state['status'] = 'failure_draining_active'
                if not children:
                    for index in pending:
                        state['jobs'][index].update(status='blocked', error='Earlier child failed; no automatic retry')
                    raise RuntimeError('A child failed; remaining unstarted jobs are blocked')
            elif not children and not pending:
                write_report(args.output, state)
                state.update(status='complete', finished_at=time.time())
                record()
                return
            else:
                found = base.devices([args.gpu_uuid])[args.gpu_uuid]
                resources = dict(total_mib=found['total_mib'], used_mib=found['used_mib'],
                                 own_used_mib=own_gpu_memory_mib(args.gpu_uuid, [p.pid for p in children.values()]),
                                 active_count=len(children), ram_gib=available_gib(),
                                 disk_gib=shutil.disk_usage(args.output).free / 2**30, parallel=args.parallel)
                state['resources'] = resources
                if pending and can_admit(**resources):
                    if verify_source_snapshot(ROOT, args.source_commit) != manifest:
                        raise ValueError('Immutable source changed while queue was active')
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
        state.update(status='failure_draining_active', error=repr(error))
        record()
        drain()
        state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                     finished_at=time.time())
        record()
        raise


if __name__ == '__main__':
    main()
