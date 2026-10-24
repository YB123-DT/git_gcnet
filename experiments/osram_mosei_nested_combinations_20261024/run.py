"""Exactly four MOSEI Nested combinations; persistent GPU2-only finite queue."""
import argparse
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import time

from experiments.osram_nested_cmerc_tuning_20261023 import run as common
from experiments.osram_nested_cmerc_tuning_20261023.worker import MOSEI_COMBINATIONS
from experiments.osram_iemocap6_nested128_3seed_20261024.run import own_gpu_memory_mib

ROOT = Path(__file__).resolve().parents[2]
GPU = 'GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4'
BAD_UUID = common.BAD_UUID
METHODS = MOSEI_COMBINATIONS
RESERVE_MIB = 6144
MARGIN_MIB = 2048


def plan_jobs(output, source_commit, gpu_uuid):
    if gpu_uuid != GPU:
        raise ValueError('Only biggpu physical GPU2 is authorized; GPU4 forbidden')
    jobs = []
    for method in METHODS:
        path = Path(output) / 'runs/CMUMOSEI/fold_1/seed_66' / method
        command = [common.PYTHON, '-u', '-m', common.MODULE, '--dataset', 'CMUMOSEI',
                   '--method', method, '--seed', '66', '--fold', '1', '--phase', 'train',
                   '--gpu-uuid', gpu_uuid, '--source-commit', source_commit, '--output', str(path)]
        jobs.append(dict(dataset='CMUMOSEI', method=method, seed=66, fold=1, epochs=60,
            learning_rate=1e-4, phase='train', source_commit=source_commit, gpu_uuid=gpu_uuid,
            output=str(path), log=str(path)+'.log', status='pending', command=command,
            mask_protocol='gcnet-random-mask-v1'))
    return jobs


def can_admit(**r):
    if any(not math.isfinite(float(v)) for v in r.values()):
        return False
    external = max(0, r['used_mib'] - min(r['own_used_mib'], r['used_mib']))
    reserved = max(r['active_count'] * RESERVE_MIB, r['own_used_mib'])
    return (r['active_count'] < 4 and r['ram_gib'] >= 64 and r['disk_gib'] >= 25
            and external + reserved + RESERVE_MIB + MARGIN_MIB <= r['total_mib'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    manifest = common.verify_source_snapshot(ROOT, args.source_commit)
    from .report import validate_source_extension
    extension_proof = validate_source_extension(ROOT)
    jobs = plan_jobs(args.output, args.source_commit, GPU)
    device = common.base.devices([GPU])[GPU]
    if device['index'] != 2:
        raise ValueError('GPU UUID no longer maps to physical GPU2')
    if args.output.exists() and not args.resume:
        raise FileExistsError('Independent output required; do not submit twice')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    lock = (args.output / 'QUEUE.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity = dict(source_commit=args.source_commit, source_manifest_sha256=manifest['manifest_sha256'],
        gpu_uuid=GPU, maximum_parallel=4, reserve_mib=RESERVE_MIB, margin_mib=MARGIN_MIB,
        planned_jobs=copy.deepcopy(jobs))
    queue_path = args.output / 'QUEUE.json'
    if args.resume:
        state = json.loads(queue_path.read_text())
        if state['identity'] != identity or state['status'] == 'complete':
            raise ValueError('Recovery identity differs or queue complete')
        common.recover_jobs(state)
        state.update(status='resuming', pid=os.getpid(), recovery_started_at=time.time())
        state.pop('error', None)
    else:
        state = dict(status='preflight', pid=os.getpid(), source_commit=args.source_commit,
            identity=identity, jobs=jobs, gpu_uuid=GPU, started_at=time.time(),
            source_extension_proof=extension_proof,
            label='INTERNAL TEST-ORACLE SCREEN', maximum_jobs=4, maximum_parallel=4)
        common.write(args.output / 'SOURCE_MANIFEST.json', manifest)
    children = {}
    common.base.ROOT = ROOT
    def record():
        state['updated_at'] = time.time()
        common.write(queue_path, state)
    try:
        record()
        while True:
            for i, proc in list(children.items()):
                if common.poll(proc, state['jobs'][i]):
                    del children[i]
            pending = [i for i,j in enumerate(state['jobs']) if j['status'] == 'pending']
            failed = any(j['status'] == 'failed' for j in state['jobs'])
            if failed:
                state['status'] = 'failure_draining_active'
                if not children:
                    raise RuntimeError('A child failed; no silent retry or protocol changes')
            elif not children and not pending:
                from .report import write_report
                write_report(args.output, state)
                state.update(status='complete', finished_at=time.time())
                record(); return
            else:
                dev = common.base.devices([GPU])[GPU]
                resources = dict(total_mib=dev['total_mib'], used_mib=dev['used_mib'],
                    own_used_mib=own_gpu_memory_mib(GPU, [p.pid for p in children.values()]),
                    active_count=len(children), ram_gib=common.available_gib(),
                    disk_gib=shutil.disk_usage(args.output).free/2**30)
                state['resources'] = resources
                if pending and can_admit(**resources):
                    if common.verify_source_snapshot(ROOT, args.source_commit) != manifest:
                        raise ValueError('Immutable source changed')
                    i = pending[0]
                    children[i] = common.base.launch(state['jobs'][i])
                    state['status'] = 'running'
                    record(); time.sleep(4); continue
                state['status'] = 'running' if children else 'waiting_for_resources'
            record(); time.sleep(15)
    except BaseException as error:
        state.update(status='failure_draining_active', error=repr(error)); record()
        for i,proc in list(children.items()):
            proc.wait(); common.poll(proc, state['jobs'][i]); record()
        state.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                     finished_at=time.time())
        record(); raise


if __name__ == '__main__':
    main()
