"""Two concurrent Flat-only jobs; no extra seed, model, smoke or HPO queue."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import time

from experiments.osram_mosi_gcnet_router_3seed_20261019.run import (
    BAD_UUID, make_job, devices, launch, poll, recover_jobs)
from experiments.osram_text_path_router_3seed_20261019.resource_gate import available_gib, write

LEARNING_RATES = (5e-4, 3e-4)
MODULE = 'experiments.osram_mosi_mid_lr_20261021.worker'


def plan_jobs(output, sha, gpus):
    if len(gpus) != 1 or not gpus[0].startswith('GPU-') or gpus[0] == BAD_UUID:
        raise ValueError('One healthy GPU UUID required')
    jobs = []
    for lr in LEARNING_RATES:
        label = f'{lr:g}'
        job = make_job(MODULE, Path(output)/'runs'/'MOSI'/'seed_66'/('flat_lr_' + label),
                       sha, gpus[0], 'flat', 66, 'CMUMOSI')
        job['command'] += ['--backbone-lr', label]
        job.update(backbone_lr=lr, final_backbone_lr=lr / 10,
                   checkpoint_selection='test-oracle-per-rate; weighted_f1')
        jobs.append(job)
    return jobs


def main(*, planner=plan_jobs, reporter=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    jobs = planner(args.output, args.source_commit, [args.gpu_uuid])
    if args.output.exists() and not args.resume:
        raise FileExistsError('Fresh queue required')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    lock = (args.output/'QUEUE.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = args.output/'QUEUE.json'
    if args.resume:
        state = json.loads(path.read_text())
        if state['source_commit'] != args.source_commit or state['gpu_uuid'] != args.gpu_uuid:
            raise ValueError('Resume identity differs')
        if state['status'] == 'complete':
            raise ValueError('Completed queue must not repeat')
        if [{k: j[k] for k in ('seed', 'mode', 'output', 'backbone_lr', 'final_backbone_lr')}
                for j in state['jobs']] != [{k: j[k] for k in
                ('seed', 'mode', 'output', 'backbone_lr', 'final_backbone_lr')} for j in jobs]:
            raise ValueError('Resume plan differs')
        recover_jobs(state)
    else:
        state = dict(status='preflight', source_commit=args.source_commit, gpu_uuid=args.gpu_uuid,
                     pid=os.getpid(), jobs=jobs, started_at=time.time(), maximum_parallel=2,
                     verification='Reuse completed same-Flat GPU validation; numerical core unchanged')
    children = {}
    try:
        state['gpu_preflight'] = devices([args.gpu_uuid])
        state['disk_free_gib'] = shutil.disk_usage(args.output).free / 2**30
        write(path, state)
        if state['disk_free_gib'] < 12:
            raise RuntimeError('Require12GiB disk; no old checkpoints will be deleted')
        while any(j['status'] in ('pending', 'running') for j in state['jobs']):
            for index, process in list(children.items()):
                if poll(process, state['jobs'][index]):
                    children.pop(index)
            failed = any(j['status'] == 'failed' for j in state['jobs'])
            for index, job in enumerate(state['jobs']):
                if failed or job['status'] != 'pending':
                    continue
                gpu = devices([args.gpu_uuid])[args.gpu_uuid]
                # Completed same-protocol Flat peak1.25GiB; reserve4GiB/job
                # plus2GiB card margin, without changing batch32 or the model.
                if gpu['total_mib'] - gpu['used_mib'] < 2*4096 + 2048 or available_gib() < 64:
                    continue
                children[index] = launch(job)
                write(path, state)
                time.sleep(2)
            state.update(status='draining_after_failure' if failed else 'training', updated_at=time.time())
            write(path, state)
            if failed and not children:
                raise RuntimeError('Child failure; no automatic retry')
            time.sleep(5)
        if reporter is None:
            from .report import write_report
            reporter = write_report
        reporter(args.output, state)
        state.update(status='complete', completed_at=time.time())
        write(path, state)
    except BaseException as error:
        state.update(status='failed', error=repr(error), surviving_children=[p.pid for p in children.values()])
        write(path, state)
        raise


if __name__ == '__main__':
    main()
