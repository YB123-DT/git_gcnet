"""Finite four-job Flat write-step queue on healthy GPU7; no Nested jobs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import time

from experiments.osram_nested_cmerc_tuning_20261023 import run as common
from experiments.osram_mosei_nested_combinations_20261024.run import can_admit
from experiments.osram_iemocap6_nested128_3seed_20261024.run import own_gpu_memory_mib
from .worker import STEPS
from .report import CONTROL, write_report

ROOT = Path(__file__).resolve().parents[2]
GPU = 'GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e'


def plan_jobs(output, source_commit):
    jobs = []
    for step in STEPS:
        path = Path(output)/'runs'/('step_'+str(step).replace('.', 'p'))
        command = [common.PYTHON, '-u', '-m', 'experiments.osram_mosei_write_step_20261024.worker',
            '--write-step', str(step), '--seed', '66', '--phase', 'train', '--gpu-uuid', GPU,
            '--source-commit', source_commit, '--output', str(path)]
        jobs.append(dict(step=step, dataset='CMUMOSEI', method='none', seed=66, fold=1,
            epochs=60, phase='train', source_commit=source_commit, gpu_uuid=GPU,
            output=str(path), log=str(path)+'.log', status='pending', command=command))
    return jobs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    manifest = common.verify_source_snapshot(ROOT, args.source_commit)
    if common.base.devices([GPU])[GPU]['index'] != 7:
        raise ValueError('Physical GPU7 mapping changed; do not use GPU4')
    if args.output.exists() and not args.resume:
        raise FileExistsError('Fresh queue required; no duplicate submission')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    lock = (args.output/'QUEUE.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    jobs = plan_jobs(args.output, args.source_commit)
    if args.resume:
        state = json.loads((args.output/'QUEUE.json').read_text())
        if state['source_commit'] != args.source_commit or state['manifest_sha256'] != manifest['manifest_sha256']:
            raise ValueError('Resume identity differs')
        common.recover_jobs(state)
        state.update(pid=os.getpid(), status='resuming')
    else:
        state = dict(status='pending', pid=os.getpid(), started_at=time.time(), jobs=jobs,
            source_commit=args.source_commit, manifest_sha256=manifest['manifest_sha256'],
            gpu_uuid=GPU, maximum_parallel=4, reserve_mib=6144,
            reused_control=str(CONTROL),
            label='INTERNAL TEST-ORACLE SCREEN')
    common.base.ROOT = ROOT
    children = {}

    def record():
        state['updated_at'] = time.time(); common.write(args.output/'QUEUE.json', state)

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
                    raise RuntimeError('Failed arm; no silent retries')
            elif not children and not pending:
                write_report(args.output, state)
                state.update(status='complete', finished_at=time.time()); record(); return
            else:
                dev = common.base.devices([GPU])[GPU]
                resources = dict(total_mib=dev['total_mib'], used_mib=dev['used_mib'],
                    own_used_mib=own_gpu_memory_mib(GPU, [p.pid for p in children.values()]),
                    active_count=len(children), ram_gib=common.available_gib(),
                    disk_gib=shutil.disk_usage(args.output).free/2**30)
                state['resources'] = resources
                if pending and can_admit(**resources):
                    if common.verify_source_snapshot(ROOT, args.source_commit) != manifest:
                        raise RuntimeError('Immutable source changed')
                    i = pending[0]
                    children[i] = common.base.launch(state['jobs'][i])
                    state['jobs'][i]['started_at'] = time.time()
                    state['status'] = 'running'; record(); time.sleep(4); continue
                state['status'] = 'running' if children else 'waiting_for_resources'
            record(); time.sleep(15)
    except BaseException as error:
        state.update(status='failure_draining_active', error=repr(error)); record()
        for i, proc in list(children.items()):
            proc.wait(); common.poll(proc, state['jobs'][i]); record()
        state.update(status='failed', finished_at=time.time()); record(); raise


if __name__ == '__main__':
    main()
