"""Finite two-dataset smoke -> four-job train -> paired report, no HPO."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.osram_tasr_20261018.runner_helpers import write_json

BAD_UUID = 'GPU-46fb379f-cc90-dc82-9b5e-5d011f552264'
MODULE = 'experiments.osram_text_path_router_joint_20261019.run'


def devices(uuids):
    if len(set(uuids)) != 2 or BAD_UUID in uuids:
        raise ValueError('Require two distinct healthy GPU UUIDs; biggpu GPU4 forbidden')
    text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.total',
                                    '--format=csv,noheader,nounits'], text=True)
    found = {}
    for row in text.strip().splitlines():
        index, uuid, used, total = [value.strip() for value in row.split(',')]
        if uuid in uuids:
            if int(index) == 4:
                raise ValueError('Physical GPU4 forbidden')
            found[uuid] = dict(index=int(index), used_mib=int(used), total_mib=int(total))
    if set(found) != set(uuids):
        raise ValueError('Requested GPU UUID unavailable')
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', nargs=2, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new independent orchestration directory')
    args.output.mkdir(parents=True)
    state = dict(status='preflight', source_commit=args.source_commit, pid=os.getpid(), jobs=[],
                 gpu_preflight=devices(args.gpus), scope='2 joint smoke + 4 finite training jobs')
    datasets = ('MELD', 'IEMOCAPSix')
    children = []

    def record():
        write_json(args.output / 'QUEUE.json', state)

    def start(dataset, mode, phase, uuid):
        output = args.output / ('smoke' if phase == 'smoke' else 'runs') / dataset / mode
        output.parent.mkdir(parents=True, exist_ok=True)
        log = output.parent / (mode + '.log')
        command = [sys.executable, '-m', MODULE, '--phase', phase, '--mode', mode, '--dataset', dataset,
                   '--gpu-uuid', uuid, '--source-commit', args.source_commit, '--output', str(output)]
        with log.open('x') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                       env=dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                                                OPENBLAS_NUM_THREADS='2', CUDA_VISIBLE_DEVICES=uuid))
        job = dict(dataset=dataset, mode=mode, phase=phase, gpu_uuid=uuid, pid=process.pid,
                   command=command, output=str(output), log=str(log), status='running')
        state['jobs'].append(job)
        children.append((process, job))
        record()
        return process, job

    def finish(process, job):
        # Parent runs under tmux, not in a tool's blocking wait. Persist child exit.
        code = process.wait()
        job['exit_code'] = code
        path = Path(job['output']) / 'SUMMARY.json'
        summary = json.loads(path.read_text()) if path.exists() else {}
        job['status'] = 'complete' if code == 0 and summary.get('status') == 'complete' else 'failed'
        record()
        if job['status'] != 'complete':
            raise RuntimeError('Child failed, inspect ' + job['log'])
        return summary

    try:
        state['status'] = 'smoke'
        record()
        smoke_jobs = [start(dataset, 'joint', 'smoke', uuid) for dataset, uuid in zip(datasets, args.gpus)]
        smoke = [finish(*job) for job in smoke_jobs]
        current = devices(args.gpus)
        parallel = {}
        for dataset, uuid, summary in zip(datasets, args.gpus, smoke):
            peak = summary['peak_memory_mib']
            available = current[uuid]['total_mib'] - current[uuid]['used_mib']
            parallel[dataset] = 2 if 2 * peak * 1.5 + 2048 < available else 1
            if peak * 1.5 + 2048 >= available:
                raise RuntimeError('Insufficient free VRAM after smoke: ' + uuid)
        state.update(status='training', concurrency_per_dataset=parallel)
        record()
        # Both datasets start immediately. At most two of our jobs per GPU.
        first = [start(dataset, 'flat', 'train', uuid) for dataset, uuid in zip(datasets, args.gpus)]
        second = {dataset: start(dataset, 'joint', 'train', uuid)
                  for dataset, uuid in zip(datasets, args.gpus) if parallel[dataset] == 2}
        for dataset, uuid, job in zip(datasets, args.gpus, first):
            finish(*job)
            if dataset not in second:
                second[dataset] = start(dataset, 'joint', 'train', uuid)
        for dataset in datasets:
            finish(*second[dataset])
        from .report import aggregate
        aggregate(args.output / 'runs', args.output / 'report')
        state.update(status='complete', completed_at=time.time())
    except BaseException as error:
        # Never kill someone else's tasks or silently resubmit failed children.
        state.update(status='failed', error=repr(error))
        raise
    finally:
        record()


if __name__ == '__main__':
    main()
