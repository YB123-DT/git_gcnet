"""Add only seeds67/68 to the completed one-stage seed66 paired experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.osram_text_path_router_joint_20261019.orchestrate import devices

ROOT = Path(__file__).resolve().parents[2]
SEED66_ROOT = Path('/home/qinjinghui/remote_experiments/osram_text_path_router_joint_20261019/execution_v1/runs')
DATASETS = ('MELD', 'IEMOCAPSix')
MODULE = 'experiments.osram_text_path_router_joint_20261019.run'
# Observed allocated peaks from the completed, identical seed66 algorithm.
PEAK_GIB = {'MELD': {'flat': .664, 'joint': 1.43},
            'IEMOCAPSix': {'flat': 2.15, 'joint': 5.51}}


def plan_jobs(output, source_commit, gpus):
    jobs = []
    for dataset, uuid in zip(DATASETS, gpus):
        for seed in (67, 68):
            for mode in ('flat', 'joint'):
                path = Path(output) / 'runs' / dataset / ('seed_' + str(seed)) / mode
                command = [sys.executable, '-m', MODULE, '--phase', 'train', '--mode', mode,
                    '--dataset', dataset, '--seed', str(seed), '--gpu-uuid', uuid,
                    '--source-commit', source_commit, '--output', str(path)]
                jobs.append(dict(dataset=dataset, seed=seed, mode=mode, gpu_uuid=uuid,
                    output=str(path), log=str(path.with_suffix('.log')), command=command, status='pending'))
    return jobs


def memory_requirements(found, gpus):
    result = {}
    for dataset, uuid in zip(DATASETS, gpus):
        required = 2 * sum(PEAK_GIB[dataset].values()) * 1024 * 1.5 + 2048
        available = found[uuid]['total_mib'] - found[uuid]['used_mib']
        result[dataset] = dict(gpu_uuid=uuid, observed_peak_gib=PEAK_GIB[dataset],
            planned_processes=4, required_free_mib=required, available_mib=available)
        if available < required:
            raise RuntimeError('Insufficient free VRAM for four ' + dataset + ' jobs: ' + str(result[dataset]))
    return result


def verify_seed66(root=SEED66_ROOT):
    """Reuse completed seed66 results only if all unchanged core files match."""
    records = {}
    for dataset in DATASETS:
        records[dataset] = {}
        for mode in ('flat', 'joint'):
            path = Path(root) / dataset / mode / 'SUMMARY.json'
            summary = json.loads(path.read_text())
            if (summary['status'] != 'complete' or summary['seed'] != 66
                    or summary['mode'] != mode or summary['dataset'] != dataset):
                raise ValueError('Seed66 reference is incomplete or mismatched: ' + str(path))
            for source, expected in summary['identity']['source_hashes'].items():
                # Only seed/loader/provenance support changed in this wrapper.
                if source == 'experiments/osram_text_path_router_joint_20261019/run.py':
                    continue
                actual = hashlib.sha256((ROOT / source).read_bytes()).hexdigest()
                if actual != expected:
                    raise ValueError('Changed seed66 algorithm source: ' + source)
            records[dataset][mode] = dict(path=str(path),
                summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                source_commit=summary['source_commit'], config=summary['identity']['config'],
                validation_mask_hashes=summary['identity']['validation_mask_hashes'],
                test_masks=summary['identity']['test_masks'], split=summary['identity']['split'])
        flat, joint = records[dataset]['flat'], records[dataset]['joint']
        for key in ('config', 'validation_mask_hashes', 'test_masks', 'split'):
            if flat[key] != joint[key]:
                raise ValueError('Seed66 pair differs in ' + key)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--gpus', nargs=2, required=True, help='MELD GPU UUID, then IEMOCAP GPU UUID')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output directory; no duplicate submission or implicit resume')
    from experiments.osram_tasr_20261018.runner_helpers import write_json
    gpu_preflight = devices(args.gpus)
    resources = memory_requirements(gpu_preflight, args.gpus)
    references = verify_seed66()
    args.output.mkdir(parents=True)
    import fcntl
    lock = (args.output / 'QUEUE.lock').open('x')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = dict(status='preflight', source_commit=args.source_commit, pid=os.getpid(),
        gpu_preflight=gpu_preflight, memory_preflight=resources, seed66_references=references,
        seed66_root=str(SEED66_ROOT), jobs=plan_jobs(args.output, args.source_commit, args.gpus),
        scope='Exactly 8 new jobs: seeds67/68 x Flat/Joint x MELD/IEMOCAP6-fold5',
        seed_protocol='Training seed varies; holdout split and evaluation mask banks remain seed66')
    children = []

    def record():
        write_json(args.output / 'QUEUE.json', state)

    try:
        state.update(status='training', started_at=time.time())
        record()
        for job in state['jobs']:
            output = Path(job['output'])
            output.parent.mkdir(parents=True, exist_ok=True)
            with Path(job['log']).open('x') as stream:
                process = subprocess.Popen(job['command'], stdout=stream, stderr=subprocess.STDOUT,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=job['gpu_uuid'],
                        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'))
            job.update(pid=process.pid, status='running', started_at=time.time())
            children.append((process, job))
            record()
        # Parent runs persistently under tmux. Never kill peers on child failure.
        while any(job['status'] == 'running' for _, job in children):
            for process, job in children:
                code = process.poll()
                if job['status'] != 'running' or code is None:
                    continue
                summary_path = Path(job['output']) / 'SUMMARY.json'
                try:
                    summary = json.loads(summary_path.read_text())
                except (OSError, json.JSONDecodeError):
                    summary = {}
                job.update(exit_code=code, finished_at=time.time(),
                    status='complete' if code == 0 and summary.get('status') == 'complete' else 'failed')
                record()
            if any(job['status'] == 'running' for _, job in children):
                time.sleep(5)
        if any(job['status'] != 'complete' for _, job in children):
            raise RuntimeError('One or more finite jobs failed; no automatic retries; inspect individual logs')
        from .report import write_report
        write_report(args.output, state)
        state.update(status='complete', completed_at=time.time())
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        record()


if __name__ == '__main__':
    main()
