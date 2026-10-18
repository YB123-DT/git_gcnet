"""Run exactly four smoke-gated TASR adapter screens in a persistent queue."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_jobs(output):
    from .protocol import build_cases, TRAINING_PROTOCOL
    cases = build_cases()
    return [dict(id=f'{dataset}_{mode}', method='tasr_frozen_adapter',
                 output=str(Path(output)/f'{dataset}_{mode}'),
                 config=dict(dataset=dataset, mode=mode, training=TRAINING_PROTOCOL,
                             anchor_sha256=case['checkpoint_sha256']))
            for dataset, case in cases.items() for mode in ('neutral', 'real')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    from experiments.osram_meld_block2_auto_20261014.scheduler import Scheduler, write_json
    from .runner_helpers import sha
    # No dataset/model imports in the launcher; UUID isolation belongs to child.
    core_files = ('gcnet_missing_m3/osram.py', 'gcnet_missing_m3/tasr.py')
    for dataset in ('MELD', 'IEMOCAPSix'):
        smoke = json.loads((Path(__file__).with_name(f'SMOKE_{dataset}.json')).read_text())
        if (smoke['status'] != 'complete' or not smoke['original_backbone_unchanged']
                or smoke['real_and_neutral_zero_logit_max_abs_error'] != 0
                or any(smoke['source_files'][name] != sha(ROOT/name) for name in core_files)):
            raise RuntimeError('Real mathematical/gradient smoke gate changed or incomplete')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.output.parent).free < 5*2**30:
        raise RuntimeError('Need5GiB headroom; no existing data/checkpoints are deleted')
    args.output.mkdir(exist_ok=True)
    jobs = build_jobs(args.output)
    plan = dict(source_commit=args.source_commit, jobs=jobs, gpus={
        '0': 'GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45'}, max_concurrent=4,
        no_expansion=True, selection='single adapter checkpoint on validation only')
    manifest = args.output/'EXPERIMENT.json'
    if manifest.exists() and json.loads(manifest.read_text()) != plan:
        raise ValueError('Immutable finite experiment configuration changed')
    write_json(manifest, plan)

    class FiniteScheduler(Scheduler):
        def command(self, job, config_path, gpu_uuid):
            cfg = job['config']
            command = [self.python, '-u', '-m', 'experiments.osram_tasr_20261018.run',
                '--phase', 'train', '--dataset', cfg['dataset'], '--mode', cfg['mode'],
                '--output', job['output'], '--gpu-uuid', gpu_uuid]
            if (Path(job['output'])/'last.pt').exists():
                command.append('--resume')
            return command

    # One idle healthy GPU suffices: measured frozen peaks .31/1.02GiB.
    # Reserve3GiB/job, still below32GiB at four jobs; never touch busy GPU5.
    scheduler = FiniteScheduler(args.output/'queue', gpus=(0,), per_gpu=4,
        code_root=ROOT, smoke_costs={'tasr_frozen_adapter': dict(
            peak_memory_mib=3072, disk_bytes=256*2**20)})
    if scheduler.gpu_snapshot()[0]['uuid'] != plan['gpus']['0']:
        raise RuntimeError('Physical GPU0 UUID changed')
    records = scheduler.run_jobs(jobs)
    write_json(args.output/'QUEUE_FINAL.json', records)
    if any(row['status'] != 'complete' for row in records):
        raise RuntimeError('Finite screening has failed/incomplete jobs; no expansion allowed')
    # Training sources stay immutable. A separately versioned, read-only
    # summarizer consumes these completed artifacts; it cannot select weights.


if __name__ == '__main__':
    main()
