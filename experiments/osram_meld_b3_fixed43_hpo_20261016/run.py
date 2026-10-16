"""Four finite TPE trials with exactly four bottlenecks and three exchanges."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiments.osram_meld_b3_capacity_hpo_20261016 import run as capacity
from experiments.osram_meld_b3_gpu3_extension_20261016.run import GPU_UUID, digest
from experiments.osram_meld_block2_auto_20261014.optuna_search import paired_objective
from experiments.osram_meld_block2_auto_20261014.scheduler import write_json
from experiments.osram_meld_b6_structure_hpo_20261015.run import StructureScheduler

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path('/home/qinjinghui/remote_experiments/osram_meld_b3_fixed43_hpo_20261016')
WORKER_ROOT = Path('/home/qinjinghui/remote_experiments/osram_meld_b3_capacity_code_6ce8ca6')
WORKER_SHA = '6ce8ca60d507fe81dd37d9ce74623104ee2e7098'
HISTORICAL_WORKER_SHA = capacity.HISTORICAL_WORKER_SHA
STUDY_NAME = 'meld_B3_fixed_four_tokens_three_rounds_seed66'
FIXED_STRUCTURE = (4, 3)
SOURCES = (('original', capacity.ORIGINAL, 20), ('extension', capacity.EXTENSION, 4),
           ('capacity', capacity.DEFAULT_OUTPUT, 4))
CHECK_KEYS = ('learning_rate', 'weight_decay', 'dropout', 'seed', 'epochs', 'batch_size',
              'missing_mask_protocol', 'osram_meaningful_block', 'osram_output_dim',
              'osram_readout_fusion', 'osram_write_step')


def freeze_reference(output, sources, flat_paired):
    """Reserve all 28 runs, but import only completed 4/3 observations into TPE."""
    path = output / 'REFERENCE.json'
    if path.exists():
        reference = json.loads(path.read_text())
        if reference['flat_paired'] != flat_paired:
            raise ValueError('Frozen common-mask Flat reference changed')
        return reference
    reserved, completed, snapshots = [], [], []
    for label, root, count in sources:
        queue = root / 'queue' / 'QUEUE.json'
        data = queue.read_bytes()
        jobs = list(json.loads(data)['jobs'].values())
        if len(jobs) != count:
            raise ValueError('Require every previous configuration: ' + label)
        snapshots.append(dict(path=str(queue), sha256=hashlib.sha256(data).hexdigest()))
        expected_sha = WORKER_SHA if label == 'capacity' else HISTORICAL_WORKER_SHA
        for job in jobs:
            config = job['config']
            params = {key: config.get(field, capacity.DEFAULT_KNOBS.get(key))
                      for key, field in capacity.FIELDS.items()}
            expected = capacity.make_config(**params)
            if any(config[key] != expected[key] for key in CHECK_KEYS):
                raise ValueError('Historical training protocol changed: ' + job['id'])
            if config.get('osram_disable_local_skip', False):
                raise ValueError('Original Local Skip must remain enabled')
            reserved.append(params)
            if (params['bottlenecks'], params['rounds']) != FIXED_STRUCTURE:
                continue
            summary_path = Path(job['output']) / 'SUMMARY.json'
            data = summary_path.read_bytes()
            summary = json.loads(data)
            if summary.get('status') != 'complete' or summary['provenance']['git_sha'] != expected_sha:
                raise ValueError('Matching observation incomplete or numerical source changed')
            completed.append(dict(key=label + ':' + job['id'], params=params,
                value=paired_objective(summary, flat_paired), source_git_sha=expected_sha,
                summary_path=str(summary_path), summary_sha256=hashlib.sha256(data).hexdigest()))
    if len({capacity.signature(p) for p in reserved}) != 28 or len(completed) < 2:
        raise ValueError('Require 28 distinct reserved tuples and two completed 4/3 observations')
    reference = dict(reserved=reserved, completed=completed, snapshots=snapshots,
                     flat_paired=flat_paired, frozen_unix=time.time())
    write_json(path, reference)
    return reference


def validate_smoke(smoke, smoke_path):
    """Reuse the prior largest real-batch smoke, not a new GPU workload."""
    worst = capacity.make_config(8, 4, 256, 8, 512, 1.)
    if (smoke.get('status') != 'complete' or not smoke.get('smoke_only')
            or smoke['provenance']['git_sha'] != WORKER_SHA
            or any(smoke['config'].get(key) != value for key, value in worst.items())):
        raise RuntimeError('Largest-capacity smoke is not the known worker/config')
    peak = smoke['peak_memory_mib']
    if not 0 < peak <= 4000:
        raise RuntimeError('Unexpected prior smoke memory; do not assume four slots fit')
    return dict(path=str(smoke_path), sha256=digest(smoke_path), peak_memory_mib=peak)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(exist_ok=True)
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    worker_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WORKER_ROOT, text=True).strip()
    for root in (ROOT, WORKER_ROOT):
        if subprocess.check_output(['git', 'diff', 'HEAD', '--name-only'], cwd=root, text=True).strip():
            raise RuntimeError('Both controller and numerical worker must be immutable')
    if worker_sha != WORKER_SHA:
        raise RuntimeError('Numerical worker SHA changed')
    with (args.output / '.controller.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runner = dict(status='waiting_for_capacity_four', pid=os.getpid(), source_git_sha=sha,
                      worker_git_sha=WORKER_SHA, worker_root=str(WORKER_ROOT), started_unix=time.time())
        write_json(args.output / 'RUNNER.json', runner)
        try:
            while True:
                path = capacity.DEFAULT_OUTPUT / 'SUMMARY.json'
                if path.exists() and json.loads(path.read_text()).get('status') in (
                        'complete', 'completed_with_failures'):
                    break
                previous_runner = capacity.DEFAULT_OUTPUT / 'RUNNER.json'
                if previous_runner.exists() and json.loads(previous_runner.read_text()).get('status') == 'failed':
                    raise RuntimeError('Prior capacity controller failed; do not bypass its workload')
                time.sleep(30)
            from experiments.osram_meld_block2_auto_20261014.worker import COMMON, DATASET, DATASET_SHA
            import numpy as np
            import optuna
            import torch
            if digest(DATASET) != DATASET_SHA or shutil.disk_usage(args.output.parent).free < 30 * 2**30:
                raise RuntimeError('Dataset hash or minimum 30GiB disk check failed')
            flat = json.loads(capacity.FLAT.read_text())
            baseline = capacity.make_config(4, 3)
            if flat.get('status') != 'complete' or flat.get('method') != 'Flat':
                raise RuntimeError('Original Flat reference invalid')
            if any(flat['config'].get(k) != baseline[k] for k in CHECK_KEYS if k != 'osram_meaningful_block'):
                raise RuntimeError('Original Flat global settings changed')
            if flat['config'].get('osram_disable_local_skip', False):
                raise RuntimeError('Original Flat Local Skip disabled')
            inventory = json.loads((ROOT / 'experiments/osram_meld_query_decision_20261014/CANONICAL_INPUTS.json').read_text())
            mask_hashes = {str(i / 10): digest(COMMON / f'predictions_miss_0p{i}.npz') for i in range(8)}
            if mask_hashes != inventory['mask_sha256']:
                raise RuntimeError('Protected common masks changed')
            smoke_path = capacity.DEFAULT_OUTPUT / 'smoke' / 'SUMMARY.json'
            smoke = validate_smoke(json.loads(smoke_path.read_text()), smoke_path)
            reference = freeze_reference(args.output, SOURCES, flat['paired'])
            provenance = dict(controller_git_sha=sha, numerical_git_sha=WORKER_SHA,
                worker_root=str(WORKER_ROOT), launcher_sha256=digest(Path(__file__)),
                reference_sha256=digest(args.output / 'REFERENCE.json'), dataset_sha256=DATASET_SHA,
                common_mask_hashes=mask_hashes, paired_mask_digest=flat['paired']['mask_digest'],
                reused_smoke=smoke, fixed_structure=FIXED_STRUCTURE,
                gpu=3, gpu_uuid=GPU_UUID, per_gpu_total_limit=4,
                new_training_budget=4, previous_budget=28, combined_budget=32,
                environment=dict(python=sys.version, torch=torch.__version__,
                                 numpy=np.__version__, optuna=optuna.__version__))
            provenance = json.loads(json.dumps(provenance))
            path = args.output / 'PREFLIGHT.json'
            if path.exists() and json.loads(path.read_text()) != provenance:
                raise RuntimeError('Frozen numerical source, reference, or resources changed')
            write_json(path, provenance)
            scheduler = StructureScheduler(args.output / 'queue', gpus=(3,), per_gpu=4,
                                            per_gpu_limit=4, code_root=WORKER_ROOT)
            scheduler.global_per_gpu_limit = 4
            scheduler.smoke_costs = {'B3': {'peak_memory_mib': smoke['peak_memory_mib'],
                                           'disk_bytes': 2 * 2**30}}
            if scheduler.gpu_snapshot()[3]['uuid'] != GPU_UUID:
                raise RuntimeError('Physical GPU3 UUID changed')
            runner['status'] = 'running'
            write_json(args.output / 'RUNNER.json', runner)
            result = capacity.run_search(args.output, reference, scheduler,
                fixed_structure=FIXED_STRUCTURE, study_name=STUDY_NAME,
                previous_budget=28, job_prefix='b3_fixed43', n_startup_trials=2)
            runner['status'] = result['status']
        except BaseException as error:
            runner.update(status='failed', error=repr(error))
            raise
        finally:
            runner['finished_unix'] = time.time()
            write_json(args.output / 'RUNNER.json', runner)


if __name__ == '__main__':
    main()
