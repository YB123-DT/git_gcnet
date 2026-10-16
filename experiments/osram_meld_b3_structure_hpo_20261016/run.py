"""Five original-Flat B3 structures, then finite, feedback-driven structural TPE."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

from experiments.osram_meld_block2_auto_20261014.methods import build_config
from experiments.osram_meld_block2_auto_20261014.scheduler import write_json
from experiments.osram_meld_b6_structure_hpo_20261015.run import (
    StructureScheduler, run_search as _run_search,
)

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'experiments/osram_meld_block2_auto_20261014/BASE_CONFIG.json'
FLAT = Path('/data2/yb/remote_experiments/osram_meld_block2_auto_20261014_v2/screen/Flat/SUMMARY.json')
DEFAULT_OUTPUT = Path('/home/qinjinghui/remote_experiments/osram_meld_b3_structure_hpo_20261016')
GPU_UUID = 'GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4'
STUDY_NAME = 'meld_B3_original_flat_structure_seed66'
INITIAL_CONFIGS = (
    {'bottlenecks': 4, 'rounds': 2}, {'bottlenecks': 2, 'rounds': 2},
    {'bottlenecks': 8, 'rounds': 2}, {'bottlenecks': 4, 'rounds': 1},
    {'bottlenecks': 4, 'rounds': 3},
)
SEARCH_SPACE = {
    'bottlenecks': {'kind': 'int', 'low': 1, 'high': 8},
    'rounds': {'kind': 'int', 'low': 1, 'high': 4},
}
METADATA = {
    'scope': 'B3 structural HPO on original Flat1600; original training hyperparameters fixed',
    'title': 'MELD B3: five specified structures followed by fifteen adaptive trials',
    'control_note': 'Trial0 freshly repeats B3 defaults (4 tokens, 2 rounds). Historical B3 scores are not new runs.',
    'fixed': {'dimension': 128, 'learning_rate': .001, 'weight_decay': 1e-5, 'dropout': .5,
              'epochs': 50, 'seed': 66, 'batch_size': 32, 'original_local_skip': True},
    'selection': 'INTERNAL TEST-ORACLE SCREEN; paired masks rank trials; Test-best results separate',
}


def make_config(bottlenecks, rounds):
    if (type(bottlenecks) is not int or not 1 <= bottlenecks <= 8
            or type(rounds) is not int or not 1 <= rounds <= 4):
        raise ValueError('B3 structural space: integer bottlenecks1..8, rounds1..4')
    base = json.loads(BASE.read_text())
    if (base['learning_rate'], base['weight_decay'], base['dropout']) != (.001, 1e-5, .5):
        raise ValueError('Original Flat training hyperparameters changed')
    config = build_config(base, 'B3', 'managed-by-queue')
    config.update(osram_mbt_bottlenecks=bottlenecks, osram_mbt_rounds=rounds)
    return config


def unique_tpe_sampler():
    import optuna

    class UniqueTPE(optuna.samplers.TPESampler):
        def sample_independent(self, study, trial, param_name, distribution):
            used = {(t.params.get('bottlenecks'), t.params.get('rounds'))
                    for t in study.get_trials(deepcopy=False)
                    if t.number != trial.number and {'bottlenecks', 'rounds'} <= set(t.params)}
            if param_name == 'bottlenecks':
                allowed = [n for n in range(1, 9) if any((n, r) not in used for r in range(1, 5))]
            elif param_name == 'rounds':
                allowed = [r for r in range(1, 5) if (trial.params['bottlenecks'], r) not in used]
            else:
                raise ValueError('Only the two registered B3 structure parameters may be searched')
            if not allowed:
                raise RuntimeError('Unique B3 structural search space exhausted')
            for _ in range(32):
                value = super().sample_independent(study, trial, param_name, distribution)
                if value in allowed:
                    return value
            study._storage.set_trial_user_attr(trial._trial_id, 'unique_fallback_' + param_name, True)
            return random.Random(66 + trial.number).choice(allowed)

    return UniqueTPE(seed=66, n_startup_trials=5, constant_liar=True, multivariate=False)


def run_search(output, flat_paired, scheduler):
    return _run_search(output, flat_paired, scheduler, budget=20, max_pending=5,
        config_factory=make_config, initial_configs=INITIAL_CONFIGS, search_space=SEARCH_SPACE,
        study_name=STUDY_NAME, sampler=unique_tpe_sampler(), metadata=METADATA,
        job_method='B3', job_prefix='b3_struct')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--gpus', default='2')
    parser.add_argument('--per-gpu', type=int, default=5)
    parser.add_argument('--trials', type=int, default=20)
    args = parser.parse_args(argv)
    try:
        args.gpus = tuple(int(value) for value in args.gpus.split(','))
    except ValueError:
        parser.error('GPU identifiers must be integers')
    if args.gpus != (2,) or args.per_gpu != 5 or args.trials != 20:
        parser.error('Authorized scope is physical GPU2 only, at most five parallel jobs, twenty total trials')
    return args


def main(argv=None):
    args = parse_args(argv)
    import numpy as np
    import optuna
    import torch
    from experiments.osram_meld_block2_auto_20261014.worker import (
        COMMON, DATASET, DATASET_SHA, source_hashes,
    )
    from gcnet_missing_m3.training_resume import _sha

    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if subprocess.check_output(['git', 'diff', 'HEAD', '--name-only'], cwd=ROOT, text=True).strip():
        raise RuntimeError('Require committed immutable training source')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if _sha(DATASET) != DATASET_SHA or shutil.disk_usage(args.output.parent).free < 30 * 2**30:
        raise RuntimeError('MELD dataset SHA or minimum 30GiB disk preflight failed')
    flat = json.loads(FLAT.read_text())
    if flat.get('status') != 'complete' or flat.get('method') != 'Flat':
        raise ValueError('Accepted original-hyperparameter common-mask Flat result missing')
    flat_config = flat['config']
    expected = make_config(4, 2)
    for name in ('learning_rate', 'weight_decay', 'dropout', 'missing_mask_protocol', 'batch_size',
                 'epochs', 'seed', 'osram_output_dim', 'osram_readout_fusion', 'osram_write_step'):
        if flat_config.get(name) != expected[name]:
            raise ValueError('Paired original Flat reference protocol mismatch: ' + name)
    if flat_config.get('osram_disable_local_skip', False):
        raise ValueError('Reference must preserve original Local Skip')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.controller.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        scheduler = StructureScheduler(args.output / 'queue', gpus=args.gpus,
                                       per_gpu=5, per_gpu_limit=5)
        scheduler.global_per_gpu_limit = 5
        snapshot = scheduler.gpu_snapshot()
        if snapshot[2]['uuid'] != GPU_UUID:
            raise RuntimeError('Physical GPU2 UUID changed; no task launched')
        common_hashes = {str(i / 10): _sha(COMMON / f'predictions_miss_0p{i}.npz') for i in range(8)}
        inventory = json.loads((ROOT / 'experiments/osram_meld_query_decision_20261014/CANONICAL_INPUTS.json').read_text())
        if common_hashes != inventory['mask_sha256']:
            raise ValueError('Protected common masks changed')
        provenance = dict(source_git_sha=sha, code_hashes=source_hashes(),
            launcher_sha256=_sha(Path(__file__)), shared_controller_sha256=_sha(
                ROOT / 'experiments/osram_meld_b6_structure_hpo_20261015/run.py'),
            base_config_sha256=_sha(BASE), dataset_sha256=DATASET_SHA,
            flat_reference_path=str(FLAT), flat_reference_sha256=_sha(FLAT),
            common_mask_hashes=common_hashes, paired_mask_digest=flat['paired']['mask_digest'],
            config=expected, gpu_whitelist=[2], gpu_uuid=GPU_UUID,
            per_gpu_task_limit=5, per_gpu_total_limit=5, global_limit=5,
            initial_configs=list(INITIAL_CONFIGS), trials=20, search_space=SEARCH_SPACE,
            sampler='independent TPE; startup5 barrier, constant_liar=True; unique token/round pairs',
            environment=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                             optuna=optuna.__version__), metadata=METADATA)
        previous = args.output / 'PREFLIGHT.json'
        if previous.exists() and json.loads(previous.read_text()) != provenance:
            raise ValueError('Do not mix source/config/data versions in this output directory')
        write_json(previous, provenance)
        runner = dict(status='running', pid=os.getpid(), source_git_sha=sha, started_unix=time.time())
        write_json(args.output / 'RUNNER.json', runner)
        try:
            smoke = args.output / 'smoke'
            smoke_config = args.output / 'smoke_config.json'
            worst = make_config(8, 4)
            write_json(smoke_config, worst)
            if not (smoke / 'SUMMARY.json').exists():
                if snapshot[2]['free_mib'] < 6000:
                    raise RuntimeError('Insufficient free GPU2 memory for largest-structure real-batch smoke')
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU_UUID, OMP_NUM_THREADS='2',
                           MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
                command = [sys.executable, '-u', '-m',
                    'experiments.osram_meld_block2_auto_20261014.worker', '--config', str(smoke_config),
                    '--output', str(smoke), '--gpu-uuid', GPU_UUID, '--smoke']
                with (args.output / 'smoke.log').open('ab') as stream:
                    subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, check=True)
            smoke_result = json.loads((smoke / 'SUMMARY.json').read_text())
            if (smoke_result.get('status') != 'complete' or not smoke_result.get('smoke_only')
                    or any(smoke_result['config'].get(k) != v for k, v in worst.items())
                    or smoke_result['provenance']['git_sha'] != sha):
                raise RuntimeError('Largest-structure smoke incomplete or from a different source/config')
            scheduler.smoke_costs = {'B3': {'peak_memory_mib': smoke_result['peak_memory_mib'],
                                           'disk_bytes': 2 * 2**30}}
            report = run_search(args.output, flat['paired'], scheduler)
            runner.update(status=report['status'], finished_unix=time.time())
        except BaseException as error:
            runner.update(status='failed', error=repr(error), finished_unix=time.time())
            raise
        finally:
            write_json(args.output / 'RUNNER.json', runner)


if __name__ == '__main__':
    main()
