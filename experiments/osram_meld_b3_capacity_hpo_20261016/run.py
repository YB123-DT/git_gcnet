"""Four capacity/optimizer B3 TPE trials on original Flat; no token-only proposals."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

from experiments.osram_meld_b3_structure_hpo_20261016.run import (
    BASE, FLAT, DEFAULT_OUTPUT as ORIGINAL, make_config as structure_config,
)
from experiments.osram_meld_b3_gpu3_extension_20261016.run import (
    DEFAULT_OUTPUT as EXTENSION, GPU_UUID, WORKER_SHA as HISTORICAL_WORKER_SHA, digest,
)
from experiments.osram_meld_b6_structure_hpo_20261015.run import StructureScheduler, export
from experiments.osram_meld_block2_auto_20261014.scheduler import write_json
from experiments.osram_meld_block2_auto_20261014.optuna_search import paired_objective

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path('/home/qinjinghui/remote_experiments/osram_meld_b3_capacity_hpo_20261016')
STUDY_NAME = 'meld_B3_four_capacity_optimizer_trials_seed66'
DEFAULT_KNOBS = dict(dim=128, heads=4, ffn_hidden=256, lr_multiplier=1.)
AXES = dict(bottlenecks=tuple(range(1, 9)), rounds=tuple(range(1, 5)),
            dim=(64, 128, 256), heads=(2, 4, 8), ffn_hidden=(128, 256, 512),
            lr_multiplier=(.1, .3, 1.))
REQUIRED_KNOBS = tuple(DEFAULT_KNOBS)
FIELDS = dict(bottlenecks='osram_mbt_bottlenecks', rounds='osram_mbt_rounds',
              dim='osram_mbt_dim', heads='osram_mbt_heads',
              ffn_hidden='osram_mbt_ffn_hidden', lr_multiplier='osram_mbt_lr_multiplier')


def make_config(bottlenecks, rounds, dim=128, heads=4, ffn_hidden=256, lr_multiplier=1.):
    values = dict(bottlenecks=bottlenecks, rounds=rounds, dim=dim, heads=heads,
                  ffn_hidden=ffn_hidden, lr_multiplier=lr_multiplier)
    for key, value in values.items():
        if value not in AXES[key] or isinstance(value, bool) or (
                key != 'lr_multiplier' and type(value) is not int):
            raise ValueError('Value outside preregistered B3 space: ' + key)
    config = structure_config(bottlenecks, rounds)
    config.update({FIELDS[key]: value for key, value in values.items()})
    return config


def signature(params):
    return tuple(params[key] for key in AXES)


def is_new_knob_candidate(params):
    return any(params[key] != value for key, value in DEFAULT_KNOBS.items())


def feasible_candidates(reserved, prefix=None, required=None, fixed_structure=None):
    excluded = {signature(row) for row in reserved}
    prefix = prefix or {}
    if fixed_structure is not None:
        fixed = dict(zip(('bottlenecks', 'rounds'), fixed_structure))
        if any(key in prefix and prefix[key] != value for key, value in fixed.items()):
            return []
        prefix = dict(prefix, **fixed)
    return [dict(zip(AXES, values)) for values in itertools.product(*AXES.values())
            if values not in excluded
            and all(values[list(AXES).index(key)] == value for key, value in prefix.items())
            and any(values[list(AXES).index(key)] != value for key, value in DEFAULT_KNOBS.items())
            and (required is None or values[list(AXES).index(required)] != DEFAULT_KNOBS[required])]


def unique_sampler(reserved, fixed_structure=None, n_startup_trials=5):
    import optuna

    class CapacityTPE(optuna.samplers.TPESampler):
        def sample_independent(self, study, trial, param_name, distribution):
            used = reserved + [t.params for t in study.get_trials(deepcopy=False)
                if t.number != trial.number and set(AXES) <= set(t.params)]
            remaining = feasible_candidates(used, trial.params,
                                            trial.user_attrs.get('required_nondefault_knob'),
                                            fixed_structure=fixed_structure)
            allowed = {row[param_name] for row in remaining}
            if not allowed:
                raise RuntimeError('No unique non-token-only completion remains')
            for _ in range(64):
                candidate = super().sample_independent(study, trial, param_name, distribution)
                if candidate in allowed:
                    return candidate
            study._storage.set_trial_user_attr(trial._trial_id,
                                               'conditional_fallback_' + param_name, True)
            return random.Random(266 + trial.number).choice(sorted(allowed))

    return CapacityTPE(seed=266, n_startup_trials=n_startup_trials,
                       constant_liar=True, multivariate=False)


def freeze_reference(output, original, extension, flat_paired):
    """Snapshot read-only completed history; all 24 previous full tuples are reserved."""
    path = output / 'REFERENCE.json'
    if path.exists():
        return json.loads(path.read_text())
    reserved, completed, snapshots = [], [], []
    for label, root, expected in [('original', original, 20), ('extension', extension, 4)]:
        queue_path = root / 'queue' / 'QUEUE.json'
        queue_bytes = queue_path.read_bytes()
        jobs = list(json.loads(queue_bytes)['jobs'].values())
        if len(jobs) != expected:
            raise ValueError('Require all existing queue configurations: ' + label)
        snapshots.append(dict(path=str(queue_path), sha256=hashlib.sha256(queue_bytes).hexdigest()))
        for job in jobs:
            config = job['config']
            params = {key: config.get(field, DEFAULT_KNOBS.get(key)) for key, field in FIELDS.items()}
            if any(params[key] != value for key, value in DEFAULT_KNOBS.items()):
                raise ValueError('Historical trials do not have actual default capacity knobs')
            expected_config = make_config(**params)
            for key in ('learning_rate', 'weight_decay', 'dropout', 'seed', 'epochs',
                        'missing_mask_protocol', 'osram_meaningful_block', 'osram_output_dim',
                        'batch_size', 'osram_readout_fusion', 'osram_write_step'):
                if config[key] != expected_config[key]:
                    raise ValueError('Historical protocol mismatch: ' + key)
            reserved.append(params)
            summary_path = Path(job['output']) / 'SUMMARY.json'
            if not summary_path.exists():
                continue
            summary_bytes = summary_path.read_bytes()
            summary = json.loads(summary_bytes)
            if summary.get('status') != 'complete':
                continue
            if summary['provenance']['git_sha'] != HISTORICAL_WORKER_SHA:
                raise ValueError('Historical numerical source is not the known default implementation')
            completed.append(dict(key=label + ':' + job['id'], params=params,
                value=paired_objective(summary, flat_paired), source_git_sha=HISTORICAL_WORKER_SHA,
                summary_path=str(summary_path), summary_sha256=hashlib.sha256(summary_bytes).hexdigest()))
    if len({signature(row) for row in reserved}) != 24 or len(completed) < 5:
        raise ValueError('Need 24 distinct reserved structures and at least five completed observations')
    reference = dict(reserved=reserved, completed=completed, snapshots=snapshots,
                     flat_paired=flat_paired, frozen_unix=time.time())
    write_json(path, reference)
    return reference


def run_search(output, reference, scheduler, *, fixed_structure=None, study_name=STUDY_NAME,
               previous_budget=24, job_prefix='b3_capacity', n_startup_trials=5):
    import optuna
    output.mkdir(parents=True, exist_ok=True)
    distributions = {key: (optuna.distributions.IntDistribution(1, 8) if key == 'bottlenecks'
        else optuna.distributions.IntDistribution(1, 4) if key == 'rounds'
        else optuna.distributions.CategoricalDistribution(values)) for key, values in AXES.items()}
    study = optuna.create_study(study_name=study_name, direction='maximize',
        storage='sqlite:///' + str(output / 'CAPACITY.sqlite3'), load_if_exists=True,
        sampler=unique_sampler(reference['reserved'], fixed_structure, n_startup_trials))
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    protocol = dict(budget=4, original_budget=previous_budget, combined_budget=previous_budget + 4, source_git_sha=sha,
        reserved=reference['reserved'], completed=reference['completed'],
        search_space=AXES, required_nondefault_knobs=REQUIRED_KNOBS,
        objective='0.6*paired_delta_ACC+0.4*paired_delta_UA',
        selection='INTERNAL TEST-ORACLE SCREEN', fixed_training=make_config(4, 2))
    if fixed_structure is not None:
        protocol.update(fixed_structure=fixed_structure, n_startup_trials=n_startup_trials)
        if any((r['params']['bottlenecks'], r['params']['rounds']) != tuple(fixed_structure)
               for r in reference['completed']):
            raise ValueError('Fixed-structure TPE must not import other structures as observations')
        protocol['fixed_training'] = make_config(*fixed_structure)
    # JSON normalization avoids tuple/list differences on SQLite resume.
    protocol = json.loads(json.dumps(protocol))
    if study.user_attrs.get('protocol', protocol) != protocol:
        raise ValueError('Frozen source, reference, budget, or search space changed')
    study.set_user_attr('protocol', protocol)
    for row in reference['completed']:
        if not any(t.user_attrs.get('reference_key') == row['key'] for t in study.trials):
            study.add_trial(optuna.trial.create_trial(params=row['params'], value=row['value'],
                distributions=distributions, user_attrs={'reference_only': True, 'reference_key': row['key']}))
    new = [t for t in study.trials if not t.user_attrs.get('reference_only')]
    if len(new) > 4:
        raise RuntimeError('Four-new-job budget exceeded')
    running = []
    for index in range(4):
        live = optuna.Trial(study, new[index]._trial_id) if index < len(new) else study.ask()
        if study._storage.get_trial(live._trial_id).state.is_finished():
            continue
        live.set_user_attr('job_id', f'{job_prefix}_{index + 1:02d}')
        live.set_user_attr('required_nondefault_knob', REQUIRED_KNOBS[index])
        live.suggest_int('bottlenecks', 1, 8)
        live.suggest_int('rounds', 1, 4)
        for key in DEFAULT_KNOBS:
            live.suggest_categorical(key, AXES[key])
        if fixed_structure is not None and (
                live.params['bottlenecks'], live.params['rounds']) != tuple(fixed_structure):
            raise RuntimeError('Fixed structure changed')
        if not is_new_knob_candidate(live.params) or (
                live.params[REQUIRED_KNOBS[index]] == DEFAULT_KNOBS[REQUIRED_KNOBS[index]]):
            raise RuntimeError('Illegal token-only proposal or required axis unchanged')
        running.append(live)
    jobs = [dict(id=t.user_attrs['job_id'], method='B3', config=make_config(**t.params),
                 output=str(output / 'runs' / t.user_attrs['job_id'])) for t in running]
    if jobs:
        write_json(output / 'PLANNED_JOBS.json', jobs)
    metadata = dict(title='MELD B3: four conditional capacity/optimizer TPE trials',
        scope='Conditional six-dimensional expansion, not controlled one-factor ablation',
        control_note='Historical 24 runs are reference-only; exactly four NEW jobs, each changes a required other knob.',
        new_training_budget=4, combined_budget=previous_budget + 4,
        reference_trials=len(reference['completed']))
    if fixed_structure is not None:
        metadata.update(title='MELD B3: fixed four tokens / three exchange rounds',
            scope='Fixed structure; only internal dimension, heads, FFN, and local learning rate change',
            control_note='Only completed 4/3 observations train TPE; all 28 prior tuples excluded. Exactly four new jobs.',
            fixed_structure=fixed_structure, n_startup_trials=n_startup_trials)

    def report():
        result = export(SimpleNamespace(trials=[t for t in study.trials
            if not t.user_attrs.get('reference_only')]), output, metadata)
        rows = result['trials']
        result['status'] = ('complete' if all(r['state'] == 'COMPLETE' for r in rows)
            else 'completed_with_failures') if len(rows) == 4 and all(
                r['state'] in ('COMPLETE', 'FAIL') for r in rows) else 'running'
        write_json(output / 'SUMMARY.json', result)
        return result

    report()
    if running:
        records = scheduler.run_jobs(jobs)
        for trial, record in zip(running, records):
            trial.set_user_attr('queue_record', {key: value for key, value in record.items() if key != 'summary'})
            try:
                if record['status'] != 'complete':
                    raise ValueError('Training/replay failed; inspect queue logs')
                value = paired_objective(record['summary'], reference['flat_paired'])
                trial.set_user_attr('summary', record['summary'])
            except (KeyError, ValueError, TypeError) as error:
                trial.set_user_attr('failure', str(error))
                study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
            else:
                study.tell(trial.number, value)
    return report()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    import numpy as np
    import optuna
    import torch
    from experiments.osram_meld_block2_auto_20261014.worker import COMMON, DATASET, DATASET_SHA, source_hashes
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if subprocess.check_output(['git', 'diff', 'HEAD', '--name-only'], cwd=ROOT, text=True).strip():
        raise RuntimeError('Require committed immutable source for BOTH worker and controller')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if digest(DATASET) != DATASET_SHA or shutil.disk_usage(args.output.parent).free < 30 * 2**30:
        raise RuntimeError('MELD dataset SHA or minimum 30GiB disk check failed')
    flat = json.loads(FLAT.read_text())
    if flat.get('status') != 'complete' or flat.get('method') != 'Flat':
        raise RuntimeError('Original Flat reference invalid')
    for key in ('learning_rate', 'weight_decay', 'dropout', 'epochs', 'seed', 'batch_size',
                'missing_mask_protocol', 'osram_output_dim', 'osram_readout_fusion', 'osram_write_step'):
        if flat['config'].get(key) != make_config(4, 2)[key]:
            raise RuntimeError('Original Flat protocol mismatch: ' + key)
    if flat['config'].get('osram_disable_local_skip', False):
        raise RuntimeError('Original Local Skip unexpectedly disabled')
    inventory = json.loads((ROOT / 'experiments/osram_meld_query_decision_20261014/CANONICAL_INPUTS.json').read_text())
    mask_hashes = {str(i / 10): digest(COMMON / f'predictions_miss_0p{i}.npz') for i in range(8)}
    if mask_hashes != inventory['mask_sha256']:
        raise RuntimeError('Protected common masks changed')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.controller.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runner = dict(status='waiting_for_previous_gpu3_jobs', pid=os.getpid(),
                      source_git_sha=sha, started_unix=time.time())
        write_json(args.output / 'RUNNER.json', runner)
        try:
            # Never signal old jobs; wait so smoke also respects the total four-process cap.
            while True:
                previous = EXTENSION / 'SUMMARY.json'
                if previous.exists() and json.loads(previous.read_text()).get('status') in (
                        'complete', 'completed_with_failures'):
                    break
                time.sleep(30)
            scheduler = StructureScheduler(args.output / 'queue', gpus=(3,), per_gpu=4,
                                            per_gpu_limit=4, code_root=ROOT)
            scheduler.global_per_gpu_limit = 4
            while True:
                snapshot = scheduler.gpu_snapshot()[3]
                if snapshot['uuid'] != GPU_UUID:
                    raise RuntimeError('Physical GPU3 UUID changed')
                if snapshot['free_mib'] >= 6000:
                    break
                time.sleep(30)
            reference = freeze_reference(args.output, ORIGINAL, EXTENSION, flat['paired'])
            worst = make_config(8, 4, 256, 8, 512, 1.)
            smoke_dir = args.output / 'smoke'
            smoke_config = args.output / 'smoke_config.json'
            write_json(smoke_config, worst)
            runner['status'] = 'largest_structure_smoke'
            write_json(args.output / 'RUNNER.json', runner)
            if not (smoke_dir / 'SUMMARY.json').exists():
                command = [sys.executable, '-u', '-m',
                    'experiments.osram_meld_block2_auto_20261014.worker', '--config', str(smoke_config),
                    '--output', str(smoke_dir), '--gpu-uuid', GPU_UUID, '--smoke']
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU_UUID, OMP_NUM_THREADS='2',
                           MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
                with (args.output / 'smoke.log').open('ab') as stream:
                    subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, check=True)
            smoke = json.loads((smoke_dir / 'SUMMARY.json').read_text())
            if (smoke.get('status') != 'complete' or not smoke.get('smoke_only')
                    or smoke['provenance']['git_sha'] != sha
                    or any(smoke['config'].get(key) != value for key, value in worst.items())):
                raise RuntimeError('Current largest-capacity smoke must complete on this exact source/config')
            provenance = dict(source_git_sha=sha, worker_root=str(ROOT), code_hashes=source_hashes(),
                launcher_sha256=digest(Path(__file__)), reference_sha256=digest(args.output / 'REFERENCE.json'),
                base_config_sha256=digest(BASE), dataset_sha256=DATASET_SHA,
                common_mask_hashes=mask_hashes, paired_mask_digest=flat['paired']['mask_digest'],
                smoke_sha256=digest(smoke_dir / 'SUMMARY.json'), peak_memory_mib=smoke['peak_memory_mib'],
                gpu=3, gpu_uuid=GPU_UUID, per_gpu_total_limit=4, new_training_budget=4,
                combined_budget=28, search_space=AXES, required_nondefault_knobs=REQUIRED_KNOBS,
                python=sys.version, torch=torch.__version__, numpy=np.__version__, optuna=optuna.__version__)
            provenance = json.loads(json.dumps(provenance))
            path = args.output / 'PREFLIGHT.json'
            if path.exists() and json.loads(path.read_text()) != provenance:
                raise RuntimeError('Do not mix numerical source/config/data/smoke versions')
            write_json(path, provenance)
            scheduler.smoke_costs = {'B3': {'peak_memory_mib': smoke['peak_memory_mib'],
                                           'disk_bytes': 2 * 2**30}}
            runner['status'] = 'running'
            write_json(args.output / 'RUNNER.json', runner)
            result = run_search(args.output, reference, scheduler)
            runner['status'] = result['status']
        except BaseException as error:
            runner.update(status='failed', error=repr(error))
            raise
        finally:
            runner['finished_unix'] = time.time()
            write_json(args.output / 'RUNNER.json', runner)


if __name__ == '__main__':
    main()
