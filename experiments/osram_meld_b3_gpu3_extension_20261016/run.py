"""Four additional unique B3 TPE configurations; never edit the original study."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

from experiments.osram_meld_b3_structure_hpo_20261016.run import (
    BASE, FLAT, make_config, DEFAULT_OUTPUT as ORIGINAL,
)
from experiments.osram_meld_b6_structure_hpo_20261015.run import StructureScheduler
from experiments.osram_meld_block2_auto_20261014.scheduler import write_json
from experiments.osram_meld_block2_auto_20261014.optuna_search import paired_objective

ROOT = Path(__file__).resolve().parents[2]
WORKER_ROOT = Path('/home/qinjinghui/remote_experiments/osram_meld_b3_code_737fdb1')
WORKER_SHA = '737fdb16e5b4f4ee7a426a0979efbc7f59c5ed8c'
GPU_UUID = 'GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a'
DEFAULT_OUTPUT = Path('/home/qinjinghui/remote_experiments/osram_meld_b3_gpu3_extension_20261016')
STUDY_NAME = 'meld_B3_gpu3_four_additional_seed66'


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def freeze_reference(output, original, flat_paired):
    """Copy the original trials once; every pending original structure is reserved."""
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'REFERENCE.json'
    if path.exists():
        return json.loads(path.read_text())
    source = original / 'TRIALS.json'
    source_bytes = source.read_bytes()
    rows = json.loads(source_bytes)
    queue_path = original / 'queue' / 'QUEUE.json'
    queue_bytes = queue_path.read_bytes()
    queue = json.loads(queue_bytes)
    jobs = list(queue['jobs'].values())
    if len(jobs) != 20:
        raise ValueError('Require all twenty original queue structures, including pending trials')
    reserved = [dict(bottlenecks=j['config']['osram_mbt_bottlenecks'],
                     rounds=j['config']['osram_mbt_rounds']) for j in jobs]
    for config in reserved:
        make_config(**config)
    if len({tuple(c[k] for k in ('bottlenecks', 'rounds')) for c in reserved}) != 20:
        raise ValueError('Original configurations must be distinct')
    completed = [dict(number=r['number'], params=r['params'],
                      value=paired_objective(r['user_attrs']['summary'], flat_paired))
                 for r in rows if r['state'] == 'COMPLETE']
    reference = dict(original_output=str(original),
                     original_trials_sha256=hashlib.sha256(source_bytes).hexdigest(),
                     original_queue_sha256=hashlib.sha256(queue_bytes).hexdigest(),
                     frozen_unix=time.time(), reserved=reserved, completed=completed,
                     flat_paired=flat_paired)
    write_json(path, reference)
    return reference


def unique_sampler(reserved):
    import optuna
    excluded = {(c['bottlenecks'], c['rounds']) for c in reserved}

    class UniqueTPE(optuna.samplers.TPESampler):
        def sample_independent(self, study, trial, param_name, distribution):
            used = excluded | {(t.params.get('bottlenecks'), t.params.get('rounds'))
                for t in study.get_trials(deepcopy=False)
                if t.number != trial.number and {'bottlenecks', 'rounds'} <= set(t.params)}
            if param_name == 'bottlenecks':
                allowed = [n for n in range(1, 9) if any((n, r) not in used for r in range(1, 5))]
            elif param_name == 'rounds':
                allowed = [r for r in range(1, 5) if (trial.params['bottlenecks'], r) not in used]
            else:
                raise ValueError('Unregistered search parameter')
            if not allowed:
                raise RuntimeError('Unique B3 space exhausted')
            for _ in range(64):
                candidate = super().sample_independent(study, trial, param_name, distribution)
                if candidate in allowed:
                    return candidate
            study._storage.set_trial_user_attr(trial._trial_id, 'unique_fallback_' + param_name, True)
            return random.Random(166 + trial.number).choice(allowed)

    return UniqueTPE(seed=166, n_startup_trials=5, constant_liar=True, multivariate=False)


def export(study, output, reference):
    rows = []
    for trial in study.trials:
        if trial.user_attrs.get('reference_only'):
            continue
        rows.append(dict(number=trial.number, job_id=trial.user_attrs.get('job_id'),
            params=trial.params, state=trial.state.name, value=trial.value,
            failure=trial.user_attrs.get('failure'), summary=trial.user_attrs.get('summary')))
    complete = [r for r in rows if r['state'] == 'COMPLETE']
    finished = len(rows) == 4 and all(r['state'] in ('COMPLETE', 'FAIL') for r in rows)
    report = dict(label='INTERNAL TEST-ORACLE SCREEN; seed66 only',
        status=('complete' if len(complete) == 4 else 'completed_with_failures') if finished else 'running',
        new_training_budget=4, original_budget=20, combined_budget=24,
        reference_trials=len(reference['completed']), reference_only_not_new_runs=True,
        original_output=reference.get('original_output'), new_runs=rows,
        best_paired_job=max(complete, key=lambda r: r['value'])['job_id'] if complete else None)
    write_json(output / 'SUMMARY.json', report)
    lines = ['# MELD B3: four GPU3 Bayesian extensions', '', report['label'], '',
        'Original twenty structures are excluded. Completed originals provide TPE evidence only.',
        'Exactly four new runs; original study is never modified. Original Flat1600/Local Skip retained.',
        'Objective: 0.6 × paired ΔACC + 0.4 × paired ΔUA. Test-best columns use their own selected masks.', '',
        '| Job | Tokens | Rounds | State | Test-best ACC | UA | Paired ACC | UA | Objective |',
        '|---|---:|---:|---|---:|---:|---:|---:|---:|']
    cells = []
    for row in rows:
        summary = row['summary'] or {}
        tb = summary.get('testbest', {}).get('mean', {})
        paired = summary.get('paired', {}).get('on', {}).get('mean', {})
        values = [tb.get('ACC'), tb.get('UA'), paired.get('ACC'), paired.get('UA'), row['value']]
        values = ['—' if value is None else f'{value:.4f}' for value in values]
        lines.append(f"| {row['job_id']} | {row['params'].get('bottlenecks')} | {row['params'].get('rounds')} | {row['state']} | {' | '.join(values)} |")
        for protocol, data in [('testbest', summary.get('testbest', {})),
                               ('paired-on', summary.get('paired', {}).get('on', {})),
                               ('paired-off', summary.get('paired', {}).get('off', {}))]:
            for rate, metrics in dict(data.get('rates', {}),
                                      **{k: data[k] for k in ('mean', 'high') if k in data}).items():
                cells.append(dict(job=row['job_id'], tokens=row['params'].get('bottlenecks'),
                    rounds=row['params'].get('rounds'), protocol=protocol, rate=rate,
                    **{k: metrics.get(k) for k in ('ACC', 'UA', 'WF1', 'CE')}))
    (output / 'RESULT.md').write_text('\n'.join(lines) + '\n')
    if cells:
        with (output / 'RESULT.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(cells[0]))
            writer.writeheader()
            writer.writerows(cells)
    return report


def run_extension(output, reference, scheduler):
    import optuna
    output.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(study_name=STUDY_NAME,
        storage='sqlite:///' + str(output / 'EXTENSION.sqlite3'), direction='maximize',
        sampler=unique_sampler(reference['reserved']), load_if_exists=True)
    protocol = dict(budget=4, reserved=reference['reserved'], references=reference['completed'],
                    objective='0.6*delta_ACC+0.4*delta_UA', worker_sha=WORKER_SHA)
    if study.user_attrs.get('protocol', protocol) != protocol:
        raise ValueError('Frozen extension protocol changed')
    study.set_user_attr('protocol', protocol)
    for row in reference['completed']:
        if not any(t.user_attrs.get('reference_source_number') == row['number'] for t in study.trials):
            study.add_trial(optuna.trial.create_trial(params=row['params'], value=row['value'],
                distributions={'bottlenecks': optuna.distributions.IntDistribution(1, 8),
                               'rounds': optuna.distributions.IntDistribution(1, 4)},
                user_attrs={'reference_only': True, 'reference_source_number': row['number']}))
    new = [t for t in study.trials if not t.user_attrs.get('reference_only')]
    if len(new) > 4:
        raise RuntimeError('Extension budget exceeded')
    running = []
    for index in range(4):
        trial = optuna.Trial(study, new[index]._trial_id) if index < len(new) else study.ask()
        frozen = study._storage.get_trial(trial._trial_id)
        if frozen.state.is_finished():
            continue
        trial.set_user_attr('job_id', f'b3_gpu3_add_{index + 1:02d}')
        trial.suggest_int('bottlenecks', 1, 8)
        trial.suggest_int('rounds', 1, 4)
        running.append(trial)
    jobs = [dict(id=t.user_attrs['job_id'], method='B3', config=make_config(**t.params),
                 output=str(output / 'runs' / t.user_attrs['job_id'])) for t in running]
    write_json(output / 'PLANNED_JOBS.json', jobs if jobs else json.loads(
        (output / 'PLANNED_JOBS.json').read_text()) if (output / 'PLANNED_JOBS.json').exists() else [])
    export(study, output, reference)
    if running:
        records = scheduler.run_jobs(jobs)
        for trial, record in zip(running, records):
            trial.set_user_attr('queue_record', {k: v for k, v in record.items() if k != 'summary'})
            try:
                if record['status'] != 'complete':
                    raise ValueError('Worker failed; inspect queue logs')
                value = paired_objective(record['summary'], reference['flat_paired'])
                trial.set_user_attr('summary', record['summary'])
            except (KeyError, TypeError, ValueError) as error:
                trial.set_user_attr('failure', str(error))
                study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
            else:
                study.tell(trial.number, value)
    return export(study, output, reference)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    import optuna
    import numpy as np
    import torch
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    worker_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WORKER_ROOT, text=True).strip()
    if worker_sha != WORKER_SHA or subprocess.check_output(
            ['git', 'diff', 'HEAD', '--name-only'], cwd=WORKER_ROOT, text=True).strip():
        raise RuntimeError('Immutable numerical worker source changed')
    if subprocess.check_output(['git', 'diff', 'HEAD', '--name-only'], cwd=ROOT, text=True).strip():
        raise RuntimeError('Controller must be committed')
    if shutil.disk_usage(args.output.parent).free < 30 * 2**30:
        raise RuntimeError('Require at least 30 GiB free disk')
    dataset = Path('/data2/yb/paper/remote_runs_20261012/data/meld/meld_emotion_semantic_features_roberta.pkl')
    if digest(dataset) != 'a0349614b2b58e8b2db768f0144aa36b96fb2348ca436a6879ee2623cc2d2f80':
        raise RuntimeError('MELD feature version mismatch')
    flat = json.loads(FLAT.read_text())
    if flat['status'] != 'complete' or flat['method'] != 'Flat':
        raise RuntimeError('Original-hyperparameter Flat paired reference invalid')
    expected = make_config(4, 2)
    for key in ('learning_rate', 'weight_decay', 'dropout', 'epochs', 'seed', 'batch_size',
                'missing_mask_protocol', 'osram_output_dim', 'osram_readout_fusion', 'osram_write_step'):
        if flat['config'].get(key) != expected[key]:
            raise RuntimeError('Flat comparison protocol mismatch: ' + key)
    if flat['config'].get('osram_disable_local_skip', False):
        raise RuntimeError('Flat comparison unexpectedly removes Local Skip')
    inventory = json.loads((ROOT / 'experiments/osram_meld_query_decision_20261014/CANONICAL_INPUTS.json').read_text())
    common = Path('/data2/yb/remote_experiments/osram_meld_testbest_ua_20261014/exact/source_0')
    mask_hashes = {str(i / 10): digest(common / f'predictions_miss_0p{i}.npz') for i in range(8)}
    if mask_hashes != inventory['mask_sha256']:
        raise RuntimeError('Protected common masks changed')
    smoke_path = ORIGINAL / 'smoke' / 'SUMMARY.json'
    smoke = json.loads(smoke_path.read_text())
    if (smoke.get('status') != 'complete' or not smoke.get('smoke_only')
            or smoke['provenance']['git_sha'] != WORKER_SHA
            or smoke['config']['osram_mbt_bottlenecks'] != 8
            or smoke['config']['osram_mbt_rounds'] != 4):
        raise RuntimeError('Previous largest-structure smoke is not reusable')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.controller.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        scheduler = StructureScheduler(args.output / 'queue', gpus=(3,), per_gpu=4,
            per_gpu_limit=4, code_root=WORKER_ROOT,
            smoke_costs={'B3': {'peak_memory_mib': smoke['peak_memory_mib'], 'disk_bytes': 2 * 2**30}})
        scheduler.global_per_gpu_limit = 4
        if scheduler.gpu_snapshot()[3]['uuid'] != GPU_UUID:
            raise RuntimeError('Physical GPU3 UUID mismatch')
        reference = freeze_reference(args.output, ORIGINAL, flat['paired'])
        if len(reference['completed']) < 5:
            raise RuntimeError('At least five completed original trials required for immediate TPE')
        provenance = dict(controller_git_sha=sha, numerical_worker_git_sha=worker_sha,
            controller_sha256=digest(Path(__file__)), worker_root=str(WORKER_ROOT),
            base_config_sha256=digest(BASE), reference_sha256=digest(args.output / 'REFERENCE.json'),
            dataset_sha256=digest(dataset), common_mask_hashes=mask_hashes,
            paired_mask_digest=flat['paired']['mask_digest'], reused_smoke_sha256=digest(smoke_path),
            peak_memory_mib=smoke['peak_memory_mib'], gpu=3, gpu_uuid=GPU_UUID,
            per_gpu_limit=4, new_training_budget=4, combined_training_budget=24,
            python=sys.version, torch=torch.__version__, numpy=np.__version__, optuna=optuna.__version__)
        path = args.output / 'PREFLIGHT.json'
        if path.exists() and json.loads(path.read_text()) != provenance:
            raise RuntimeError('Do not mix controller/worker/data/protocol versions')
        write_json(path, provenance)
        runner = dict(status='running', pid=os.getpid(), started_unix=time.time())
        write_json(args.output / 'RUNNER.json', runner)
        try:
            report = run_extension(args.output, reference, scheduler)
            runner.update(status=report['status'])
        except BaseException as error:
            runner.update(status='failed', error=repr(error))
            raise
        finally:
            runner['finished_unix'] = time.time()
            write_json(args.output / 'RUNNER.json', runner)


if __name__ == '__main__':
    main()
