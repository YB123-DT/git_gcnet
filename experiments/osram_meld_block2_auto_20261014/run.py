"""Finite MELD screen -> registered gate -> bounded real-training Optuna."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .methods import METHODS, build_config
from .scheduler import Scheduler, write_json
from .summarize import summarize
from .optuna_search import run_search

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path('/data2/yb/remote_experiments/osram_meld_block2_auto_20261014_v2')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('all', 'smoke', 'screen', 'hpo', 'summarize'), default='all')
    parser.add_argument('--gpus', default='0,2,3')
    parser.add_argument('--per-gpu', type=int, default=4)
    parser.add_argument('--seed', type=int, default=66)
    parser.add_argument('--top-k', type=int, default=2)
    parser.add_argument('--trials-per-model', type=int, default=20)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    args.gpus = [int(v) for v in args.gpus.split(',')]
    if (not 1 <= len(args.gpus) <= 3 or len(set(args.gpus)) != len(args.gpus)
            or 4 in args.gpus or any(v < 0 for v in args.gpus)
            or not 1 <= args.per_gpu <= 4 or args.seed != 66
            or not 1 <= args.top_k <= 2 or not 16 <= args.trials_per_model <= 20):
        parser.error('Finite seed66 scope: <=3 healthy GPUs, <=4/GPU, <=2 winners, 16–20 trials each')
    return args


def study_export(rows, selected):
    studies = []
    for method in selected:
        trials = [v for v in rows if v['method'] == method]
        complete = sorted((v for v in trials if v['state'] == 'COMPLETE'), key=lambda v: (-v['value'], v['number']))
        best = complete[0] if complete else None
        studies.append(dict(method=method, status='complete' if complete else 'failed', trials=trials,
            best_trial=None if best is None else best['number'],
            best_config=None if best is None else best['user_attrs']['queue_record']['config'],
            best_params=None if best is None else best['params']))
    return dict(studies=studies, status='complete', training_hpo_only=True,
                note='No module structural parameters exposed; every trial really trains50epochs. Test-selected exploration.')


def preflight(args):
    import optuna
    import torch
    import numpy
    from .worker import DATASET, DATASET_SHA, COMMON, source_hashes
    from gcnet_missing_m3.training_resume import _sha
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(['git', 'diff', 'HEAD', '--name-only'], cwd=ROOT, text=True).strip()
    if dirty:
        raise RuntimeError('Formal queue requires a committed immutable source snapshot: ' + dirty)
    if not DATASET.is_file() or _sha(DATASET) != DATASET_SHA:
        raise ValueError('Accepted official ECERC MELD feature pickle unavailable/changed')
    for i in range(8):
        if not (COMMON / f'predictions_miss_0p{i}.npz').is_file():
            raise FileNotFoundError('Missing protected common evaluation masks')
    if shutil.disk_usage(args.output.parent).free < 8 * 2**30:
        raise RuntimeError('Insufficient disk reserve for seven retained checkpoint sets')
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot = Scheduler(args.output / 'queue', args.gpus, args.per_gpu).gpu_snapshot()
    if any(g not in snapshot or g == 4 for g in args.gpus):
        raise RuntimeError('Requested healthy GPU missing')
    result = dict(experiment_version='MELD-BLOCK2-v2', source_git_sha=sha, source_hashes=source_hashes(),
        dataset_sha256=DATASET_SHA, gpu_inventory=snapshot, requested_gpus=args.gpus,
        per_gpu_limit=args.per_gpu, global_limit=min(12, len(args.gpus) * args.per_gpu),
        disk_free_bytes=shutil.disk_usage(args.output).free,
        environment=dict(python=sys.version, executable=sys.executable, torch=torch.__version__,
                         numpy=numpy.__version__, optuna=optuna.__version__),
        mask_protocol='gcnet-random-mask-v1', label='INTERNAL TEST-ORACLE SCREEN', seed=66,
        dataset_splits='official train/dev/test; dev not used for selection in this authorized exploration',
        selection_rule=dict(ACC_floor=-.30, UA_floor=-.50, score='0.6*dACC+0.4*dUA', positive_score_required=True),
        checkpoint_policy='retain all BEST weights/exact artifacts; bounded active recovery cache; never touch old baseline')
    old = args.output / 'PREFLIGHT.json'
    if old.exists():
        previous = json.loads(old.read_text())
        if any(previous[k] != result[k] for k in ('source_git_sha', 'source_hashes', 'dataset_sha256', 'seed')):
            raise ValueError('Existing experiment version incompatible: use new output and do not combine rankings')
    write_json(old, result)
    return result


def measure_smokes(args, base):
    costs = {}
    for method in METHODS:
        output = args.output / 'smoke' / method
        settings = build_config(base, method, output)
        config_path = args.output / 'configs' / f'smoke_{method}.json'
        write_json(config_path, settings)
        existing = output / 'SUMMARY.json'
        if existing.exists():
            value = json.loads(existing.read_text())
            if value.get('status') == 'complete' and value.get('smoke_only'):
                costs[method] = value['peak_memory_mib']
                continue
        snapshot = Scheduler(args.output / 'queue', args.gpus, args.per_gpu).gpu_snapshot()
        gpu = max(args.gpus, key=lambda g: snapshot[g]['free_mib'])
        if snapshot[gpu]['free_mib'] < 6000:
            raise RuntimeError('No admitted GPU room for worst-length real-batch smoke; external jobs untouched')
        uuid = snapshot[gpu]['uuid']
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='2')
        log = args.output / 'smoke' / f'{method}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-u', '-m', 'experiments.osram_meld_block2_auto_20261014.worker',
                   '--config', str(config_path), '--output', str(output), '--gpu-uuid', uuid, '--smoke']
        print(f'SMOKE {method} hostGPU{gpu}', flush=True)
        with log.open('ab') as stream:
            process = subprocess.Popen(command, env=env, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            code = process.wait()
        if code != 0:
            raise RuntimeError(f'{method} smoke failed; see {log}; no formal training launched')
        value = json.loads(existing.read_text())
        if value.get('status') != 'complete' or not value.get('smoke_only'):
            raise RuntimeError('Smoke did not produce an audited complete smoke result')
        costs[method] = value['peak_memory_mib']
    write_json(args.output / 'SMOKE_COSTS.json', costs)
    return costs


def main(argv=None):
    args = parse_args(argv)
    if args.phase == 'summarize':
        summarize(args.output, top_k=args.top_k)
        return
    provenance = preflight(args)
    runner = dict(status='running', pid=os.getpid(), started_unix=time.time(), command=sys.argv,
                  source_git_sha=provenance['source_git_sha'], phase=args.phase)
    write_json(args.output / 'RUNNER.json', runner)
    try:
        base = json.loads(Path(__file__).with_name('BASE_CONFIG.json').read_text())
        for method in METHODS:
            build_config(base, method, args.output / 'screen' / method)
        if args.phase in ('all', 'smoke', 'screen'):
            costs = measure_smokes(args, base)
        else:
            costs = json.loads((args.output / 'SMOKE_COSTS.json').read_text())
        if args.phase == 'smoke':
            runner['status'] = 'complete'
            return
        queue = Scheduler(args.output / 'queue', args.gpus, args.per_gpu, python=sys.executable,
                          code_root=ROOT, smoke_costs=costs)
        if args.phase in ('all', 'screen'):
            jobs = [dict(id='screen_' + method, method=method,
                config=build_config(base, method, args.output / 'screen' / method),
                output=str(args.output / 'screen' / method)) for method in METHODS]
            queue.run_jobs(jobs)
        screen = summarize(args.output, top_k=args.top_k)
        runner['screen_status'] = screen['status']
        runner['selected'] = screen['selected']
        write_json(args.output / 'RUNNER.json', runner)
        if screen['status'] in ('incomplete', 'invalid_comparison'):
            raise RuntimeError('Screen incomplete/incompatible; no HPO on invalid results')
        if args.phase in ('all', 'hpo') and screen['selected']:
            flat = screen['reports']['Flat']['paired']
            rows = run_search(args.output / 'optuna', screen['selected'], base, flat, queue,
                              trials_per_model=args.trials_per_model, top_k=args.top_k)
            export = study_export(rows, screen['selected'])
            write_json(args.output / 'optuna_search.json', export)
            summarize(args.output, top_k=args.top_k, optuna_results=export)
        elif not screen['selected']:
            write_json(args.output / 'optuna_search.json', dict(studies=[], status='skipped_no_positive_signal',
                reason='No candidate passed pre-registered gate; do not force a winner'))
            summarize(args.output, top_k=args.top_k)
        runner['status'] = 'complete'
    except BaseException as error:
        runner.update(status='failed', error=repr(error))
        raise
    finally:
        runner['finished_unix'] = time.time()
        write_json(args.output / 'RUNNER.json', runner)


if __name__ == '__main__':
    main()
