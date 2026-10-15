"""Resume the existing finite study on GPU2/3/5 without restarting workers.

Execute this standalone file with the frozen training checkout as source-root.
Only scheduling changes: original source, five-start feedback barrier, six
pending trials, twenty-trial budget, configs, and optimization protocol remain.
This entry point never sends signals. The old controller must release its lock.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=Path(
        '/home/qinjinghui/remote_experiments/osram_meld_b6_threshold_hpo_20261015'))
    parser.add_argument('--gpus', default='2,3,5')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(argv)
    try:
        args.gpus = [int(value) for value in args.gpus.split(',')]
    except ValueError:
        parser.error('GPU indices must be integers')
    if (not args.gpus or len(set(args.gpus)) != len(args.gpus)
            or not set(args.gpus) <= {2, 3, 5} or 2 not in args.gpus):
        parser.error('Only authorized physical GPUs2/3/5, including existing GPU2')
    return args


def validate_protocol(protocol, config, initial):
    expected = dict(budget=20, max_pending=6, base_config=config,
                    initial=list(initial), warmstart_parallel=5)
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError('Resource handoff must preserve study budget/config/startup barrier')


def make_scheduler(frozen, output, source_root, gpus, peak):
    return frozen.ThresholdScheduler(output / 'queue', gpus=gpus, per_gpu=6,
        code_root=source_root,
        smoke_costs={'B6': {'peak_memory_mib': peak, 'disk_bytes': 2 * 2**30}})


def main(argv=None):
    args = parse_args(argv)
    args.source_root = args.source_root.resolve(strict=True)
    sys.path.insert(0, str(args.source_root))
    from experiments.osram_meld_b6_threshold_hpo_20261015 import run as frozen
    from experiments.osram_meld_block2_auto_20261014.worker import source_hashes
    import optuna

    if Path(frozen.__file__).resolve().parents[2] != args.source_root:
        raise RuntimeError('Imports did not resolve to the original frozen training checkout')
    preflight = json.loads((args.output / 'PREFLIGHT.json').read_text())
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                   cwd=args.source_root, text=True).strip()
    if sha != preflight['source_git_sha'] or source_hashes() != preflight['code_hashes']:
        raise ValueError('Training source no longer matches the original immutable provenance')
    study = optuna.load_study(study_name='meld_B6_rank2_steps14_threshold_lr_seed66',
        storage='sqlite:///' + str(args.output / 'STRUCTURE.sqlite3'))
    validate_protocol(study.user_attrs['protocol'], frozen.make_config(.1, 1.), frozen.INITIAL_CONFIGS)
    smoke = json.loads((args.output / 'smoke/SUMMARY.json').read_text())
    scheduler = make_scheduler(frozen, args.output, args.source_root,
                               args.gpus, smoke['peak_memory_mib'])
    snapshot = scheduler.gpu_snapshot()
    if any(gpu not in snapshot or not snapshot[gpu]['uuid'].startswith('GPU-')
           or snapshot[gpu]['uuid'] == snapshot.get(4, {}).get('uuid') for gpu in args.gpus):
        raise RuntimeError('Healthy GPU UUID validation failed')
    print(json.dumps(dict(check='passed', source_git_sha=sha, gpus=args.gpus,
                         max_pending=6, total_trials=20, snapshot=snapshot)), flush=True)
    if args.check_only:
        return
    with (args.output / '.controller.lock').open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        queue = json.loads((args.output / 'queue/QUEUE.json').read_text())
        record = dict(status='running', pid=os.getpid(), started_unix=time.time(),
            source_git_sha=sha, controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            source_root=str(args.source_root), gpus=args.gpus, max_pending=6,
            total_trials=20, gpu_snapshot=snapshot,
            preserved_worker_pids=[row['pid'] for row in queue['jobs'].values()
                                   if row['status'] == 'running'],
            change='User requested GPU3/5 additions; resource-only handoff; no worker termination')
        frozen.write_json(args.output / 'RESOURCE_RESUME.json', record)
        runner = json.loads((args.output / 'RUNNER.json').read_text())
        runner.update(status='running', pid=os.getpid(), resource_resume=record)
        frozen.write_json(args.output / 'RUNNER.json', runner)
        try:
            flat = json.loads(frozen.FLAT.read_text())
            result = frozen.run_search(args.output, flat['paired'], scheduler,
                budget=20, max_pending=6, config_factory=frozen.make_config,
                initial_configs=frozen.INITIAL_CONFIGS, search_space=frozen.SEARCH_SPACE,
                study_name='meld_B6_rank2_steps14_threshold_lr_seed66',
                sampler=frozen.make_sampler(), metadata=frozen.REPORT_METADATA)
            runner.update(status=result['status'], finished_unix=time.time())
        except BaseException as error:
            runner.update(status='failed', error=repr(error), finished_unix=time.time())
            raise
        finally:
            frozen.write_json(args.output / 'RUNNER.json', runner)


if __name__ == '__main__':
    main()
