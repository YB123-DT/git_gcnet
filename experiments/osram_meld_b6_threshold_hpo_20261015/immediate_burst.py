"""User-approved early five-trial burst on GPU0(3),GPU3(1),GPU5(1).

Existing GPU2 workers are adopted, never stopped. The five extra starts consume
the existing twenty-trial budget. They may precede startup feedback and must not
be advertised as feedback-informed Bayesian proposals. After the burst finishes,
the original finite adaptive controller resumes with its original protocol.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

CAPACITY = {0: 3, 3: 1, 5: 1}


def burst_count(issued):
    if type(issued) is not int or not 0 <= issued <= 20:
        raise ValueError('The total authorized budget remains twenty trials')
    return min(5, 20 - issued)


def make_scheduler(frozen, output, source_root, peak, rebalance=False):
    if rebalance:
        scheduler = frozen.ThresholdScheduler(output / 'queue', gpus=(0, 2, 3),
            per_gpu=6, code_root=source_root,
            smoke_costs={'B6': {'peak_memory_mib': peak, 'disk_bytes': 2 * 2**30}})
        scheduler.capacity = {0: 3, 2: 6, 3: 1, 5: 1}
        return scheduler
    scheduler = frozen.ThresholdScheduler(output / 'queue', gpus=tuple(CAPACITY),
        per_gpu=3, code_root=source_root,
        smoke_costs={'B6': {'peak_memory_mib': peak, 'disk_bytes': 2 * 2**30}})
    scheduler.capacity.update({gpu: min(6, added + sum(
        row.get('gpu') == gpu and row['status'] in ('running', 'launching')
        for row in scheduler.state['jobs'].values())) for gpu, added in CAPACITY.items()})
    return scheduler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=Path(
        '/home/qinjinghui/remote_experiments/osram_meld_b6_threshold_hpo_20261015'))
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--rebalance', action='store_true',
                        help='After audited full-state migrations, cap GPU3/5 at one; use GPU2 for excess')
    args = parser.parse_args()
    source = args.source_root.resolve(strict=True)
    sys.path.insert(0, str(source))
    from experiments.osram_meld_b6_threshold_hpo_20261015 import run as frozen
    from experiments.osram_meld_block2_auto_20261014.worker import source_hashes
    from experiments.osram_meld_b6_structure_hpo_20261015.run import export
    from experiments.osram_meld_block2_auto_20261014.optuna_search import paired_objective
    import optuna

    provenance = json.loads((args.output / 'PREFLIGHT.json').read_text())
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    if (Path(frozen.__file__).resolve().parents[2] != source
            or sha != provenance['source_git_sha'] or source_hashes() != provenance['code_hashes']):
        raise ValueError('Frozen training source changed')
    smoke = json.loads((args.output / 'smoke/SUMMARY.json').read_text())
    scheduler = make_scheduler(frozen, args.output, source, smoke['peak_memory_mib'], args.rebalance)
    snapshot = scheduler.gpu_snapshot()
    checked_gpus = set(scheduler.gpus) | {5}
    if any(gpu not in snapshot or not snapshot[gpu]['uuid'].startswith('GPU-')
           or snapshot[gpu]['uuid'] == snapshot.get(4, {}).get('uuid') for gpu in checked_gpus):
        raise RuntimeError('GPU UUID whitelist validation failed')
    if any(snapshot[gpu]['free_mib'] < 4000 for gpu in checked_gpus):
        raise RuntimeError('Insufficient current GPU headroom for immediate launch')
    print(json.dumps(dict(preflight='passed', capacity=scheduler.capacity, source_git_sha=sha)), flush=True)
    if args.check_only:
        return
    with (args.output / '.controller.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        study = optuna.load_study(study_name='meld_B6_rank2_steps14_threshold_lr_seed66',
            storage='sqlite:///' + str(args.output / 'STRUCTURE.sqlite3'), sampler=frozen.make_sampler())
        protocol = study.user_attrs['protocol']
        if (protocol['budget'] != 20 or protocol['max_pending'] != 6
                or protocol['base_config'] != frozen.make_config(.1, 1.)
                or protocol['initial'] != list(frozen.INITIAL_CONFIGS)):
            raise ValueError('Study settings changed; refuse mixing configurations')
        issued = [t for t in study.trials if t.state != optuna.trial.TrialState.WAITING]
        if len(issued) < 5:
            raise RuntimeError('Expected the five initial configurations already issued')
        journal_path = args.output / 'IMMEDIATE_BURST_V2.json'
        journal = json.loads(journal_path.read_text()) if journal_path.exists() else dict(
            user_override='Launch immediately rather than wait for initial-five feedback',
            first_started_unix=time.time(), source_git_sha=sha, capacity=scheduler.capacity,
            total_budget=20, issued_before_burst=len(issued),
            early_issued_target=len(issued) + burst_count(len(issued)), later_pending_limit=6,
            preserved_worker_pids=[j['pid'] for j in scheduler.state['jobs'].values()
                                   if j['status'] == 'running'])
        if not 5 <= journal['early_issued_target'] <= 20 or len(issued) > 20:
            raise ValueError('Invalid immediate-burst resume budget')
        if not args.rebalance:
            scheduler.capacity = {int(g): c for g, c in journal['capacity'].items()}
        scheduler.save()
        if args.rebalance:
            frozen.write_json(args.output / 'RESOURCE_REBALANCE.json', dict(
                pid=os.getpid(), started_unix=time.time(), capacity=scheduler.capacity,
                source_git_sha=sha, source_root=str(source),
                controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                reason='User requests GPU3/5 each one; audited full-checkpoint migration to GPU2'))
        journal.update(pid=os.getpid(), controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        frozen.write_json(journal_path, journal)
        for _ in range(max(0, journal['early_issued_target'] - len(issued))):
            t = study.ask()
            for name, spec in frozen.SEARCH_SPACE.items():
                t.suggest_float(name, **spec)
            t.set_user_attr('proposal_phase', 'user_authorized_early_burst')
            t.set_user_attr('completed_feedback_at_proposal', sum(
                x.state == optuna.trial.TrialState.COMPLETE for x in study.trials))
        active = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]
        jobs = []
        for t in active:
            live = optuna.Trial(study, t._trial_id)
            for name, spec in frozen.SEARCH_SPACE.items():
                live.suggest_float(name, **spec)
            name = f'b6_struct_{t.number:03d}'
            jobs.append(dict(id=name, method='B6', config=frozen.make_config(**live.params),
                             output=str(args.output / 'optuna/runs' / name)))
        runner = json.loads((args.output / 'RUNNER.json').read_text())
        runner.update(status='running', pid=os.getpid(), immediate_burst=journal)
        frozen.write_json(args.output / 'RUNNER.json', runner)
        try:
            if jobs:
                records = scheduler.run_jobs(jobs)
                flat = json.loads(frozen.FLAT.read_text())
                for t, record in zip(active, records):
                    study._storage.set_trial_user_attr(t._trial_id, 'queue_record',
                        {k: v for k, v in record.items() if k != 'summary'})
                    if record['status'] == 'complete':
                        value = paired_objective(record['summary'], flat['paired'])
                        study._storage.set_trial_user_attr(t._trial_id, 'summary', record['summary'])
                        study.tell(t.number, value)
                    else:
                        study.tell(t.number, state=optuna.trial.TrialState.FAIL)
                export(study, args.output, frozen.REPORT_METADATA)
            flat = json.loads(frozen.FLAT.read_text())
            result = frozen.run_search(args.output, flat['paired'], scheduler,
                budget=20, max_pending=6, config_factory=frozen.make_config,
                initial_configs=frozen.INITIAL_CONFIGS, search_space=frozen.SEARCH_SPACE,
                study_name='meld_B6_rank2_steps14_threshold_lr_seed66', sampler=frozen.make_sampler(),
                metadata=frozen.REPORT_METADATA)
            runner.update(status=result['status'], finished_unix=time.time())
        except BaseException as error:
            runner.update(status='failed', error=repr(error), finished_unix=time.time())
            raise
        finally:
            frozen.write_json(args.output / 'RUNNER.json', runner)


if __name__ == '__main__':
    main()
