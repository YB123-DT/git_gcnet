"""Frozen two-B6 replication plus one shared training-hyperparameter Flat control.

Run this standalone launcher against the existing immutable9fa9f57 training
checkout. No training/model source is edited. Seven new runs; two seed66 runs
are reused with their original source provenance, never relabelled as new runs.
"""
import argparse
from copy import deepcopy
import csv
import fcntl
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

TRAINING_SHA = '9fa9f571e7095d5b09f9c256d888c31a76c82744'
BLOCK = 'learned_robust_pca_evidence_decomposition'
REMOTE = Path('/home/qinjinghui/remote_experiments')
SOURCE_PATHS = {
    'B6_default': REMOTE / 'osram_meld_b6_structure_hpo_20261015/optuna/runs/b6_struct_003/SUMMARY.json',
    'B6_tuned': REMOTE / 'osram_meld_b6_threshold_hpo_20261015/optuna/runs/b6_struct_011/SUMMARY.json',
}
DEFAULT_OUTPUT = REMOTE / 'osram_meld_b6_top2_matched_flat_3seed_20261015'


def make_jobs(sources, output):
    if set(sources) != {'B6_default', 'B6_tuned'}:
        raise ValueError('Exactly the two user-chosen B6 sources are required')
    configs = {}
    for group, item in sources.items():
        s = item['summary']
        c = deepcopy(s['config'])
        c.setdefault('osram_rpca_threshold_init', .1)
        c.setdefault('osram_rpca_lr_multiplier', 1.)
        expected = dict(dataset='MELD', seed=66, epochs=50, batch_size=32,
            learning_rate=.0003433234536467373, weight_decay=4.4857894231052665e-6,
            dropout=.22489055023388654, missing_mask_protocol='gcnet-random-mask-v1',
            training_objective='emotion-only', checkpoint_selection='test-oracle-per-rate',
            checkpoint_selection_metric='accuracy', osram_meaningful_block=BLOCK,
            osram_rpca_rank=2, osram_rpca_steps=14, osram_disable_local_skip=False,
            preserve_selection_artifacts=True, evaluate_test=True)
        if s['status'] != 'complete' or s['method'] != 'B6' or any(c.get(k) != v for k, v in expected.items()):
            raise ValueError('Source selection or fixed training protocol mismatch: ' + group)
        target = (.1, 1.) if group == 'B6_default' else (.33599569865951234, 2.8200380294401013)
        if not all(math.isclose(c[k], v, rel_tol=1e-12) for k, v in zip(
                ('osram_rpca_threshold_init', 'osram_rpca_lr_multiplier'), target)):
            raise ValueError('Do not silently substitute another threshold/LR configuration')
        configs[group] = c
    allowed = {'osram_rpca_threshold_init', 'osram_rpca_lr_multiplier'}
    if any(configs['B6_default'][k] != configs['B6_tuned'].get(k)
           for k in set(configs['B6_default']) | set(configs['B6_tuned']) if k not in allowed):
        raise ValueError('B6 configurations differ beyond their two selected settings')
    flat = deepcopy(configs['B6_default'])
    flat['osram_meaningful_block'] = 'none'
    configs = {'Flat_matched': flat, **configs}
    jobs = []
    for group, config in configs.items():
        for seed in ((66, 67, 68) if group == 'Flat_matched' else (67, 68)):
            c = deepcopy(config)
            c['seed'] = seed
            name = f'{group}_seed{seed}'
            jobs.append(dict(id=name, group=group, method='Flat' if group == 'Flat_matched' else 'B6',
                             config=c, output=str(Path(output) / 'runs' / name)))
    return jobs


def make_scheduler(frozen, root, source, peak=2048.):
    class ReplicationScheduler(frozen.ThresholdScheduler):
        def command(self, job, config_path, uuid):
            command = super().command(job, config_path, uuid)
            if job['config']['seed'] != 66:
                command += ['--replication-seed', str(job['config']['seed'])]
            return command

        def recover(self):
            super().recover()
            for row in self.state['jobs'].values():
                checkpoint = Path(row['output']) / 'last_training.pt'
                if row['status'] == 'pending' and checkpoint.exists():
                    import torch
                    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
                    if not state.get('rng', {}).get('cuda'):
                        row.update(status='failed', resume_blocked='Saved CUDA RNG list is empty; refuse incomplete recovery')

        def failure(self, row, reason, returncode=None):
            super().failure(row, reason, returncode)
            if row['status'] == 'pending':
                row.update(status='failed', resume_blocked='Do not auto-retry incomplete CUDA RNG recovery; inspect explicitly')

    scheduler = ReplicationScheduler(root, gpus=(2, 3, 5), per_gpu=5, code_root=source,
        smoke_costs={name: {'peak_memory_mib': peak, 'disk_bytes': 2 * 2**30} for name in ('Flat', 'B6')})
    scheduler.capacity = {2: 5, 3: 1, 5: 1}
    return scheduler


def summarize(sources, records, output):
    report = dict(status='complete', label='INTERNAL TEST-ORACLE THREE-SEED REPLICATION', groups={})
    for group in ('Flat_matched', 'B6_default', 'B6_tuned'):
        seeds = {} if group == 'Flat_matched' else {'66': sources[group]['summary']}
        relevant = [r for r in records if r['group'] == group]
        failures = []
        for row in relevant:
            if row['status'] == 'complete':
                seeds[str(row['config']['seed'])] = row['summary']
            else:
                failures.append(dict(id=row['id'], status=row['status'], failures=row.get('failures', []),
                                     resume_blocked=row.get('resume_blocked')))
        compact = {seed: {k: s[k] for k in ('testbest', 'paired', 'provenance') if k in s}
                   for seed, s in seeds.items()}
        result = dict(seeds=compact, failures=failures)
        if set(seeds) == {'66', '67', '68'}:
            result['three_seed'] = {section: {metric: dict(
                mean=statistics.mean(s['testbest'][section][metric] for s in seeds.values()),
                std_ddof1=statistics.stdev(s['testbest'][section][metric] for s in seeds.values()))
                for metric in ('ACC', 'UA', 'WF1', 'CE')} for section in ('mean', 'high')}
        else:
            report['status'] = 'incomplete_or_failed'
        report['groups'][group] = result
    output = Path(output)
    (output / 'SUMMARY.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    table = []
    lines = ['# Matched Flat and two B6 configurations', '', report['label'], '',
             '| Configuration | Seed | Eight-rate ACC | UA | W-F1 |', '|---|---:|---:|---:|---:|']
    for group, result in report['groups'].items():
        for seed, s in sorted(result['seeds'].items()):
            mean = s['testbest']['mean']
            lines.append(f"| {group} | {seed} | {mean['ACC']:.3f} | {mean['UA']:.3f} | {mean['WF1']:.3f} |")
            for section, values in s['testbest'].items():
                if section == 'rates':
                    for rate, m in values.items():
                        table.append(dict(group=group, seed=seed, rate=rate, **m))
                elif section in ('mean', 'high'):
                    table.append(dict(group=group, seed=seed, rate=section, **values))
        if 'three_seed' in result:
            values = result['three_seed']['mean']
            cells = [f"{values[k]['mean']:.3f} ± {values[k]['std_ddof1']:.3f}" for k in ('ACC', 'UA', 'WF1')]
            lines.append(f"| {group} | mean ± sample SD | {' | '.join(cells)} |")
    fields = sorted({k for row in table for k in row})
    with (output / 'SUMMARY.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)
    lines += ['', 'ACC-selected exact-epoch UA/W-F1; eight independent rate selections, not one deployment checkpoint.',
              'Two selected B6 seed66 results are reused with original source provenance; seven new jobs use frozen9fa9f57.',
              'Original accepted Flat60.693/UA36.255 is a different hyperparameter baseline and is not replaced.',
              'Common-mask replay remains separate in each worker summary. Test-best table is not paired-mask performance.',
              'Seed66 selected configurations; this is exploratory, not an independent test estimate.', '']
    (output / 'RESULT.md').write_text('\n'.join(lines))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    source = args.source_root.resolve(strict=True)
    sys.path.insert(0, str(source))
    from experiments.osram_meld_b6_threshold_hpo_20261015 import run as frozen
    from experiments.osram_meld_block2_auto_20261014.worker import DATASET, DATASET_SHA, source_hashes, _sha
    from experiments.osram_meld_block2_auto_20261014.scheduler import write_json
    if (Path(frozen.__file__).resolve().parents[2] != source or
            subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip() != TRAINING_SHA):
        raise RuntimeError('Require original immutable9fa9f57 training source')
    original = json.loads((REMOTE / 'osram_meld_b6_threshold_hpo_20261015/PREFLIGHT.json').read_text())
    if source_hashes() != original['code_hashes'] or _sha(DATASET) != DATASET_SHA:
        raise RuntimeError('Training source or MELD features no longer match frozen provenance')
    sources = {group: dict(source_summary=str(path), source_summary_sha256=_sha(path),
                          summary=json.loads(path.read_text())) for group, path in SOURCE_PATHS.items()}
    jobs = make_jobs(sources, args.output)
    if shutil.disk_usage(args.output.parent).free < 25 * 2**30:
        raise RuntimeError('Insufficient disk headroom')
    scheduler = make_scheduler(frozen, args.output / 'queue', source)
    snapshot = scheduler.gpu_snapshot()
    if any(snapshot[g]['uuid'] == snapshot.get(4, {}).get('uuid') or not snapshot[g]['uuid'].startswith('GPU-')
           for g in scheduler.gpus):
        raise RuntimeError('Healthy device UUID validation failed')
    manifest = dict(label='INTERNAL TEST-ORACLE THREE-SEED REPLICATION', training_source_git_sha=TRAINING_SHA,
        launcher_sha256=_sha(Path(__file__)), source_root=str(source), sources=sources, jobs=jobs,
        dataset_sha256=DATASET_SHA, capacities={2: 5, 3: 1, 5: 1}, total_new_jobs=7,
        admission_reserve_mib=2048, protocol='gcnet-random-mask-v1;50epochs;per-rate Test-best ACC',
        provenance_note='Default B6 seed66 uses original9048ffd; .1/1 defaults are preserved in9fa9f57; no bitwise rerun asserted')
    print(json.dumps(dict(preflight='passed', jobs=[j['id'] for j in jobs],
                         capacity=scheduler.capacity, gpu_snapshot=snapshot)), flush=True)
    if args.check_only:
        return
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.controller.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = args.output / 'MANIFEST.json'
        if old.exists() and json.loads(old.read_text()) != json.loads(json.dumps(manifest)):
            raise ValueError('Do not mix sources/configs/launcher versions in this output')
        write_json(old, manifest)
        import os
        runner = dict(status='running', pid=os.getpid(), started_unix=time.time(),
                      training_source_git_sha=TRAINING_SHA, launcher_sha256=manifest['launcher_sha256'])
        write_json(args.output / 'RUNNER.json', runner)
        try:
            records = scheduler.run_jobs(jobs)
            result = summarize(sources, records, args.output)
            runner.update(status=result['status'], finished_unix=time.time())
        except BaseException as error:
            runner.update(status='failed', error=repr(error), finished_unix=time.time())
            raise
        finally:
            write_json(args.output / 'RUNNER.json', runner)


if __name__ == '__main__':
    main()
