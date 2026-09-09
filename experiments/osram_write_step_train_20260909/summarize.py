#!/usr/bin/env python3
"""Summarize complete paired frozen-reference/retrained write-step evaluations."""
import argparse
import importlib.util
import itertools
from pathlib import Path
import statistics

EXPERIMENTS = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location('grid_summary',
    EXPERIMENTS / 'osram_write_step_grid_20260909/summarize_grid.py')
base = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(base)
SEEDS, RATES, DATASETS = base.SEEDS, base.RATES, base.DATASETS
ARMS = ('A', 'B', 'C', 'D')
COMPARISONS = (('C', 'A'), ('C', 'B'), ('B', 'A'), ('D', 'C'))


def normalized_config(config):
    result = dict(config)
    result.setdefault('osram_forward_slot_reuse', False)
    result.setdefault('osram_write_step', 1.)
    return result


def load_cross(root, grid_root, range_root, inherited_root):
    grid, provenance = {}, []
    for dataset, seed in itertools.product(DATASETS, SEEDS):
        paths = {
            'A': (inherited_root / f'iemocap4_seed{seed}/metadata.json' if dataset == 'iemocap4'
                  else grid_root / dataset / f'seed{seed}/metadata.json'),
            'B': range_root / dataset / f'seed{seed}/metadata.json',
            'C': root / dataset / f'seed_{seed}/C_train06_test06/metadata.json',
            'D': root / dataset / f'seed_{seed}/D_train06_test1/metadata.json',
        }
        metadata = {}
        for arm, path in paths.items():
            if not path.exists():
                raise ValueError(f'PENDING: missing {path}; no complete crossed summary written')
            data, records = base.read_metadata(path, seed)
            metadata[arm] = data
            for field, expected in (('evaluation_only', True), ('new_checkpoint_selection', False)):
                if data.get(field) is not expected:
                    raise ValueError(f'{field} mismatch: {path}')
            if data.get('selection_protocol') != '8-rate-mean-test-oracle':
                raise ValueError(f'selection_protocol mismatch: {path}')
            if data.get('epoch') is None:
                raise ValueError(f'missing epoch: {path}')
            mode = 'fixed0.6' if arm == 'B' else 'reference'
            wanted = set(itertools.product(RATES, (mode,)))
            if not wanted <= set(records) or (arm in ('C', 'D') and set(records) != wanted):
                raise ValueError(f'incomplete or unexpected rate/mode grid: {path}')
            if arm in ('C', 'D'):
                for field, expected in (('training_write_step', .6),
                                        ('evaluation_write_step', .6 if arm == 'C' else 1.)):
                    if data.get(field) != expected:
                        raise ValueError(f'{field} mismatch: {path}')
            for rate in RATES:
                result = records[rate, mode]
                grid[dataset, seed, rate, arm] = result
                provenance.append(dict(dataset=dataset, seed=seed, rate=f'{rate:g}', arm=arm,
                    source_metadata=str(path.resolve()), checkpoint=data['checkpoint'],
                    checkpoint_sha256=data['checkpoint_sha256'], epoch=data['epoch'],
                    selection_protocol=data['selection_protocol'], mask_sha256=result.get('mask_sha256'),
                    training_write_step=1. if arm in ('A', 'B') else .6,
                    evaluation_write_step=.6 if arm in ('B', 'C') else 1.))
        for first, second in (('A', 'B'), ('C', 'D')):
            for field in ('checkpoint', 'checkpoint_sha256', 'config', 'epoch', 'dataset', 'selection_protocol'):
                left, right = metadata[first].get(field), metadata[second].get(field)
                if field == 'config':
                    left, right = normalized_config(left), normalized_config(right)
                if left != right:
                    raise ValueError(f'{first}/{second} {field} mismatch: {dataset} seed {seed}')
        old, new = normalized_config(metadata['A']['config']), normalized_config(metadata['C']['config'])
        if old['osram_write_step'] != 1. or new['osram_write_step'] != .6:
            raise ValueError(f'config osram_write_step mismatch: {dataset} seed {seed}')
        if dict(old, osram_write_step=.6) != new:
            raise ValueError(f'config changed beyond osram_write_step: {dataset} seed {seed}')
        if metadata['A']['dataset'] != metadata['C']['dataset']:
            raise ValueError(f'dataset mismatch: {dataset} seed {seed}')
        for rate in RATES:
            hashes = {grid[dataset, seed, rate, arm].get('mask_sha256') for arm in ARMS}
            if len(hashes) != 1 or None in hashes or '' in hashes:
                raise ValueError(f'A/B/C/D mask mismatch/missing: {dataset} seed {seed} rate {rate}')
    return grid, provenance


def summarize(root, grid_root=None, range_root=None, inherited_root=None):
    root = Path(root)
    grid_root = Path(grid_root) if grid_root else EXPERIMENTS / 'osram_write_step_grid_20260909'
    range_root = Path(range_root) if range_root else EXPERIMENTS / 'osram_write_step_range_20260909'
    inherited_root = Path(inherited_root) if inherited_root else EXPERIMENTS / 'osram_write_intervention_20260909/full5'
    grid, provenance = load_cross(root, grid_root, range_root, inherited_root)
    samples, per_seed, seed_table, differences = {}, [], [], {}
    for dataset, seed in itertools.product(DATASETS, SEEDS):
        flat = {(rate, arm): base.flatten(grid[dataset, seed, rate, arm])
                for rate, arm in itertools.product(RATES, ARMS)}
        task_keys = {k for k in flat[0., 'A'] if k[0] == 'task'}
        if any({k for k in v if k[0] == 'task'} != task_keys for v in flat.values()):
            raise ValueError('task metric mismatch')
        for arm in ARMS:
            common = set.intersection(*(set(flat[rate, arm]) for rate in RATES))
            flat['all', arm] = {k: statistics.mean(flat[rate, arm][k] for rate in RATES) for k in common}
        for (rate, arm), values in flat.items():
            label = 'all' if rate == 'all' else f'{rate:g}'
            for (section, modality, metric), value in sorted(values.items()):
                key = dataset, label, arm, section, modality, metric
                samples.setdefault(key, []).append(value)
                per_seed.append(dict(dataset=dataset, seed=seed, rate=label, arm=arm,
                    section=section, modality=modality, metric=metric, value=value))
        for rate in (*RATES, 'all'):
            label = 'all' if rate == 'all' else f'{rate:g}'
            points = {arm: flat[rate, arm]['task', '', 'weighted_f1'] for arm in ARMS}
            row = dict(dataset=dataset, seed=seed, rate=label, **points)
            for left, right in COMPARISONS:
                comparison = f'{left}-{right}'
                row[comparison] = points[left]-points[right]
                differences.setdefault((dataset, label, comparison), []).append(row[comparison])
            seed_table.append(row)
    summary = []
    for key, values in sorted(samples.items()):
        if len(values) != len(SEEDS):
            raise ValueError(f'incomplete metric across seeds: {key}')
        summary.append(dict(zip(('dataset', 'rate', 'arm', 'section', 'modality', 'metric'), key),
            n_seeds=len(values), mean=statistics.mean(values), sample_std=statistics.stdev(values)))
    paired = [dict(dataset=d, rate=r, comparison=c, n_seeds=len(v), mean=statistics.mean(v),
        sample_std=statistics.stdev(v), positive_seeds=sum(x > 0 for x in v))
        for (d, r, c), v in sorted(differences.items())]
    for filename, rows in (('summary.csv', summary), ('per_seed_metrics.csv', per_seed),
        ('per_seed_cross.csv', seed_table), ('paired_deltas.csv', paired), ('provenance.csv', provenance)):
        base.write_csv(root / filename, rows)
    lines = ['# Write-step training: crossed evaluation', '',
        'COMPLETE paired table. INTERNAL DIAGNOSTIC ONLY. A=train1/test1; B=train1/test.6; '
        'C=train.6/test.6; D=train.6/test1. C/D use the same newly trained checkpoint with native model write steps. '
        'B is the inherited frozen-checkpoint intervention; A/B share the original checkpoint.', '',
        'Eta .6 was chosen AFTER previous Test results. These comparisons are adaptive, not independent generalization evidence. '
        'No ordering, new hypothesis, or retraining gate is imposed.', '',
        'Checkpoint selection retains the inherited eight-rate-mean Test-oracle policy. The tables below use a distinct '
        'five-rate mean (0/.1/.3/.5/.7), not the checkpoint-selection score. C/D share checkpoint hash/config/epoch; '
        'A/B/C/D masks match within each seed/rate. Configuration differs only in osram_write_step after normalizing '
        'legacy missing osram_forward_slot_reuse=False and osram_write_step=1. Metadata flags prohibit evaluation-time '
        'checkpoint reselection or weight changes. This is metadata validation, not an independent checkpoint-file rehash.', '',
        'Wf1 is percent; deltas are percentage points. Means and sample SD use five equally weighted paired seeds. '
        '`all` averages five rates inside each seed before computing mean/SD. Rates/heads are not independent replicates.', '']
    for dataset in DATASETS:
        lines += [f'## {dataset}: Wf1 (%)', '', '| Rate | A | B | C | D |', '|---|---|---|---|---|']
        for rate in ('0', '0.1', '0.3', '0.5', '0.7', 'all'):
            selected = {r['arm']:r for r in summary if r['dataset']==dataset and r['rate']==rate and r['metric']=='weighted_f1' and r['section']=='task'}
            lines.append('| '+rate+' | '+' | '.join(f"{100*selected[a]['mean']:.3f} ± {100*selected[a]['sample_std']:.3f}" for a in ARMS)+' |')
        lines += ['', '| Rate | Comparison | Paired mean ± SD (pp) | Positive seeds |', '|---|---|---|---|']
        for row in paired:
            if row['dataset']==dataset:
                lines.append(f"| {row['rate']} | {row['comparison']} | {100*row['mean']:.3f} ± {100*row['sample_std']:.3f} | {row['positive_seeds']}/5 |")
        lines += ['', '| Seed | Rate | A | B | C | D | C−A | C−B | B−A | D−C |', '|---|---|---|---|---|---|---|---|---|---|']
        for row in seed_table:
            if row['dataset']==dataset:
                lines.append(f"| {row['seed']} | {row['rate']} | "+' | '.join(f'{100*row[k]:.3f}' for k in (*ARMS, 'C-A', 'C-B', 'B-A', 'D-C'))+' |')
    lines += ['', '## Old-read and observed new-write errors', '',
        'E_old=retention.err_decay excludes NO_HISTORY (not err_post). E_new=current_observed_write_fit.err_after. '
        'Modalities remain separate; undefined rate-zero retention is omitted, never zero-filled. No five-rate error '
        'aggregate is fabricated if a rate is undefined. Run-level means are equally weighted across seeds. '
        'These diagnostics alone do not establish downstream causality.', '',
        '| Dataset | Rate | Arm | Error | Modality | Mean ± SD |', '|---|---|---|---|---|---|']
    for row in summary:
        if row['section'] != 'task':
            lines.append(f"| {row['dataset']} | {row['rate']} | {row['arm']} | {row['metric']} | {row['modality']} | {row['mean']:.6g} ± {row['sample_std']:.6g} |")
    lines += ['', 'CSV files retain raw units. provenance.csv records every source/checkpoint/hash/epoch/mask; '
        'per_seed_cross.csv and paired_deltas.csv retain paired Wf1 comparisons; per_seed_metrics.csv and summary.csv '
        'retain task and error metrics. No causal or significance claim follows from five descriptive seeds.', '']
    (root / 'SUMMARY.md').write_text('\n'.join(lines), encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', type=Path)
    parser.add_argument('--remote-root', type=Path, help='Filesystem root containing remote experiment folders')
    parser.add_argument('--grid-root', type=Path)
    parser.add_argument('--range-root', type=Path)
    parser.add_argument('--inherited-root', type=Path)
    args = parser.parse_args()
    parent = args.remote_root or EXPERIMENTS
    try:
        summarize(args.root or parent / 'osram_write_step_train_20260909',
            args.grid_root or parent / 'osram_write_step_grid_20260909',
            args.range_root or parent / 'osram_write_step_range_20260909',
            args.inherited_root or parent / 'osram_write_intervention_20260909/full5')
    except ValueError as error:
        parser.exit(2, str(error)+'\n')
    print('Wrote complete SUMMARY.md and five CSV artifacts')
