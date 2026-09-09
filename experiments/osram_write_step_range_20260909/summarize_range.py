#!/usr/bin/env python3
"""Describe the adaptively extended frozen-checkpoint eta range, without a retrain gate."""
import argparse
import importlib.util
import itertools
from pathlib import Path
import statistics

_SPEC = importlib.util.spec_from_file_location('write_step_grid_summary',
    Path(__file__).resolve().parents[1] / 'osram_write_step_grid_20260909/summarize_grid.py')
base = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(base)
SEEDS, RATES, DATASETS = base.SEEDS, base.RATES, base.DATASETS
NEW_MODES = ('fixed0.6', 'fixed0.4', 'fixed0.2', 'fixed0.0')
MODES = base.MODES + NEW_MODES
ETAS = (1., .95, .9, .8, .6, .4, .2, 0.)


def load_grid(root, grid_root, inherited_root):
    grid, provenance = base.load_grid(grid_root, inherited_root)
    for row in provenance:
        row['inherited'] = True
    for dataset, seed in itertools.product(DATASETS, SEEDS):
        path = root / dataset / f'seed{seed}/metadata.json'
        data, records = base.read_metadata(path, seed)
        old, _ = base.read_metadata(grid_root / dataset / f'seed{seed}/metadata.json', seed)
        for field in ('checkpoint', 'checkpoint_sha256', 'config', 'dataset', 'epoch', 'selection_protocol'):
            if data.get(field) != old.get(field):
                raise ValueError(f'inherited {field} mismatch: {dataset} seed {seed}')
        if set(records) != set(itertools.product(RATES, NEW_MODES)):
            raise ValueError(f'incomplete or unexpected range rate/mode grid: {path}')
        for (rate, mode), result in records.items():
            grid[dataset, seed, rate, mode] = result
            provenance.append(dict(dataset=dataset, seed=seed, rate=f'{rate:g}', mode=mode,
                inherited=False, source_metadata=str(path.resolve()), checkpoint=data['checkpoint'],
                checkpoint_sha256=data['checkpoint_sha256'], mask_sha256=result.get('mask_sha256')))
        for rate in RATES:
            hashes = {grid[dataset, seed, rate, mode].get('mask_sha256') for mode in MODES}
            if len(hashes) != 1 or None in hashes or '' in hashes:
                raise ValueError(f'mask mismatch/missing: {dataset} seed {seed} rate {rate}')
    return grid, provenance


def summarize(root, grid_root=None, inherited_root=None):
    root = Path(root)
    experiments = Path(__file__).resolve().parents[1]
    grid_root = Path(grid_root) if grid_root else experiments / 'osram_write_step_grid_20260909'
    inherited_root = Path(inherited_root) if inherited_root else experiments / 'osram_write_intervention_20260909/full5'
    grid, provenance = load_grid(root, grid_root, inherited_root)
    samples, per_seed, points, best_rows, adjacent = {}, [], {}, [], []
    for dataset, seed in itertools.product(DATASETS, SEEDS):
        flat = {(rate, mode): base.flatten(grid[dataset, seed, rate, mode])
                for rate, mode in itertools.product(RATES, MODES)}
        task_keys = {k for k in flat[0., 'reference'] if k[0] == 'task'}
        if any({k for k in values if k[0] == 'task'} != task_keys for values in flat.values()):
            raise ValueError('task metric mismatch')
        for mode in MODES:
            common = set.intersection(*(set(flat[rate, mode]) for rate in RATES))
            flat['all', mode] = {k: statistics.mean(flat[rate, mode][k] for rate in RATES) for k in common}
        for (rate, mode), values in flat.items():
            label = 'all' if rate == 'all' else f'{rate:g}'
            points[dataset, seed, label, mode] = values['task', '', 'weighted_f1']
            for (section, modality, metric), value in sorted(values.items()):
                key = dataset, label, mode, section, modality, metric
                samples.setdefault(key, []).append(value)
                per_seed.append(dict(dataset=dataset, seed=seed, rate=label, mode=mode,
                    section=section, modality=modality, metric=metric, value=value))
        for rate in (*RATES, 'all'):
            label = 'all' if rate == 'all' else f'{rate:g}'
            values = [points[dataset, seed, label, mode] for mode in MODES]
            best_rows.append(dict(dataset=dataset, seed=seed, rate=label,
                best_sampled_etas=';'.join(f'{eta:g}' for eta, value in zip(ETAS, values) if value == max(values)),
                best_weighted_f1=max(values)))
            for i in range(len(MODES)-1):
                adjacent.append(dict(dataset=dataset, seed=seed, rate=label,
                    higher_eta=ETAS[i], lower_eta=ETAS[i+1],
                    lower_minus_higher=values[i+1]-values[i]))
    summary = []
    for key, values in sorted(samples.items()):
        if len(values) != len(SEEDS):
            raise ValueError(f'incomplete metric across seeds: {key}')
        summary.append(dict(zip(('dataset', 'rate', 'mode', 'section', 'modality', 'metric'), key),
            n_seeds=len(values), mean=statistics.mean(values), sample_std=statistics.stdev(values)))
    brackets, support = [], []
    for dataset, rate in itertools.product(DATASETS, ('0', '0.1', '0.3', '0.5', '0.7', 'all')):
        means = [statistics.mean(points[dataset, seed, rate, mode] for seed in SEEDS) for mode in MODES]
        # Preserve every exact tied sampled maximum instead of silently selecting one.
        for i, mean in enumerate(means):
            if mean != max(means):
                continue
            row = dict(dataset=dataset, rate=rate, peak_eta=ETAS[i],
                lower_eta=ETAS[i+1] if i+1 < len(ETAS) else '',
                higher_eta=ETAS[i-1] if i else '',
                location=('off-memory boundary; no interior optimum established' if ETAS[i] == 0 else
                          'upper tested boundary; no interior optimum established' if i == 0 else 'interior sampled peak'),
                tied_sampled_peaks=sum(value == mean for value in means))
            for direction, j in (('lower', i+1), ('higher', i-1)):
                differences = []
                if 0 <= j < len(MODES):
                    for seed in SEEDS:
                        delta = points[dataset, seed, rate, MODES[i]] - points[dataset, seed, rate, MODES[j]]
                        differences.append(delta)
                        support.append(dict(dataset=dataset, seed=seed, rate=rate, peak_eta=ETAS[i],
                            neighbor_eta=ETAS[j], neighbor_direction=direction, peak_minus_neighbor=delta))
                row[f'mean_vs_{direction}'] = statistics.mean(differences) if differences else ''
                row[f'std_vs_{direction}'] = statistics.stdev(differences) if differences else ''
                row[f'positive_vs_{direction}'] = sum(d > 0 for d in differences) if differences else ''
            brackets.append(row)
    for filename, rows in (('summary.csv', summary), ('per_seed_metrics.csv', per_seed),
        ('provenance.csv', provenance), ('per_seed_best.csv', best_rows),
        ('per_seed_adjacent.csv', adjacent), ('peak_brackets.csv', brackets), ('per_seed_peak_support.csv', support)):
        base.write_csv(root / filename, rows)
    lines = ['# Frozen write-step range extension', '',
        'INTERNAL DIAGNOSTIC ONLY. The extension (.6, .4, .2, 0) was chosen AFTER the previous four-point results; '
        'the combined eight points were not all preregistered. Inference only; no training, new mechanisms, or retrain gate.', '',
        'Eta 0 disables memory context at frozen weights; it is not a trained Local model. '
        'Checkpoint/config/epoch/selection metadata and per-seed/per-rate masks match across inherited and new sources; '
        'weights-unchanged flags are true. This validates metadata, not an independent checkpoint-file rehash. '
        'Exact source paths and recorded hashes are in provenance.csv.', '',
        'Wf1 is percent; differences are percentage points (pp). Mean ± sample SD uses five equally weighted seeds. '
        '`all` first averages five rates within each seed; rates and heads are not independent replicates. '
        'A sampled-mean peak and its nearest tested lower/higher eta only describe a sampled bracket, not a continuous optimum. '
        'Five seeds and adaptive extension do not support a significance or unbiased model-selection claim.', '']
    for dataset in DATASETS:
        lines += [f'## {dataset}: weighted F1 (%)', '',
            '| Rate | '+' | '.join(f'eta {eta:g}' for eta in ETAS)+' |', '|---|'+'---|'*len(ETAS)]
        for rate in ('0', '0.1', '0.3', '0.5', '0.7', 'all'):
            selected = {r['mode']:r for r in summary if r['dataset']==dataset and r['rate']==rate and r['section']=='task' and r['metric']=='weighted_f1'}
            lines.append('| '+rate+' | '+' | '.join(f"{100*selected[m]['mean']:.3f} ± {100*selected[m]['sample_std']:.3f}" for m in MODES)+' |')
        lines += ['', 'Five-rate per-seed best sampled eta(s): '+', '.join(f"seed {r['seed']}: {r['best_sampled_etas']}" for r in best_rows if r['dataset']==dataset and r['rate']=='all')+'.', '',
            '| Rate | Sampled peak eta | Lower / higher tested eta | Location | Peak−lower pp (positive seeds) | Peak−higher pp (positive seeds) |',
            '|---|---|---|---|---|---|']
        for row in brackets:
            if row['dataset'] != dataset:
                continue
            comparisons = [f"{100*row[f'mean_vs_{d}']:.3f} ± {100*row[f'std_vs_{d}']:.3f} ({row[f'positive_vs_{d}']}/5)" if row[f'mean_vs_{d}'] != '' else 'N/A' for d in ('lower', 'higher')]
            lines.append(f"| {row['rate']} | {row['peak_eta']:g} | {row['lower_eta']} / {row['higher_eta']} | {row['location']} | "+' | '.join(comparisons)+' |')
    peak_sets = [{r['peak_eta'] for r in brackets if r['dataset']==dataset and r['rate']=='all'} for dataset in DATASETS]
    lines += ['', '## Cross-dataset comparison', '',
        ('The datasets agree on the five-rate sampled-mean peak set.' if peak_sets[0] == peak_sets[1] else
         'The datasets do not agree on the five-rate sampled-mean peak set; do not claim one shared best region.'),
        'Peak sets: '+ '; '.join(f'{d}: {sorted(peaks)}' for d, peaks in zip(DATASETS, peak_sets))+'.', '',
        '## Old-read and new-write errors', '',
        'E_old is actual-read retention.err_decay (NO_HISTORY excluded), not err_post. '
        'E_new is observed current_observed_write_fit.err_after. Modalities remain separate. '
        'Undefined retention at rate zero is omitted, never zero-filled; no five-rate aggregate is fabricated when a rate is undefined. '
        'Run-level means are equally weighted across seeds, not weighted by token counts.', '',
        '| Dataset | Rate | Mode | Error | Modality | Mean ± SD |', '|---|---|---|---|---|---|']
    for row in summary:
        if row['section'] != 'task':
            lines.append(f"| {row['dataset']} | {row['rate']} | {row['mode']} | {row['metric']} | {row['modality']} | {row['mean']:.6g} ± {row['sample_std']:.6g} |")
    lines += ['', 'All per-seed sampled best points, every adjacent difference, and matched seed differences at each '
        'sampled-mean peak versus its neighbors are in per_seed_best.csv, per_seed_adjacent.csv, and per_seed_peak_support.csv. '
        'CSV metrics use raw units. Existing checkpoint-selection bias remains; error diagnostics alone do not establish causality.', '']
    (root / 'SUMMARY.md').write_text('\n'.join(lines), encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--grid-root', type=Path)
    parser.add_argument('--inherited-root', type=Path)
    args = parser.parse_args()
    summarize(args.root, args.grid_root, args.inherited_root)
    print(f'Wrote SUMMARY.md and seven CSV artifacts under {args.root}')
