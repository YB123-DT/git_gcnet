#!/usr/bin/env python3
"""Summarize the frozen eta grid without copying raw or inherited artifacts."""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import statistics

SEEDS = tuple(range(66, 71))
RATES = (0., .1, .3, .5, .7)
MODES = ('reference', 'fixed0.95', 'fixed0.9', 'fixed0.8')
DATASETS = ('iemocap4', 'mosi')


def finite(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('nonfinite metric')
    return value


def read_metadata(path, seed):
    if not path.exists():
        raise ValueError(f'incomplete: missing {path}')
    data = json.loads(path.read_text())
    if data.get('seed') != seed or data.get('config', {}).get('seed') != seed:
        raise ValueError(f'seed/config mismatch: {path}')
    for flag in ('weights_unchanged', 'all_mode_masks_equal'):
        if data.get(flag) is not True:
            raise ValueError(f'{flag} must be true: {path}')
    for field in ('checkpoint', 'checkpoint_sha256'):
        if not data.get(field):
            raise ValueError(f'missing {field}: {path}')
    records = {}
    for result in data['results']:
        key = (float(result['rate']), result['mode'])
        if key in records:
            raise ValueError(f'duplicate result: {path}: {key}')
        for field in ('checkpoint', 'checkpoint_sha256'):
            if field in result and result[field] != data[field]:
                raise ValueError(f'{field} mismatch: {path}')
        records[key] = result
    return data, records


def load_grid(root, inherited_root):
    grid, provenance = {}, []
    for dataset in DATASETS:
        for seed in SEEDS:
            path = root / dataset / f'seed{seed}' / 'metadata.json'
            data, records = read_metadata(path, seed)
            wanted = MODES if dataset == 'mosi' else ('fixed0.95', 'fixed0.8')
            if set(records) != set(itertools.product(RATES, wanted)):
                raise ValueError(f'incomplete or unexpected new rate/mode grid: {path}')
            sources = {key: (path, False) for key in records}
            if dataset == 'iemocap4':
                old_path = inherited_root / f'iemocap4_seed{seed}' / 'metadata.json'
                old, old_records = read_metadata(old_path, seed)
                for field in ('checkpoint', 'checkpoint_sha256', 'config', 'dataset', 'epoch', 'selection_protocol'):
                    if data.get(field) != old.get(field):
                        raise ValueError(f'inherited {field} mismatch: seed {seed}')
                for key in itertools.product(RATES, ('reference', 'fixed0.9')):
                    if key not in old_records:
                        raise ValueError(f'incomplete inherited grid: {old_path}: {key}')
                    records[key] = old_records[key]
                    sources[key] = old_path, True
            for rate in RATES:
                hashes = {records[rate, mode].get('mask_sha256') for mode in MODES}
                if len(hashes) != 1 or None in hashes or '' in hashes:
                    raise ValueError(f'mask mismatch/missing: {dataset} seed {seed} rate {rate}')
                for mode in MODES:
                    key = rate, mode
                    grid[dataset, seed, rate, mode] = records[key]
                    source, inherited = sources[key]
                    provenance.append(dict(dataset=dataset, seed=seed, rate=f'{rate:g}', mode=mode,
                        inherited=inherited, source_metadata=str(source.resolve()),
                        checkpoint=data['checkpoint'], checkpoint_sha256=data['checkpoint_sha256'],
                        mask_sha256=records[key]['mask_sha256']))
    return grid, provenance


def flatten(result):
    values = {('task', '', metric): finite(value)
              for metric, value in result['task_metrics'].items()}
    if ('task', '', 'weighted_f1') not in values:
        raise ValueError('weighted_f1 is required')
    for section, metric in (('retention', 'err_decay'), ('current_observed_write_fit', 'err_after')):
        for modality, group in result.get(section, {}).items():
            stat = group.get('metrics', {}).get(metric)
            if stat and stat.get('count', 0) > 0:
                if section == 'retention' and stat['count'] != group.get('counts', {}).get('retention'):
                    raise ValueError('retention count must exclude NO_HISTORY')
                values[section, modality, metric] = finite(stat['mean'])
    return values


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def summarize(root, inherited_root=None):
    root = Path(root)
    inherited_root = Path(inherited_root) if inherited_root else Path(__file__).resolve().parent.parent / 'osram_write_intervention_20260909/full5'
    grid, provenance = load_grid(root, inherited_root)
    samples, per_seed, shape = {}, [], []
    for dataset in DATASETS:
        for seed in SEEDS:
            flat = {(rate, mode): flatten(grid[dataset, seed, rate, mode])
                    for rate, mode in itertools.product(RATES, MODES)}
            task_keys = {k for k in flat[0., 'reference'] if k[0] == 'task'}
            for values in flat.values():
                if {k for k in values if k[0] == 'task'} != task_keys:
                    raise ValueError('task metric mismatch')
            for mode in MODES:
                common = set.intersection(*(set(flat[rate, mode]) for rate in RATES))
                flat['all', mode] = {k: statistics.mean(flat[rate, mode][k] for rate in RATES) for k in common}
            for (rate, mode), values in flat.items():
                rate_label = 'all' if rate == 'all' else f'{rate:g}'
                for (section, modality, metric), value in sorted(values.items()):
                    key = dataset, rate_label, mode, section, modality, metric
                    samples.setdefault(key, []).append(value)
                    per_seed.append(dict(dataset=dataset, seed=seed, rate=rate_label, mode=mode,
                                         section=section, modality=modality, metric=metric, value=value))
            for rate in (*RATES, 'all'):
                points = [flat[rate, mode]['task', '', 'weighted_f1'] for mode in MODES]
                shape.append(dict(dataset=dataset, seed=seed, rate='all' if rate == 'all' else f'{rate:g}',
                    eta1=points[0], eta095=points[1], eta090=points[2], eta080=points[3],
                    delta_1_to_095=points[1]-points[0], delta_095_to_090=points[2]-points[1],
                    delta_090_to_080=points[3]-points[2], delta_1_to_090=points[2]-points[0]))
    summary = []
    for key, values in sorted(samples.items()):
        if len(values) != len(SEEDS):
            raise ValueError(f'incomplete metric across seeds: {key}')
        summary.append(dict(zip(('dataset', 'rate', 'mode', 'section', 'modality', 'metric'), key),
                            n_seeds=len(values), mean=statistics.mean(values), sample_std=statistics.stdev(values)))
    gate = {'gate_passed': True, 'datasets': {}, 'action': 'report only; no training or tuning'}
    for dataset in DATASETS:
        rows = [r for r in shape if r['dataset']==dataset and r['rate']=='all']
        gain = statistics.mean(r['delta_1_to_090'] for r in rows)
        positive = sum(r['delta_1_to_090'] > 0 for r in rows)
        downturn = statistics.mean(r['delta_090_to_080'] for r in rows)
        passed = gain > 0 and positive >= 3 and downturn < 0
        gate['datasets'][dataset] = dict(mean_090_minus_1=gain, positive_seeds=positive,
                                       mean_080_minus_090=downturn, passed=passed)
        gate['gate_passed'] &= passed
    write_csv(root / 'per_seed_metrics.csv', per_seed)
    write_csv(root / 'summary.csv', summary)
    write_csv(root / 'per_seed_shape.csv', shape)
    write_csv(root / 'provenance.csv', provenance)
    (root / 'gate.json').write_text(json.dumps(gate, indent=2)+'\n')
    lines = ['# Frozen write-step grid', '',
        'INTERNAL DIAGNOSTIC ONLY. Seeds 66–70; rates 0, 0.1, 0.3, 0.5, 0.7; eta 1, 0.95, 0.90, 0.80.', '',
        'Weighted F1 is shown in percent; increments are percentage points. Mean ± sample SD uses five equally weighted seeds. '
        '`all` first averages the five rates within each seed. No rates/heads are treated as independent replicates.', '',
        'IEMOCAP4 eta 1/0.90 are inherited in place. Checkpoint identity, config, epoch/selection metadata, '
        'weight-unchanged flags and same-seed/per-rate masks across all four modes passed metadata validation. '
        'This is not an independent checkpoint-file rehash. Exact source paths are in provenance.csv; no inherited raw artifacts were duplicated.', '']
    for dataset in DATASETS:
        lines += [f'## {dataset}: weighted F1 (%)', '', '| Rate | eta 1 | eta .95 | eta .90 | eta .80 |', '|---|---|---|---|---|']
        for rate in ('0', '0.1', '0.3', '0.5', '0.7', 'all'):
            selected = {r['mode']: r for r in summary if r['dataset']==dataset and r['rate']==rate and r['section']=='task' and r['metric']=='weighted_f1'}
            lines.append('| '+rate+' | '+' | '.join(f"{100*selected[m]['mean']:.3f} ± {100*selected[m]['sample_std']:.3f}" for m in MODES)+' |')
        best_value = max(selected[m]['mean'] for m in MODES)
        best = [m for m in MODES if selected[m]['mean']==best_value]
        lines += ['', f'Descriptive best grid point(s), five-rate mean: {", ".join(best)}. This is not an optimality claim.', '',
                  '| Seed | eta 1 | eta .95 | eta .90 | eta .80 | 1→.95 | .95→.90 | .90→.80 |',
                  '|---|---|---|---|---|---|---|---|']
        for row in shape:
            if row['dataset']==dataset and row['rate']=='all':
                lines.append('| '+str(row['seed'])+' | '+' | '.join(f"{100*row[k]:.3f}" for k in ('eta1','eta095','eta090','eta080','delta_1_to_095','delta_095_to_090','delta_090_to_080'))+' |')
    lines += ['', '## Old-read and new-write errors', '',
        'Old error is the actual-read `retention.err_decay` (NO_HISTORY excluded), not err_post. '
        'New error is `current_observed_write_fit.err_after`. Each evaluator run-level mean is one observation; '
        'five seeds are equally weighted and modalities remain separate. Undefined rate-zero retention is omitted, never zero-filled. '
        'No five-rate old-error aggregate is fabricated when rate zero is undefined.', '',
        '| Dataset | Rate | Mode | Error | Modality | Mean ± SD |', '|---|---|---|---|---|---|']
    for row in summary:
        if row['section'] != 'task':
            lines.append(f"| {row['dataset']} | {row['rate']} | {row['mode']} | {row['metric']} | {row['modality']} | {row['mean']:.6g} ± {row['sample_std']:.6g} |")
    lines += ['', '## Frozen operational retrain gate', '',
        'Rule: on BOTH datasets, eta .90 has greater five-rate mean Wf1 than eta 1 and positive paired gains in ≥3/5 seeds; '
        'eta .80 has lower mean Wf1 than eta .90 on BOTH datasets. Strict comparisons; no significance or extra effect-size threshold.', '',
        '| Dataset | .90−1 (pp) | Positive seeds | .80−.90 (pp) | Pass |', '|---|---|---|---|---|']
    for dataset, row in gate['datasets'].items():
        lines.append(f"| {dataset} | {100*row['mean_090_minus_1']:.3f} | {row['positive_seeds']}/5 | {100*row['mean_080_minus_090']:.3f} | {row['passed']} |")
    lines += ['', f"Gate passed: **{gate['gate_passed']}**. Report only: no training, checkpoint selection, or tuning performed.", '',
        'Per-rate/per-seed points and adjacent increments are in per_seed_shape.csv; all task metrics and '
        'selected error diagnostics are in per_seed_metrics.csv and summary.csv (raw units). '
        'Five paired seeds provide descriptive evidence only. Existing checkpoint-selection bias remains; '
        'error diagnostics alone do not establish downstream causality.', '']
    (root / 'SUMMARY.md').write_text('\n'.join(lines), encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--inherited-root', type=Path)
    args = parser.parse_args()
    summarize(args.root, args.inherited_root)
    print(f'Wrote SUMMARY.md and five CSV/JSON artifacts under {args.root}')
