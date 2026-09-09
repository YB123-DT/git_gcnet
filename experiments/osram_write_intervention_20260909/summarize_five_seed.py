#!/usr/bin/env python3
"""Summarize frozen four-mode evaluations; stdlib only, no model imports.

Usage: python summarize_five_seed.py [full5-directory]
"""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import statistics

SEEDS = tuple(range(66, 71))
RATES = (0., .1, .3, .5, .7)
MODES = ('reference', 'protected', 'global', 'fixed0.9')
PAIRED = 'global-minus-fixed0.9'


def finite(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('nonfinite metric')
    return value


def load_runs(root):
    runs = {}
    for seed in SEEDS:
        path = root / f'iemocap4_seed{seed}' / 'metadata.json'
        if not path.exists():
            raise ValueError(f'incomplete: missing {path}')
        data = json.loads(path.read_text())
        if data.get('seed') != seed or data.get('config', {}).get('seed') != seed:
            raise ValueError(f'seed/config mismatch: {path}')
        for flag in ('weights_unchanged', 'all_mode_masks_equal'):
            if data.get(flag) is not True:
                raise ValueError(f'{flag} must be true: {path}')
        if not data.get('checkpoint') or not data.get('checkpoint_sha256'):
            raise ValueError(f'missing checkpoint identity: {path}')
        records = {}
        for result in data['results']:
            key = (float(result['rate']), result['mode'])
            if key in records:
                raise ValueError(f'duplicate result {seed}: {key}')
            for field in ('checkpoint', 'checkpoint_sha256'):
                if field in result and result[field] != data[field]:
                    raise ValueError(f'checkpoint mismatch: {path}, {key}')
            records[key] = result
        if set(records) != set(itertools.product(RATES, MODES)):
            raise ValueError(f'incomplete or unexpected rate/mode grid: {path}')
        for rate in RATES:
            hashes = {records[rate, mode].get('mask_sha256') for mode in MODES}
            if len(hashes) != 1 or None in hashes or '' in hashes:
                raise ValueError(f'mask mismatch/missing: seed {seed}, rate {rate}')
        data['indexed_results'] = records
        runs[seed] = data
    return runs


def flatten(result):
    """Each value is one run-level mean, never an independent head sample."""
    values = {('task', '', metric): finite(value)
              for metric, value in result['task_metrics'].items()}
    for section, source in (('retention', 'retention'),
                            ('write_fit', 'current_observed_write_fit')):
        for modality, group in result.get(source, {}).items():
            for metric, stats in group.get('metrics', {}).items():
                if stats.get('count', 0) > 0:
                    values[section, modality, metric] = finite(stats['mean'])
    for metric, stats in result.get('write_audit', {}).get('metrics', {}).items():
        if stats.get('count', 0) > 0:
            values['write_audit', '', metric] = finite(stats['mean'])
    return values


def stats_row(key, samples):
    section, rate, mode, modality, metric = key
    values = list(samples)
    paired = mode == PAIRED
    result = dict(section=section, rate=rate, mode=mode, modality=modality,
                  metric=metric, n_seeds=len(values), mean=statistics.mean(values),
                  sample_std=statistics.stdev(values) if len(values) > 1 else '',
                  positive=sum(v > 0 for v in values) if paired else '',
                  zero=sum(v == 0 for v in values) if paired else '',
                  negative=sum(v < 0 for v in values) if paired else '',
                  sign_flip_p_two_sided='')
    if paired and rate == 'all' and section == 'task':
        observed = abs(sum(values))
        # Exact, paired two-sided sign randomization; only 32 sign assignments.
        extremes = sum(abs(sum(s * v for s, v in zip(signs, values))) >= observed - 1e-12
                       for signs in itertools.product((-1, 1), repeat=len(values)))
        result['sign_flip_p_two_sided'] = extremes / 2 ** len(values)
    return result


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def summarize(root):
    root = Path(root)
    runs = load_runs(root)
    samples = {}
    per_seed = []
    for seed, data in runs.items():
        grid = {key: flatten(value) for key, value in data['indexed_results'].items()}
        task_keys = {key for key in grid[0., 'reference'] if key[0] == 'task'}
        if ('task', '', 'weighted_f1') not in task_keys:
            raise ValueError('weighted_f1 is required')
        for rate in RATES:
            for mode in MODES:
                if {k for k in grid[rate, mode] if k[0] == 'task'} != task_keys:
                    raise ValueError('task metric mismatch')
            global_values, fixed_values = grid[rate, 'global'], grid[rate, 'fixed0.9']
            grid[rate, PAIRED] = {k: global_values[k] - fixed_values[k]
                                  for k in global_values.keys() & fixed_values.keys()}
        for mode in (*MODES, PAIRED):
            # Across-rate entries require the metric to exist at all five rates.
            common = set.intersection(*(set(grid[rate, mode]) for rate in RATES))
            grid['all', mode] = {k: statistics.mean(grid[rate, mode][k] for rate in RATES)
                                 for k in common}
        for (rate, mode), values in grid.items():
            rate_label = 'all' if rate == 'all' else f'{rate:g}'
            for (section, modality, metric), value in sorted(values.items()):
                key = section, rate_label, mode, modality, metric
                samples.setdefault(key, []).append(value)
                per_seed.append(dict(seed=seed, section=section, rate=rate_label,
                                     mode=mode, modality=modality, metric=metric, value=value))
    # A diagnostic present for fewer seeds is not silently reported as full-five.
    for key, values in samples.items():
        if len(values) != len(SEEDS):
            raise ValueError(f'incomplete metric across seeds: {key}')
    summary = [stats_row(key, values) for key, values in sorted(samples.items())]
    write_csv(root / 'per_seed_rate.csv', per_seed)
    write_csv(root / 'summary.csv', summary)
    lines = ['# Four-mode frozen-checkpoint evaluation: five seeds', '',
             'INTERNAL DIAGNOSTIC ONLY — NOT A FORMAL PAPER RESULT.', '',
             'Seeds: 66–70. Missing rates: 0, 0.1, 0.3, 0.5, 0.7. '
             'Dynamic = `global` (norm-matched); fixed comparator = `fixed0.9`.', '',
             'All 100 evaluations are present. Each seed uses the single checkpoint/config '
             'declared below for its 20 evaluations. Weight-unchanged flags and both mask '
             'flags and per-rate hashes passed validation. This checks metadata consistency, '
             'not an independent rehash of checkpoint contents.', '',
             '| Seed | Epoch | Checkpoint | SHA256 |', '|---|---|---|---|']
    for seed, data in runs.items():
        lines.append(f"| {seed} | {data.get('epoch', 'unknown')} | `{data['checkpoint']}` | `{data['checkpoint_sha256']}` |")
    lines += ['', '## Task metrics', '',
              'Raw metric units; mean ± sample SD across five seeds. `all` first averages '
              'the five rates within each seed. Positive/zero/negative counts apply to paired '
              'dynamic minus fixed differences, not unpaired pooled evaluations.', '',
              '| Rate | Mode | Metric | Mean ± SD | + / 0 / − |', '|---|---|---|---|---|']
    for row in summary:
        if row['section'] == 'task':
            counts = f"{row['positive']} / {row['zero']} / {row['negative']}" if row['mode'] == PAIRED else '—'
            lines.append(f"| {row['rate']} | {row['mode']} | {row['metric']} | "
                         f"{row['mean']:.6f} ± {row['sample_std']:.6f} | {counts} |")
    lines += ['', '## Per-seed five-rate weighted F1', '',
              '| Seed | Reference | Protected | Dynamic | Fixed 0.9 | Dynamic − fixed |',
              '|---|---|---|---|---|---|']
    for seed in SEEDS:
        lookup = {r['mode']: r['value'] for r in per_seed if r['seed'] == seed
                  and r['section'] == 'task' and r['rate'] == 'all' and r['metric'] == 'weighted_f1'}
        lines.append('| ' + str(seed) + ' | ' + ' | '.join(f'{lookup[m]:.6f}' for m in (*MODES, PAIRED)) + ' |')
    lines += ['', '## Retention and current-write fit', '',
              'Head/record observations are averaged inside each run by the evaluator; '
              'the run means are then averaged with equal seed weight. Counts are not '
              'independent replicates. Modalities remain separate. Missing retention at '
              'rate zero is omitted, not imputed as zero. Full write-audit means are in CSV.', '',
              '| Diagnostic | Rate | Mode | Modality | Metric | Mean ± SD |',
              '|---|---|---|---|---|---|']
    for row in summary:
        if row['section'] in ('retention', 'write_fit'):
            lines.append(f"| {row['section']} | {row['rate']} | {row['mode']} | {row['modality']} | "
                         f"{row['metric']} | {row['mean']:.6g} ± {row['sample_std']:.6g} |")
    primary = next(r for r in summary if r['section'] == 'task' and r['rate'] == 'all'
                   and r['mode'] == PAIRED and r['metric'] == 'weighted_f1')
    lines += ['', '## Interpretation and limitations', '',
              f"Dynamic − fixed five-rate weighted-F1 difference: {primary['mean']:.6f} "
              f"± {primary['sample_std']:.6f}; positive in {primary['positive']}/5 seeds. "
              f"Descriptive exact two-sided sign-flip p = {primary['sign_flip_p_two_sided']:.4f}.", '',
              'With five paired seeds there are only 32 sign assignments (minimum two-sided '
              'p = 0.0625); this is descriptive, not a significance claim. The test assumes '
              'sign exchangeability and does not correct multiple exploratory comparisons. '
              'Rates and heads are not independent samples. Retention and current-write fit '
              'measure different behaviors and neither alone establishes downstream causality.', '',
              'Fixed 0.9 scales all writes, including complete-input/no-history writes; '
              'complete-input identity is therefore not expected for that comparator. '
              'This is frozen-checkpoint evaluation only: no retraining, new checkpoint '
              'selection, or automatic module changes are justified by these results. '
              'Existing checkpoint-selection bias remains; see per-run selection protocol.', '']
    (root / 'RESULT_FULL5.md').write_text('\n'.join(lines), encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', type=Path, default=Path(__file__).parent / 'full5')
    args = parser.parse_args()
    summarize(args.root)
    print(f'Wrote per_seed_rate.csv, summary.csv, RESULT_FULL5.md under {args.root}')
