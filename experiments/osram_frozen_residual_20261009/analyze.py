"""Paired fixed-last-epoch residual evaluation; no test epoch selection."""
import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

GROUPS = ('A_local', 'A_donor', 'A_real', 'B_local_history', 'B_donor_history',
          'B_real_history', 'B_gold_history')
SEEDS = (66, 67, 68)
RATES = tuple(i / 10 for i in range(8))
METRICS = ('weighted_f1_nonzero', 'acc_nonzero', 'mse', 'n', 'nonneutral_n',
           'corrections', 'harms')
PAIRS = tuple((model, ref) for model, refs in (
    ('A_real', ('Original', 'A_local', 'A_donor')),
    ('B_real_history', ('Original', 'B_local_history', 'B_donor_history')))
    for ref in refs)


def load_inputs(paths):
    runs, protocols = [], []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            path = path / 'SUMMARY.json'
        payload = json.loads(path.read_text())
        runs.extend(payload['runs'])
        protocols.append({'source': str(path), 'protocol': payload.get('protocol', {})})
    return runs, protocols


def summarize(runs):
    indexed, baseline = {}, {}
    for run in runs:
        seed, rate, group = int(run['seed']), round(float(run['rate']), 7), run['group']
        key = seed, rate, group
        if seed not in SEEDS or rate not in RATES or group not in GROUPS:
            raise ValueError(f'Unexpected run key {key}')
        if key in indexed:
            raise ValueError(f'Duplicate run {key}')
        if run['selection'] != 'fixed_last_epoch' or int(run['epoch']) != 100:
            raise ValueError(f'Not fixed last epoch 100: {key}')
        for split in ('train', 'test', 'baseline_train', 'baseline_test'):
            for metric in METRICS:
                if metric not in run[split] or not math.isfinite(float(run[split][metric])):
                    raise ValueError(f'Invalid {split}/{metric} for {key}')
            if any(not 0 <= float(run[split][metric]) <= 1 for metric in ('weighted_f1_nonzero', 'acc_nonzero')):
                raise ValueError(f'Expected fractional metrics for {split} in {key}')
        for split in ('train', 'test'):
            bkey = rate, split
            metrics = run['baseline_' + split]
            if bkey in baseline and any(not math.isclose(float(metrics[m]), float(baseline[bkey][m]),
                                                       rel_tol=1e-8, abs_tol=1e-8) for m in METRICS):
                raise ValueError(f'Inconsistent matched baseline {bkey}')
            baseline[bkey] = metrics
        indexed[key] = run
    missing = [{'seed': seed, 'rate': rate, 'group': group}
               for seed in SEEDS for rate in RATES for group in GROUPS
               if (seed, rate, group) not in indexed]
    per_rate, lookup = [], {}
    for (seed, rate, group), run in sorted(indexed.items()):
        for name, values in ((group, run['test']), ('Original', run['baseline_test'])):
            key = seed, rate, name
            if key in lookup:
                continue
            row = dict(seed=seed, rate=rate, group=name, **values)
            row['parameter_count'] = 0 if name == 'Original' else run['parameter_count']
            lookup[key] = row
            per_rate.append(row)
    per_seed = []
    for seed in SEEDS:
        for group in ('Original',) + GROUPS:
            rows = [lookup[seed, rate, group] for rate in RATES if (seed, rate, group) in lookup]
            if not rows:
                continue
            row = dict(seed=seed, group=group, rate_count=len(rows))
            for name, rates in (('mean8', RATES), ('high', RATES[5:])):
                selected = [lookup[seed, rate, group] for rate in rates if (seed, rate, group) in lookup]
                full = len(selected) == len(rates)
                for short, metric in (('wf1', 'weighted_f1_nonzero'), ('acc', 'acc_nonzero'), ('mse', 'mse')):
                    row[f'{name}_{short}'] = statistics.mean(r[metric] for r in selected) if full else None
                for metric in ('corrections', 'harms'):
                    row[f'{name}_{metric}_sum'] = sum(r[metric] for r in selected) if full else None
            per_seed.append(row)
    aggregate = []
    for group in ('Original',) + GROUPS:
        rows = [r for r in per_seed if r['group'] == group]
        if not rows:
            continue
        row = {'group': group}
        for scope in ('mean8', 'high'):
            for metric in ('wf1', 'acc', 'mse'):
                name = f'{scope}_{metric}'
                values = [r[name] for r in rows if r[name] is not None]
                row[name + '_n_seeds'] = len(values)
                row[name + '_mean'] = statistics.mean(values) if values else None
                row[name + '_std'] = statistics.stdev(values) if len(values) > 1 else None
        aggregate.append(row)
    comparisons = []
    for seed in SEEDS:
        for model, reference in PAIRS:
            for scope, rates in [(f'rate_{rate:.1f}', (rate,)) for rate in RATES] + [('mean8', RATES), ('high', RATES[5:])]:
                if not all((seed, rate, name) in lookup for rate in rates for name in (model, reference)):
                    continue
                row = dict(seed=seed, model=model, reference=reference, scope=scope, rate_count=len(rates))
                for short, metric in (('wf1_delta_pp', 'weighted_f1_nonzero'), ('acc_delta_pp', 'acc_nonzero'), ('mse_delta', 'mse')):
                    row[short] = statistics.mean(lookup[seed, rate, model][metric] - lookup[seed, rate, reference][metric] for rate in rates)
                    if short.endswith('_pp'):
                        row[short] *= 100
                comparisons.append(row)
    return dict(complete=not missing, run_count=len(indexed), expected_run_count=168, missing=missing,
                per_rate=per_rate, per_seed=per_seed, aggregate=aggregate, comparisons=comparisons)


def write_report(result, output, protocols=()):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    payload = dict(result, protocols=list(protocols))
    (output / 'SUMMARY.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    for name in ('per_rate', 'per_seed', 'comparisons'):
        rows = result[name]
        if rows:
            with (output / f'{name}.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    def display(mean, std):
        if mean is None:
            return '—'
        mean *= 100
        std = std * 100 if std is not None else None
        return f'{mean:.3f} ± {std:.3f}' if std is not None else f'{mean:.3f} (n=1)'
    lines = ['# Frozen Memory Residual Learning', '', '**INTERNAL DIAGNOSTIC ONLY**', '',
             f"Status: {'COMPLETE' if result['complete'] else 'PARTIAL — not a completed comparison'}; "
             f"{result['run_count']}/{result['expected_run_count']} runs.", '',
             'Original backbone and historical probes inherit test-selected checkpoint provenance. '
             'Residual heads use fixed last epoch 100, with no test epoch selection. '
             'The backbone remains frozen. The three seeds are residual-head initializations, '
             'not independent backbone seeds. Gold-history uses unavailable-at-inference true historical labels '
             'and is an offline diagnostic only; it is not a guaranteed mathematical upper bound.', '',
             'Metrics below average eight rates equally within each residual-head seed, then report '
             'mean ± sample standard deviation across seeds. High missing averages .5/.6/.7. '
             'No predictions are pooled across rates. Incomplete eight-rate results are not called eight-rate means.', '',
             '| Group | 8-rate W-F1 (%) | High W-F1 (%) | Complete 8-rate seeds |',
             '|---|---:|---:|---:|']
    for row in result['aggregate']:
        lines.append(f"| {row['group']} | {display(row['mean8_wf1_mean'], row['mean8_wf1_std'])} | "
                     f"{display(row['high_wf1_mean'], row['high_wf1_std'])} | {row['mean8_wf1_n_seeds']} |")
    lines += ['', '## Per-rate W-F1 (%)', '', '| Group | ' + ' | '.join(f'{r:.1f}' for r in RATES) + ' |',
              '|---|' + '---:|' * len(RATES)]
    for group in ('Original',) + GROUPS:
        values = []
        for rate in RATES:
            scores = [r['weighted_f1_nonzero'] for r in result['per_rate'] if r['group'] == group and r['rate'] == rate]
            values.append(display(statistics.mean(scores) if scores else None, statistics.stdev(scores) if len(scores)>1 else None))
        lines.append('| ' + group + ' | ' + ' | '.join(values) + ' |')
    lines += ['', '## Paired gains (percentage points)', '',
              '| Model − reference | Seed | 8-rate ΔW-F1 | High ΔW-F1 |', '|---|---:|---:|---:|']
    for model, ref in PAIRS:
        for seed in SEEDS:
            rows = {r['scope']: r for r in result['comparisons'] if r['model']==model and r['reference']==ref and r['seed']==seed}
            vals = [f"{rows[s]['wf1_delta_pp']:+.3f}" if s in rows else '—' for s in ('mean8', 'high')]
            lines.append(f'| {model} − {ref} | {seed} | {vals[0]} | {vals[1]} |')
    lines += ['', '## Interpretation boundaries', '',
              'Real Memory must beat Original, Local Control, and Donor Control in paired comparisons. '
              'A gain supports exploitable residual information only relative to this frozen backbone and corrector class. '
              'Historical-scalar gains do not establish pure semantic mediation. '
              'One backbone cannot establish multi-backbone stability or a general mechanism.', '',
              'Corrections/harms are relative to Original on nonneutral labels. The sums in per_seed.csv '
              'repeat utterances across missing rates and are not counts of unique utterances. '
              'No significance test treats rates as independent observations.', '',
              'Detailed files: per_rate.csv, per_seed.csv, comparisons.csv, SUMMARY.json. '
              'Accuracy and W-F1 in JSON/CSV are fractions except explicitly named _pp differences; report tables are percentages. '
              'Provenance and effective source protocols are retained in SUMMARY.json.', '']
    (output / 'RESULT.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs='+', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--allow-partial', action='store_true', help='Allow successful exit with incomplete coverage')
    args = parser.parse_args()
    runs, protocols = load_inputs(args.inputs)
    result = summarize(runs)
    write_report(result, args.output, protocols)
    print(json.dumps({'complete': result['complete'], 'run_count': result['run_count'], 'missing': len(result['missing'])}))
    if not result['complete'] and not args.allow_partial:
        sys.exit(2)


if __name__ == '__main__':
    main()
