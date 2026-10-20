"""Complete-only report for 24 paired seed/rate frozen-readout audits."""

import csv
import json
from pathlib import Path
import statistics

from .analysis import ROLES, SOURCES


SEEDS = (66, 67, 68)
RATES = tuple(f'{i / 10:.1f}' for i in range(8))


def _mean(values):
    values = [value for value in values if value is not None]
    return statistics.mean(values) if values else None


def _distribution(values):
    values = [value for value in values if value is not None]
    return {'mean': _mean(values), 'std': statistics.stdev(values) if len(values) > 1 else None}


def _number(value, signed=False):
    return 'N/A' if value is None else format(value, '+.3f' if signed else '.3f')


def _write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_run(output, summary):
    """Require all three seeds/eight selected BESTs; never fill incomplete cells."""
    output = Path(output)
    records = []
    conditions = summary['conditions']
    entries = conditions.values() if isinstance(conditions, dict) else conditions
    for entry in entries:
        path = Path(entry['path'])
        if not path.is_absolute():
            path = output / path
        record = json.loads(path.read_text())
        seed, rate = int(record['seed']), f"{float(record['rate']):.1f}"
        if seed != int(entry['seed']) or rate != f"{float(entry['rate']):.1f}":
            raise ValueError('Condition metadata differs from summary manifest')
        if 'statistics' not in record or len(record['statistics']['slots']) != 12:
            raise ValueError('Incomplete condition statistics')
        records.append(dict(record, seed=seed, rate=rate, path=str(path)))
    expected = {(seed, rate) for seed in SEEDS for rate in RATES}
    keys = [(record['seed'], record['rate']) for record in records]
    if len(keys) != 24 or set(keys) != expected:
        raise ValueError('Require exactly 24 unique conditions: seeds66/67/68 x rates0.0..0.7')
    records.sort(key=lambda record: (record['seed'], record['rate']))
    per_seed = {}
    for seed in SEEDS:
        per_seed[str(seed)] = {}
        for scope, rates in (('mean', RATES), ('high', RATES[5:])):
            cohort = [r['statistics'] for r in records if r['seed'] == seed and r['rate'] in rates]
            per_seed[str(seed)][scope] = {
                'models': {model: {metric: _mean(r['models'][model][metric] for r in cohort)
                                   for metric in ('ACC', 'WF1', 'MSE')}
                           for model in ('flat', 'neutral', 'learned')},
                'comparisons': {comparison: {metric: _mean(r['comparisons'][comparison][metric] for r in cohort)
                                            for metric in ('ACC_pp', 'WF1_pp', 'MSE')}
                                for comparison in ('total', 'mainpath', 'direct')}}
    aggregate = {}
    for scope in ('mean', 'high'):
        aggregate[scope] = {
            'models': {model: {metric: _distribution(per_seed[str(seed)][scope]['models'][model][metric]
                                                    for seed in SEEDS)
                               for metric in ('ACC', 'WF1', 'MSE')}
                       for model in ('flat', 'neutral', 'learned')},
            'comparisons': {comparison: {metric: _distribution(per_seed[str(seed)][scope]['comparisons'][comparison][metric]
                                                              for seed in SEEDS)
                                        for metric in ('ACC_pp', 'WF1_pp', 'MSE')}
                            for comparison in ('total', 'mainpath', 'direct')}}
    slots = {}
    for role in ROLES:
        for source in SOURCES:
            name = f'{role}/{source}'
            seed_gains = {str(seed): _mean(r['statistics']['slots'][name]['full_test']['gain_WF1_pp']
                                          for r in records if r['seed'] == seed) for seed in SEEDS}
            values = list(seed_gains.values())
            signs = [('positive' if value > 1e-9 else 'negative' if value < -1e-9 else 'zero')
                     for value in values]
            slots[name] = {'per_seed_eight_rate_full_test_gain_WF1_pp': seed_gains,
                           'across_seed': _distribution(values), 'directions': dict(zip(map(str, SEEDS), signs)),
                           'mixed_nonzero_directions': 'positive' in signs and 'negative' in signs}
    report = {'label': 'INTERNAL DIAGNOSTIC ONLY', 'status': 'complete',
              'seeds': list(SEEDS), 'rates': list(RATES), 'conditions': records,
              'per_seed': per_seed, 'aggregate': aggregate, 'slots': slots,
              'aggregation': 'Equal rate mean within each seed; then mean and sample std (ddof=1) across seeds.',
              'limitation': 'Exploratory per-rate Test-best checkpoints, not one fixed checkpoint or independent-test evidence.'}
    (output / 'REPORT.json').write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    metric_rows, slot_rows = [], []
    for record in records:
        for model, values in record['statistics']['models'].items():
            metric_rows.append(dict(seed=record['seed'], rate=record['rate'], model=model, **values))
        for name, slot in record['statistics']['slots'].items():
            role, source = name.split('/')
            slot_rows.append(dict(seed=record['seed'], rate=record['rate'], role=role, source=source,
                                  active_N=slot['active_cohort']['N'], active_polarity_N=slot['active_cohort']['N_polarity'],
                                  full_test_gain_WF1_pp=slot['full_test']['gain_WF1_pp'],
                                  active_gain_WF1_pp=slot['active_cohort']['gain_WF1_pp'],
                                  active_gain_ACC_pp=slot['active_cohort']['gain_ACC_pp'],
                                  active_delta_MSE=slot['active_cohort']['delta_MSE'],
                                  corrections=slot['active_cohort']['corrections'], harms=slot['active_cohort']['harms'],
                                  gamma_mean=slot['gamma']['mean'], gamma_std=slot['gamma']['std'],
                                  gamma_below_one_fraction=slot['gamma']['below_one_fraction'],
                                  gamma_above_one_fraction=slot['gamma']['above_one_fraction']))
    _write_csv(output / 'METRICS.csv', metric_rows)
    _write_csv(output / 'SLOTS.csv', slot_rows)
    lines = ['# MOSI 12-slot learned-weight neutralization audit', '', 'INTERNAL DIAGNOSTIC ONLY', '',
             'Frozen inference only. Each condition uses its own already-selected BEST checkpoint and exact mask.',
             'Neutralization is gamma=1, NOT gamma=0 source removal. Local and Memory trajectory stay fixed.',
             'Positive learned-minus-reset W-F1 means that learned slot calibration helps relative to retaining the original source contribution.',
             'The primary cells below are equal eight-rate mean FULL-TEST W-F1 changes (percentage points), not active-subset F1.', '']
    for seed in SEEDS:
        lines += [f'## Seed {seed}: 4 x 3 full-test contribution table', '',
                  '| Read role | Historical A | Historical T | Historical V |',
                  '|---|---:|---:|---:|']
        for role in ROLES:
            gains = [_number(slots[f'{role}/{source}']['per_seed_eight_rate_full_test_gain_WF1_pp'][str(seed)], True)
                     for source in SOURCES]
            lines.append(f'| {role} | ' + ' | '.join(gains) + ' |')
        lines.append('')
    lines += ['## Complete-path comparison', '',
              '| Seed | Scope | Original Flat W-F1 | All gamma=1 W-F1 | Learned W-F1 | Total pp | Main-path pp | Direct pp |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for seed in SEEDS:
        for scope in ('mean', 'high'):
            row = per_seed[str(seed)][scope]
            values = [_number(row['models'][model]['WF1']) for model in ('flat', 'neutral', 'learned')]
            values += [_number(row['comparisons'][name]['WF1_pp'], True) for name in ('total', 'mainpath', 'direct')]
            lines.append(f'| {seed} | {scope} | ' + ' | '.join(values) + ' |')
    lines += ['', '| Across seeds | Flat W-F1 | Neutral W-F1 | Learned W-F1 | Total pp | Direct pp |',
              '|---|---:|---:|---:|---:|---:|']
    for scope in ('mean', 'high'):
        row = aggregate[scope]
        values = [f"{_number(row['models'][model]['WF1']['mean'])} ± {_number(row['models'][model]['WF1']['std'])}"
                  for model in ('flat', 'neutral', 'learned')]
        values += [_number(row['comparisons'][name]['WF1_pp']['mean'], True) for name in ('total', 'direct')]
        lines.append(f'| {scope} | ' + ' | '.join(values) + ' |')
    lines += ['', 'Total=learned−original Flat; main-path=all gamma1−original Flat; direct=learned−all gamma1.',
              'The differences telescope algebraically, but main-path change is not a causal attribution of training dynamics.',
              'High is rates0.5/0.6/0.7. Means are computed within seed first; ± is sample standard deviation across three seeds.', '',
              '## Cross-seed directions (descriptive only)', '',
              '| Slot | Seed66 | Seed67 | Seed68 | Mixed nonzero signs? |', '|---|---|---|---|---|']
    for name, slot in slots.items():
        lines.append('| ' + name + ' | ' + ' | '.join(slot['directions'][str(seed)] for seed in SEEDS)
                     + f" | {slot['mixed_nonzero_directions']} |")
    lines += ['', 'No statistical-significance claim is made. Do not sum the 12 independent nonlinear intervention effects.',
              'Inactive slots have full-test gain0 and active-only N/A, not evidence that a present source is useless.',
              'Corrections/harms count reset→learned polarity changes on the identical nonzero-label active cohort.',
              'N includes neutral labels for MSE; N_polarity excludes label0 and uses prediction>0.',
              'Repeated observations across rates/seeds are not independent utterances; counts must not be interpreted as unique corpus totals.',
              'All per-rate active-cohort results, counts and gamma statistics are in PER_RATE.md and SLOTS.csv.',
              'Gamma magnitude alone is not reliability, semantic conflict, or causal source importance.']
    (output / 'RESULT.md').write_text('\n'.join(lines) + '\n')
    detail = ['# Per-rate active-cohort details', '', 'INTERNAL DIAGNOSTIC ONLY', '',
              'Full-test gain is learned−single-slot gamma1 reset. Active gain recomputes F1 on the same active cohort.',
              'Inactive active-cohort cells are N/A. Corrections/harms are reset→learned and exclude neutral labels.', '']
    for record in records:
        detail += [f"## Seed {record['seed']}, missing rate {record['rate']}", '',
                   '| Prediction | N | Nonzero labels | ACC | W-F1 | MSE |',
                   '|---|---:|---:|---:|---:|---:|']
        for name, values in record['statistics']['models'].items():
            detail.append(f"| {name} | {values['N']} | {values['N_polarity']} | {_number(values['ACC'])} | {_number(values['WF1'])} | {_number(values['MSE'])} |")
        detail += ['', '| Comparison | ΔW-F1 pp | ΔACC pp | ΔMSE | Corrections | Harms |',
                   '|---|---:|---:|---:|---:|---:|']
        for name, values in record['statistics']['comparisons'].items():
            detail.append(f"| {name} | {_number(values['WF1_pp'], True)} | {_number(values['ACC_pp'], True)} | {_number(values['MSE'], True)} | {values['corrections']} | {values['harms']} |")
        detail += ['',
                   '| Slot | Active N | Polarity N | Full-test ΔW-F1 pp | Active ΔW-F1 pp | Corrections | Harms | Active mean gamma |',
                   '|---|---:|---:|---:|---:|---:|---:|---:|']
        for name, slot in record['statistics']['slots'].items():
            active = slot['active_cohort']
            detail.append(f"| {name} | {active['N']} | {active['N_polarity']} | {_number(slot['full_test']['gain_WF1_pp'], True)} | "
                          f"{_number(active['gain_WF1_pp'], True)} | {active['corrections']} | {active['harms']} | {_number(slot['gamma']['mean'])} |")
        detail.append('')
    (output / 'PER_RATE.md').write_text('\n'.join(detail) + '\n')
    return report
