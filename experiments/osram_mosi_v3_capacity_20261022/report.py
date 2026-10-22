"""Read-only scoring of completed finite MOSI capacity arms."""
import csv
import json
from pathlib import Path

from .worker import MODES, RATES


def summarize(output):
    output = Path(output)
    states, complete, rows, aggregate = {}, {}, [], []
    for mode in MODES:
        p = output / 'runs' / mode / 'SUMMARY.json'
        s = json.loads(p.read_text()) if p.exists() else {'status': 'not_started'}
        states[mode] = {k: s.get(k) for k in ('status', 'epoch', 'error', 'seconds', 'parameters')}
        if s.get('status') != 'complete':
            continue
        if s.get('epoch') != 150 or set(s.get('evaluation', {})) != set(RATES):
            raise ValueError('Incomplete formal result: ' + mode)
        complete[mode] = s
        for rate in RATES:
            rows.append(dict(model=mode, rate=rate, epoch=s['best_by_rate'][rate]['epoch'],
                             **s['evaluation'][rate]['router']))
        for scope, rates in (('mean', RATES), ('high', RATES[5:])):
            aggregate.append(dict(model=mode, scope=scope,
                **{key: sum(s['evaluation'][r]['router'][key] for r in rates) / len(rates)
                   for key in ('ACC', 'WF1', 'MSE')}))
    if complete:
        first = next(iter(complete.values()))['identity']
        for s in complete.values():
            for key in ('source_commit', 'source_hashes', 'mask_hashes', 'split', 'seed_protocol', 'data_files'):
                if s['identity'][key] != first[key]:
                    raise ValueError('Completed capacity arms differ: ' + key)
    result = dict(label='INTERNAL TEST-ORACLE SCREEN', states=states,
                  per_rate=rows, aggregate=aggregate)
    (output / 'REPORT.json').write_text(json.dumps(result, indent=2) + '\n')
    if rows:
        with (output / 'PER_RATE.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    lines = ['# MOSI small Flat / larger V3', '', 'INTERNAL TEST-ORACLE SCREEN', '',
             'Seed66,150epochs,3e-4(1..100)/3e-5(101..150),new GCNet missing.',
             'Only completed arms enter performance tables. Missing runs are not zeros.', '',
             '| Model | Scope | ACC | W-F1 | MSE |', '|---|---|---:|---:|---:|']
    for row in aggregate:
        lines.append(f"| {row['model']} | {row['scope']} | {row['ACC']:.3f} | {row['WF1']:.3f} | {row['MSE']:.6f} |")
    lines += ['', '## State', ''] + [f'- {name}: {s}' for name, s in states.items()]
    lines += ['', '## Per rate', '', '| Model | Rate | Epoch | ACC | W-F1 | MSE |',
              '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['model']} | {row['rate']} | {row['epoch']} | {row['ACC']:.3f} | {row['WF1']:.3f} | {row['MSE']:.6f} |")
    lines += ['', 'Compare each V3 arm with its same-width Flat control. This is capacity',
              'screening, not a parameter-matched proof or independent test generalization.',
              'Per-rate Test-best can select different epochs. No additional inference.']
    (output / 'RESULT.md').write_text('\n'.join(lines) + '\n')
    return result
