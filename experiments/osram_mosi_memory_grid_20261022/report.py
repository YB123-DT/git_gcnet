"""Paired, read-only scoring of the finite MOSI memory hyperparameter screen."""
import argparse
import csv
import json
import math
from numbers import Real
from pathlib import Path

from .config import FAMILIES, MODES, RATES


METRICS = ('ACC', 'WF1', 'MSE')
IDENTITY_FIELDS = ('source_commit', 'source_hashes', 'mask_hashes', 'split',
                   'seed_protocol', 'data_files', 'environment')


def validate_completed(mode, summary):
    if (summary.get('epoch') != 150
            or set(summary.get('evaluation', {})) != set(RATES)
            or set(summary.get('best_by_rate', {})) != set(RATES)):
        raise ValueError('Incomplete formal result: ' + mode)
    for key in IDENTITY_FIELDS:
        if key not in summary.get('identity', {}):
            raise ValueError(f'Missing identity {key}: {mode}')
    for rate in RATES:
        selected = summary['best_by_rate'][rate].get('epoch')
        if not isinstance(selected, int) or not 1 <= selected <= 150:
            raise ValueError(f'Invalid best epoch: {mode}/{rate}')
        scores = summary['evaluation'][rate].get('router', {})
        if any(not isinstance(scores.get(key), Real) or not math.isfinite(scores[key])
               for key in METRICS):
            raise ValueError(f'Incomplete or nonfinite metrics: {mode}/{rate}')


def delta(scores, baseline):
    return ({key: scores[key] - baseline[key] for key in METRICS}
            if baseline is not None else None)


def summarize(output):
    output = Path(output)
    states, complete = {}, {}
    for mode in MODES:
        path = output / 'runs' / mode / 'SUMMARY.json'
        summary = json.loads(path.read_text()) if path.exists() else {'status': 'not_started'}
        states[mode] = {key: summary.get(key) for key in
                        ('status', 'epoch', 'error', 'seconds', 'parameters')}
        if summary.get('status') == 'complete':
            validate_completed(mode, summary)
            complete[mode] = summary
    if complete:
        reference = next(iter(complete.values()))['identity']
        for mode, summary in complete.items():
            for key in IDENTITY_FIELDS:
                if summary['identity'][key] != reference[key]:
                    raise ValueError(f'Completed memory arms differ in {key}: {mode}')

    baseline = complete.get('baseline')
    rows, aggregate = [], []
    for mode, summary in complete.items():
        for rate in RATES:
            scores = {key: summary['evaluation'][rate]['router'][key] for key in METRICS}
            base = baseline['evaluation'][rate]['router'] if baseline else None
            rows.append(dict(model=mode, rate=rate, epoch=summary['best_by_rate'][rate]['epoch'],
                             **scores, delta=delta(scores, base)))
        for scope, rates in (('mean', RATES), ('high', RATES[5:])):
            scores = {key: sum(summary['evaluation'][r]['router'][key] for r in rates) / len(rates)
                      for key in METRICS}
            base = ({key: sum(baseline['evaluation'][r]['router'][key] for r in rates) / len(rates)
                     for key in METRICS} if baseline else None)
            aggregate.append(dict(model=mode, scope=scope, **scores, delta=delta(scores, base)))
    families = {family: dict(modes=list(modes),
                    aggregate=[row for mode in modes for row in aggregate if row['model'] == mode],
                    per_rate=[row for mode in modes for row in rows if row['model'] == mode])
                for family, modes in FAMILIES.items()}
    result = dict(label='INTERNAL TEST-ORACLE SCREEN', states=states,
                  per_rate=rows, aggregate=aggregate, families=families)
    (output / 'REPORT.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    with (output / 'PER_RATE.csv').open('w', newline='') as stream:
        fields = ['model', 'rate', 'epoch', *METRICS, 'delta_ACC', 'delta_WF1', 'delta_MSE']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**{key: row[key] for key in fields if not key.startswith('delta_')},
                **{'delta_' + key: row['delta'][key] if row['delta'] is not None else None
                   for key in METRICS}})
    lines = ['# MOSI Flat memory grid', '', 'INTERNAL TEST-ORACLE SCREEN', '',
             'Seed66; 150 epochs; LR3e-4 (1–100), LR3e-5 (101–150); GCNet random missing.',
             'Mean: all eight rates0.0–0.7. High: rates0.5–0.7.',
             'Nine independent training jobs; the same current-round baseline appears in both families.',
             'Only complete150-epoch/eight-rate runs enter performance tables.',
             'Deltas are candidate minus current-round baseline; unavailable baseline means no delta.',
             'Per-rate Test-best W-F1 may select different epochs; these are internal oracle results.', '']
    for family, data in families.items():
        lines.extend([f'## {family}', '',
            '| Model | Scope | ACC | W-F1 | MSE | ΔACC | ΔW-F1 | ΔMSE |',
            '|---|---|---:|---:|---:|---:|---:|---:|'])
        for row in data['aggregate']:
            differences = row['delta']
            values = ([f"{differences[key]:+.6f}" for key in METRICS]
                      if differences is not None else ['—'] * 3)
            lines.append(f"| {row['model']} | {row['scope']} | {row['ACC']:.3f} | "
                         f"{row['WF1']:.3f} | {row['MSE']:.6f} | {' | '.join(values)} |")
        lines.extend(['', '| Model | Rate | Best epoch | ACC | W-F1 | MSE |',
                      '|---|---:|---:|---:|---:|---:|'])
        for row in data['per_rate']:
            lines.append(f"| {row['model']} | {row['rate']} | {row['epoch']} | "
                         f"{row['ACC']:.3f} | {row['WF1']:.3f} | {row['MSE']:.6f} |")
        lines.append('')
    lines.extend(['## Progress and parameter counts', '',
                  '| Model | Status | Epoch | Parameters | Flat adapter | Memory elements/conversation |',
                  '|---|---|---:|---:|---:|---:|'])
    for mode, state in states.items():
        counts = state.get('parameters') or {}
        lines.append(f"| {mode} | {state['status']} | {state['epoch']} | {counts.get('total')} | "
                     f"{counts.get('flat_adapter')} | {counts.get('memory_state_elements_per_conversation')} |")
        if state.get('error'):
            lines.append(f"\n{mode} error: {state['error']}\n")
    (output / 'RESULT.md').write_text('\n'.join(lines) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    summarize(parser.parse_args().output)
