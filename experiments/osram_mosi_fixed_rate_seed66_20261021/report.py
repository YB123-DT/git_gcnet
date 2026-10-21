"""Read saved fixed-rate results only; no inference or model selection."""
import argparse
import csv
import json
import math
from pathlib import Path

RATES = tuple(f'{i/10:.1f}' for i in range(8))
METRICS = ('ACC', 'WF1', 'MSE')


def summarize(root):
    root = Path(root); rows = []; states = {}; provenance = {}
    for rate in RATES:
        path = root/'runs'/('rate_'+rate.replace('.', 'p'))/'seed_66'/'SUMMARY.json'
        state = json.loads(path.read_text()) if path.exists() else {'status': 'not_started'}
        states[rate] = dict(status=state['status'], epoch=state.get('epoch'), error=state.get('error'))
        if state['status'] != 'complete':
            continue
        if (state.get('epoch') != 150 or state['config']['epochs'] != 150
                or not math.isclose(float(state['fixed_rate']), float(rate), abs_tol=0., rel_tol=0.)):
            raise ValueError('Completed fixed-rate record budget/rate mismatch: '+rate)
        selected = state['testbest']; fixed = selected['metrics']
        reference = state['identity']['cyclic_reference']
        cyclic = json.loads((Path(reference['path'])/'SUMMARY.json').read_text())
        baseline = cyclic['evaluation'][rate]['router']
        if not all(math.isfinite(v[k]) for v in (fixed, baseline) for k in METRICS):
            raise ValueError('Nonfinite completed metric: '+rate)
        row = dict(rate=rate, seed=66, selected_epoch=selected['epoch'])
        for metric in METRICS:
            row['fixed_'+metric] = fixed[metric]
            row['cyclic_'+metric] = baseline[metric]
            row['delta_'+metric+('_pp' if metric != 'MSE' else '')] = fixed[metric]-baseline[metric]
        rows.append(row)
        provenance[rate] = dict(cyclic_reference=reference, source_commit=state.get('source_commit'),
            testbest=selected, total_parameters=state.get('total_parameters'),
            seconds=state.get('seconds'), peak_memory_mib=state.get('peak_memory_mib'))
    aggregates = []
    for scope, required in [('mean', RATES), ('high', RATES[5:])]:
        subset = [r for r in rows if r['rate'] in required]
        if len(subset) == len(required):
            keys = [k for k in subset[0] if k not in ('rate', 'seed', 'selected_epoch')]
            aggregates.append(dict(scope=scope, **{k: sum(row[k] for row in subset)/len(subset) for k in keys}))
    payload = dict(label='INTERNAL TEST-ORACLE SCREEN',
        interpretation='eight independently trained models, not one deployment checkpoint',
        states=states, per_rate=rows, aggregate=aggregates, provenance=provenance)
    (root/'REPORT.json').write_text(json.dumps(payload, indent=2, sort_keys=True)+'\n')
    if rows:
        with (root/'METRICS.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ['# MOSI seed66: independent fixed-rate Flat vs cyclic Flat', '',
        'INTERNAL TEST-ORACLE SCREEN', '',
        'This is eight independently trained models, not one deployment checkpoint.',
        'Each fixed-rate model trains and evaluates only its designated missing rate.',
        'Per-rate TEST-best W-F1, earliest strict-best ties; original GCNet random-mask-v1.', '',
        '| Rate | Epoch | Fixed ACC | Cyclic ACC | ΔACC | Fixed W-F1 | Cyclic W-F1 | ΔW-F1 | Fixed MSE | Cyclic MSE |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['rate']} | {row['selected_epoch']} | {row['fixed_ACC']:.3f} | {row['cyclic_ACC']:.3f} | {row['delta_ACC_pp']:+.3f} | {row['fixed_WF1']:.3f} | {row['cyclic_WF1']:.3f} | {row['delta_WF1_pp']:+.3f} | {row['fixed_MSE']:.6f} | {row['cyclic_MSE']:.6f} |")
    lines += ['', '## Completed aggregates', '', '| Scope | Fixed ACC | Cyclic ACC | Fixed W-F1 | Cyclic W-F1 | ΔW-F1 |',
        '|---|---:|---:|---:|---:|---:|']
    for row in aggregates:
        lines.append(f"| {row['scope']} | {row['fixed_ACC']:.3f} | {row['cyclic_ACC']:.3f} | {row['fixed_WF1']:.3f} | {row['cyclic_WF1']:.3f} | {row['delta_WF1_pp']:+.3f} |")
    lines += ['', 'Only completed designated-rate results enter tables. A scope is aggregated only when all its rates finish.',
        'The fixed-rate training distribution differs from cyclic; this is not a paired same-training-trajectory comparison.',
        'Same seed/data/original task/architecture and original fixed evaluation mask are checked by the worker.',
        'No multi-seed or independent generalization claim.', '', '## State', '',
        *[f'- rate{r}: {s}' for r, s in states.items()]]
    (root/'RESULT.md').write_text('\n'.join(lines)+'\n')
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    summarize(parser.parse_args().output)
