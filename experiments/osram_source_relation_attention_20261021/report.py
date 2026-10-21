"""Publish only completed two-dataset screen results; no inference/selection."""
import argparse
import csv
import json
from pathlib import Path

from .worker import REFERENCES, RATES


def summarize(root):
    root = Path(root); rows = []; aggregates = []; states = {}
    for dataset, reference in REFERENCES.items():
        path = root/'runs'/dataset/'SUMMARY.json'
        state = json.loads(path.read_text()) if path.exists() else {'status': 'not_started'}
        states[dataset] = dict(status=state['status'], epoch=state.get('epoch'), error=state.get('error'))
        if state['status'] != 'complete': continue
        prior = json.loads((reference/'SUMMARY.json').read_text())
        for rate in RATES:
            for name, data in [('Flat', prior), ('SourceRelation', state)]:
                v = data['evaluation'][rate]['router']; best = data['best_by_rate'][rate]
                rows.append(dict(dataset=dataset, model=name, seed=66, rate=rate, epoch=best['epoch'],
                    ACC=v['ACC'], WF1=v['WF1'], MSE=v['MSE']))
        for scope in ('mean', 'high'):
            for name, data in [('Flat', prior), ('SourceRelation', state)]:
                aggregates.append(dict(dataset=dataset, model=name, scope=scope, **data['aggregate']['router'][scope]))
    payload = dict(label='INTERNAL TEST-ORACLE SCREEN', states=states, aggregate=aggregates, per_rate=rows)
    (root/'REPORT.json').write_text(json.dumps(payload, indent=2, sort_keys=True)+'\n')
    if rows:
        with (root/'METRICS.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ['# Source-relation attention: MOSI66 / MOSEI66', '', 'INTERNAL TEST-ORACLE SCREEN', '',
        'Fresh one-stage training, original GCNet masks/task loss; old source gamma disabled.',
        'Eight-rate means combine per-rate TEST-best W-F1 checkpoints; not one deployment checkpoint.', '',
        '| Dataset | Model | Scope | ACC | W-F1 | MSE |', '|---|---|---|---:|---:|---:|']
    for row in aggregates:
        lines.append(f"| {row['dataset']} | {row['model']} | {row['scope']} | {row['ACC']:.3f} | {row['WF1']:.3f} | {row['MSE']:.6f} |")
    lines += ['', '## State', '', *[f'- {d}: {s}' for d, s in states.items()], '',
        'Only complete runs appear in the result table. No multi-seed or generalization claim.',
        'The original Flat references predate this opt-in source integration; original initialization, masks, '
        'data/config and every training epoch mask/update/LR trace are checked. Default-path regression tests '
        'cover the reviewed gated source changes. Source files are not claimed byte-identical to older runs.']
    (root/'RESULT.md').write_text('\n'.join(lines)+'\n')
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    summarize(parser.parse_args().output)
