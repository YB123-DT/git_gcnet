"""Publish completed same-mask V3/Flat tables without additional inference."""
import argparse
import csv
import json
from pathlib import Path

from .worker import REFERENCES, RATES


def summarize(root):
    root = Path(root); rows = []; aggregates = []; states = {}; provenance = {}
    for dataset, reference in REFERENCES.items():
        path = root/'runs'/dataset/'SUMMARY.json'
        state = json.loads(path.read_text()) if path.exists() else {'status': 'not_started'}
        states[dataset] = dict(status=state['status'], epoch=state.get('epoch'), error=state.get('error'))
        if state['status'] != 'complete':
            continue
        prior = json.loads((reference/'SUMMARY.json').read_text())
        provenance[dataset] = dict(source_commit=state['source_commit'],
            original_reference=str(reference), parameters=state['total_parameters'],
            added_parameters=state['source_role_v3_parameters'], seconds=state['seconds'],
            peak_memory_mib=state['peak_memory_mib'], mask_hashes=state['identity']['mask_hashes'])
        for rate in RATES:
            for name, data in [('Flat', prior), ('SourceRoleV3', state)]:
                value = data['evaluation'][rate]['router']; best = data['best_by_rate'][rate]
                rows.append(dict(dataset=dataset, model=name, seed=66, rate=rate,
                    epoch=best['epoch'], ACC=value['ACC'], WF1=value['WF1'], MSE=value['MSE']))
        for scope in ('mean', 'high'):
            for name, data in [('Flat', prior), ('SourceRoleV3', state)]:
                aggregates.append(dict(dataset=dataset, model=name, scope=scope,
                    **data['aggregate']['router'][scope]))
    payload = dict(label='INTERNAL TEST-ORACLE SCREEN', states=states,
        provenance=provenance, aggregate=aggregates, per_rate=rows)
    (root/'REPORT.json').write_text(json.dumps(payload, indent=2, sort_keys=True)+'\n')
    if rows:
        with (root/'METRICS.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ['# V3 Hierarchical Source–Role Interaction: MOSI66 / MOSEI66', '',
        'INTERNAL TEST-ORACLE SCREEN', '',
        'Fresh one-stage training; original GCNet masks, MSE, CHM and Flat retained.',
        'Same-source Base–Gap pairs → same-role cross-source attention → role-specific Local queries.',
        'Four zero-init64→512 corrections enter original Base/Gap, not a global1600-d residual.',
        'Eight-rate means combine per-rate TEST-best W-F1 checkpoints, not a deployment checkpoint.', '',
        '| Dataset | Model | Scope | ACC | W-F1 | MSE |', '|---|---|---|---:|---:|---:|']
    for row in aggregates:
        lines.append(f"| {row['dataset']} | {row['model']} | {row['scope']} | {row['ACC']:.3f} | {row['WF1']:.3f} | {row['MSE']:.6f} |")
    lines += ['', '## Per-rate', '', '| Dataset | Model | Rate | Epoch | ACC | W-F1 | MSE |',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['dataset']} | {row['model']} | {row['rate']} | {row['epoch']} | {row['ACC']:.3f} | {row['WF1']:.3f} | {row['MSE']:.6f} |")
    lines += ['', '## State', '', *[f'- {d}: {s}' for d, s in states.items()], '',
        'Only completed runs appear in metric tables. Smoke is not a performance result.',
        'Flat references predate opt-in V3; original initialization/config/split/exact masks and '
        'each training epoch mask/update/LR trace are checked. Historical source bytes are not claimed identical.',
        'Seed66/Test-oracle only; no independent generalization or source-semantic attribution claim.']
    (root/'RESULT.md').write_text('\n'.join(lines)+'\n')
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    summarize(parser.parse_args().output)
