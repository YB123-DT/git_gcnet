"""Summarize exact selected artifacts only; never launch new inference."""
import argparse
import csv
import json
from pathlib import Path

from .worker import REFERENCE, RATES, VARIANTS


def summarize(root):
    root = Path(root)
    states, per_rate, aggregate, provenance = {}, [], [], {}
    reference_path = REFERENCE / 'SUMMARY.json'
    reference = json.loads(reference_path.read_text()) if reference_path.is_file() else None
    sources = [('Flat', reference)]
    for variant in VARIANTS:
        path = root / 'runs' / ('MOSI66_' + variant) / 'SUMMARY.json'
        data = json.loads(path.read_text()) if path.is_file() else dict(status='not_started')
        states[variant] = dict(status=data['status'], epoch=data.get('epoch'), error=data.get('error'))
        if data['status'] == 'complete' and data.get('phase') == 'train' and data.get('epoch') == 150:
            sources.append((variant, data))
            provenance[variant] = dict(source_commit=data['source_commit'], total_parameters=data['total_parameters'],
                retrieval_parameters=data['retrieval_parameters'], seconds=data['seconds'],
                peak_memory_mib=data['peak_memory_mib'], support_bank=data['identity']['support_bank'],
                exact_test_masks=data['identity']['mask_hashes']['test'])
    for name, data in sources:
        if data is None or data.get('status') != 'complete':
            continue
        for rate in RATES:
            values = data['evaluation'][rate]['router']
            per_rate.append(dict(model=name, dataset='CMUMOSI', seed=66, rate=rate,
                epoch=data['best_by_rate'][rate]['epoch'], **{k: values[k] for k in ('ACC', 'WF1', 'MSE')}))
        for scope in ('mean', 'high'):
            values = data['aggregate']['router'][scope]
            aggregate.append(dict(model=name, scope=scope, **values))
    payload = dict(label='INTERNAL TEST-ORACLE SCREEN', reference=str(REFERENCE), states=states,
        aggregate=aggregate, per_rate=per_rate, provenance=provenance)
    (root / 'REPORT.json').write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    if per_rate:
        with (root / 'METRICS.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(per_rate[0]))
            writer.writeheader()
            writer.writerows(per_rate)
    lines = ['# MOSI66 TRAIN-exemplar transfer', '', 'INTERNAL TEST-ORACLE SCREEN', '',
        'Only completed full training runs enter scores; smoke is not performance evidence.',
        'All eight rates use exact reference test masks. Each rate selects its own TEST-best W-F1 checkpoint;',
        'the eight-rate mean is not one deployable checkpoint. This is a seed66 exploratory comparison.', '',
        '| Model | Scope | ACC | W-F1 | MSE |', '|---|---|---:|---:|---:|']
    for row in aggregate:
        lines.append(f"| {row['model']} | {row['scope']} | {row['ACC']:.3f} | {row['WF1']:.3f} | {row['MSE']:.6f} |")
    lines += ['', '## Per rate', '', '| Model | Missing | Best epoch | ACC | W-F1 | MSE |', '|---|---:|---:|---:|---:|---:|']
    for row in per_rate:
        lines.append(f"| {row['model']} | {row['rate']} | {row['epoch']} | {row['ACC']:.3f} | {row['WF1']:.3f} | {row['MSE']:.6f} |")
    lines += ['', '## Scope and interpretation', '',
        '- Feature-only and label-aware use identical retrieval structure and nominal parameter count.',
        '- Feature-only supplies zero instead of neighbor labels; its label encoder weight has no task gradient.',
        '- TRAIN bank excludes the entire query conversation during training. No validation/test labels enter retrieval.',
        '- Inference requires the TRAIN feature/label bank. Its storage/runtime are not represented by parameter count.',
        '- CHM scan, original Local Skip, Flat and task head are preserved. Only original sentiment MSE trains the model.',
        '- Added behavior is a migration hypothesis, not evidence of a previously localized OSRAM bottleneck.',
        '- F2 must exceed Flat and feature-only before discussing follow-up; no automatic seeds/HPO/dataset expansion.',
        '', '## Run status', '', *[f'- {name}: {value}' for name, value in states.items()]]
    (root / 'RESULT.md').write_text('\n'.join(lines) + '\n')
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    summarize(parser.parse_args().output)
