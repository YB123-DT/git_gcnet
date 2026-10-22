"""Report six paired MOSI runs and pre-registered B minus A effects."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from .worker import REFERENCES, RATES, VARIANTS, PAIRS, settings


def _write_json(path, payload):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def paired_differences(rows, key='scope'):
    """Compare only matching completed A/B records; do not rank partial runs."""
    differences = []
    for a, b in PAIRS:
        left = {row[key]: row for row in rows if row['model'] == a}
        right = {row[key]: row for row in rows if row['model'] == b}
        for scope in left.keys() & right.keys():
            differences.append(dict(pair=a + ' / ' + b, a=a, b=b, **{key: scope},
                **{'delta_' + metric: right[scope][metric] - left[scope][metric]
                   for metric in ('ACC', 'WF1', 'MSE')}))
    return differences


def summarize(output):
    output = Path(output)
    states, completed, provenance = {}, {}, {}
    rows, aggregate = [], []
    for variant in VARIANTS:
        for dataset in REFERENCES:
            identifier = dataset + '_' + variant
            path = output / 'runs' / identifier / 'SUMMARY.json'
            summary = json.loads(path.read_text()) if path.exists() else {'status': 'not_started'}
            states[identifier] = {key: summary.get(key) for key in
                ('status', 'epoch', 'error', 'seconds', 'parameters', 'peak_memory_mib', 'gpu_uuid')}
            if summary.get('status') != 'complete':
                continue
            if (summary.get('phase') != 'train' or summary.get('smoke_only')
                    or summary.get('dataset') != dataset or summary.get('variant') != variant
                    or summary.get('seed') != 66 or summary.get('epoch') != settings(dataset)['epochs']
                    or set(summary.get('evaluation', {})) != set(RATES)
                    or set(summary.get('best_by_rate', {})) != set(RATES)):
                raise ValueError('Not a completed authorized formal result: ' + identifier)
            completed.setdefault(dataset, {})[variant] = summary
            provenance[identifier] = {key: summary.get(key) for key in
                ('source_commit', 'parameters', 'seconds', 'peak_memory_mib', 'gpu_uuid')}
    for dataset, variants in completed.items():
        first = next(iter(variants.values()))['identity']
        for summary in variants.values():
            for name in ('source_commit', 'source_hashes', 'mask_hashes', 'split', 'seed_protocol', 'data_files'):
                if summary['identity'][name] != first[name]:
                    raise ValueError('Completed same-dataset variants differ: ' + name)
        reference = REFERENCES[dataset] / 'SUMMARY.json'
        reference_bytes = reference.read_bytes()
        prior = json.loads(reference_bytes)
        for summary in variants.values():
            if hashlib.sha256(reference_bytes).hexdigest() != summary['identity']['reference']['summary_sha256']:
                raise ValueError('Accepted Flat reference summary changed')
        if (prior.get('status') != 'complete' or prior.get('epoch') != settings(dataset)['epochs']
                or set(prior.get('evaluation', {})) != set(RATES)):
            raise ValueError('Incomplete accepted Flat reference')
        for model, summary in [('Flat66 reference', prior)] + list(variants.items()):
            for rate in RATES:
                rows.append(dict(dataset=dataset, model=model, seed=66, rate=rate,
                    epoch=summary['best_by_rate'][rate]['epoch'], **summary['evaluation'][rate]['router']))
            for scope, rates in (('mean', RATES), ('high', RATES[5:])):
                metrics = {key: sum(summary['evaluation'][rate]['router'][key] for rate in rates) / len(rates)
                           for key in ('ACC', 'WF1', 'MSE')}
                flat_wf1 = sum(prior['evaluation'][rate]['router']['WF1'] for rate in rates) / len(rates)
                aggregate.append(dict(dataset=dataset, model=model, seed=66, scope=scope,
                    delta_WF1_vs_Flat66=metrics['WF1'] - flat_wf1, **metrics))
    for a, b in PAIRS:
        variants = completed.get('CMUMOSI', {})
        if a in variants and b in variants:
            if variants[a]['identity']['initial_v3_sha256'] != variants[b]['identity']['initial_v3_sha256']:
                raise ValueError('Paired A/B added parameters did not initialize identically: ' + a)
    queue_path = output / 'QUEUE.json'
    queue = json.loads(queue_path.read_text()) if queue_path.is_file() else {}
    result = dict(label='INTERNAL TEST-ORACLE SCREEN', states=states,
        queue_status=queue.get('status'), smoke=queue.get('smoke', []),
        provenance=provenance, per_rate=rows, aggregate=aggregate,
        pair_differences=paired_differences(aggregate),
        per_rate_pair_differences=paired_differences(rows, key='rate'))
    _write_json(output / 'REPORT.json', result)
    with (output / 'PER_RATE.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['dataset', 'model', 'seed', 'rate', 'epoch', 'ACC', 'WF1', 'MSE'])
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# V3 paired interfaces: MOSI seed66', '', 'INTERNAL TEST-ORACLE SCREEN', '',
        'Three pre-registered A/B pairs: topology, Local Skip input, task outlet normalization.',
        'MOSI only: 150 epochs, 3e-4 for 100 epochs then 3e-5 for 50; six formal runs.',
        'Original Flat1600, cyclic GCNet random masks 0.0–0.7, batch32, Adam, WD1e-5 and MSE.',
        'Existing same-seed Flat results are reused; no new baseline training.', '',
        '| Dataset | Model | Scope | ACC | W-F1 | ΔW-F1 vs Flat | MSE |',
        '|---|---|---|---:|---:|---:|---:|']
    for row in aggregate:
        lines.append(f"| {row['dataset']} | {row['model']} | {row['scope']} | {row['ACC']:.3f} | "
                     f"{row['WF1']:.3f} | {row['delta_WF1_vs_Flat66']:+.3f} | {row['MSE']:.6f} |")
    lines += ['', '## Paired effects: B minus A', '',
        '| Pair | Scope | ΔACC | ΔW-F1 | ΔMSE |',
        '|---|---|---:|---:|---:|']
    for row in result['pair_differences']:
        lines.append(f"| {row['pair']} | {row['scope']} | {row['delta_ACC']:+.3f} | "
                     f"{row['delta_WF1']:+.3f} | {row['delta_MSE']:+.6f} |")
    lines += ['', '## Run state', '', *[f'- {name}: {value}' for name, value in states.items()], '',
        'Only completed formal runs appear in score tables. Smoke checks are not performance evidence.',
        'The eight rates may select different epochs by Test-best W-F1. Their mean is an internal',
        'Test-oracle statistic, not one checkpoint deployed across all rates. No extra seeds or HPO.',
        'Time and GPU memory are measured under shared-resource scheduling; no isolated speed claim.']
    temporary = output / 'RESULT.tmp'
    temporary.write_text('\n'.join(lines) + '\n')
    temporary.replace(output / 'RESULT.md')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    summarize(parser.parse_args().output)

