"""Fixed full-precision promotion rules; never select from partial batches."""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import statistics

from .manifest import read, write, validate_round

RATES = tuple(f'{i / 10:.1f}' for i in range(8))


def scores(metrics):
    if metrics.get('selection_protocol') != 'per-rate-test-oracle' or set(metrics.get('test', {})) != set(RATES):
        raise ValueError('Expected all eight per-rate test-oracle results')
    values = {rate: float(metrics['test'][rate]['weighted_f1']) for rate in RATES}
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values.values()):
        raise ValueError('Invalid WF1')
    return {'per_rate': values, 'mean8': sum(values.values()) / 8,
            'high': sum(values[r] for r in RATES[5:]) / 3}


def rank_promotions(candidate_ids, rows, baseline_mean):
    if len(candidate_ids) != 20 or len(set(candidate_ids)) != 20 or any(
            name not in rows or rows[name].get('status') != 'complete' for name in candidate_ids):
        raise ValueError('Finish all twenty valid seed66 results before ranking')
    if any(not math.isfinite(rows[name]['mean8']) for name in candidate_ids):
        raise ValueError('Nonfinite candidate score')
    eligible = [name for name in candidate_ids if rows[name]['mean8'] > baseline_mean]
    return sorted(eligible, key=lambda name: (-rows[name]['mean8'], name))[:3]


def replication_decision(methods, baseline):
    baseline = {int(k): v for k, v in baseline.items()}
    if set(baseline) != {66, 67, 68}: raise ValueError('Three exact matched Flat seeds required')
    records = {}
    for method, values in methods.items():
        values = {int(k): v for k, v in values.items()}
        if set(values) != {66, 67, 68}: raise ValueError('Finish the fixed promotion set before decision')
        if any(not math.isfinite(v) for v in (*values.values(), *baseline.values())):
            raise ValueError('Nonfinite replication score')
        deltas = {s: values[s] - baseline[s] for s in (66, 67, 68)}
        records[method] = {'per_seed': values, 'paired_delta': deltas,
                           'mean_delta': sum(deltas.values()) / 3,
                           'seed_sd': statistics.stdev(values.values())}
    positive = [name for name in records if records[name]['mean_delta'] > 0]
    best = min(positive, key=lambda name: (-records[name]['mean_delta'], name)) if positive else None
    return {'next': 'verified_improvement' if best else 'new_round', 'best': best, 'methods': records,
            'label': 'INTERNAL ADAPTIVE TEST-ORACLE SEARCH; NOT AN UNBIASED OR SIGNIFICANCE CLAIM'}


def main():
    from .run import completion_status
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--baseline-audit', type=Path, required=True)
    args = parser.parse_args()
    ids = validate_round(read(args.manifest))
    baseline = {r['seed']: r['mean8'] for r in read(args.baseline_audit)['runs']}
    rows = {}
    for candidate in ids:
        output = args.root / 'runs' / candidate / 'seed_66'
        status, detail = completion_status(output)
        rows[candidate] = {'status': status, 'detail': detail}
        if status == 'complete': rows[candidate].update(scores(read(output / 'metrics.json')))
    result = {'seed66': rows, 'label': 'INTERNAL ADAPTIVE TEST-ORACLE SEARCH'}
    if all(row['status'] == 'complete' for row in rows.values()):
        result['promotions'] = rank_promotions(ids, rows, baseline[66])
    write(args.root / 'SUMMARY.json', result)


if __name__ == '__main__': main()
