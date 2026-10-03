"""Report all twenty fixed candidates, including missing/failed runs, without selection claims."""
import argparse
import csv
import math
from pathlib import Path
import statistics
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.osram_readout20_20261003.run import (
    CANDIDATES, LABEL, REFERENCE, completion_status, now, read, sha, write,
)


def rate_scores(metrics):
    values = {f'{i / 10:.1f}': float(metrics['test'][f'{i / 10:.1f}']['weighted_f1']) * 100
              for i in range(8)}
    if any(not math.isfinite(v) or not 0 <= v <= 100 for v in values.values()):
        raise ValueError('Invalid W-F1 values')
    return dict(per_rate=values, mean_8rate=statistics.mean(values.values()),
                high_missing=statistics.mean(values[f'{i / 10:.1f}'] for i in (5, 6, 7)))


def summarize(root, reference=REFERENCE):
    baseline = rate_scores(read(reference / 'metrics.json'))
    queue_path = root / 'QUEUE.json'
    queued = read(queue_path).get('children', {}) if queue_path.exists() else {}
    rows, flat_rows = [], []
    for candidate in CANDIDATES:
        output = root / 'runs' / candidate
        status, detail = completion_status(output)
        if status != 'complete' and candidate in queued:
            queue_status = queued[candidate].get('status', status)
            if queue_status != 'complete':
                status = queue_status
            detail = queued[candidate].get('detail', detail)
        row = dict(candidate=candidate, status=status, detail=detail, seed=66)
        if status == 'complete':
            scores = rate_scores(read(output / 'metrics.json'))
            row.update(scores, delta_8rate=scores['mean_8rate']-baseline['mean_8rate'],
                       delta_high=scores['high_missing']-baseline['high_missing'],
                       metrics_sha256=sha(output / 'metrics.json'))
            for rate, value in scores['per_rate'].items():
                flat_rows.append(dict(candidate=candidate, status=status, rate=rate,
                    weighted_f1=value, baseline=baseline['per_rate'][rate],
                    delta=value-baseline['per_rate'][rate]))
        else:
            flat_rows.append(dict(candidate=candidate, status=status, rate='', weighted_f1='', baseline='', delta=''))
        rows.append(row)
    result = dict(label=LABEL, generated_at=now(), seed=66, baseline=baseline,
                  baseline_metrics_sha256=sha(reference / 'metrics.json'),
                  completed=sum(r['status'] == 'complete' for r in rows), requested=20, candidates=rows)
    summary_dir = root / 'summary'
    write(summary_dir / 'SUMMARY.json', result)
    with (summary_dir / 'per_rate.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=('candidate', 'status', 'rate', 'weighted_f1', 'baseline', 'delta'))
        writer.writeheader()
        writer.writerows(flat_rows)
    with (summary_dir / 'aggregate.csv').open('w', newline='') as stream:
        keys = ('candidate', 'status', 'mean_8rate', 'high_missing', 'delta_8rate', 'delta_high', 'detail')
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--reference', type=Path, default=REFERENCE)
    a = p.parse_args()
    summary = summarize(a.root, a.reference)
    print(f"Completed {summary['completed']}/20; {LABEL}")
