"""Aggregate frozen storage diagnostics by rate, modality, head and distance."""
import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

METRICS = ('relative_error', 'norm_ratio', 'raw_cosine', 'centered_cosine',
           'retrieval_top1', 'retrieval_mrr', 'wrong_key_centered_cosine',
           'wrong_key_relative_error', 'error_change_from_write_after')


def aggregate(paths):
    groups = defaultdict(lambda: defaultdict(lambda: [0., 0]))
    for path in paths:
        with gzip.open(path, 'rt') as stream:
            for line in stream:
                row = json.loads(line)
                row['error_change_from_write_after'] = row['relative_error'] - row['write_after_relative_error']
                for head in (row['head'], 'all'):
                    key = (row['rate'], row['modality'], head, row['phase'], row['distance'], row['branch'])
                    for name in METRICS:
                        if row.get(name) is not None:
                            groups[key][name][0] += row[name]
                            groups[key][name][1] += 1
    result = []
    for key, stats in groups.items():
        row = dict(zip(('rate', 'modality', 'head', 'phase', 'distance', 'branch'), key))
        for name in METRICS:
            total, count = stats[name]
            row[name] = total / count if count else None
            row[name + '_n'] = count
        result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.input / 'SUMMARY.json').read_text())
    if summary['status'] != 'complete':
        raise ValueError('Full eight-rate completed input required')
    rows = aggregate([args.input / value['rows_file'] for value in summary['rates'].values()])
    if not rows:
        raise ValueError('No audit rows')
    with (args.input / 'AGGREGATE.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Curves remain per-rate: each rate uses its own retained checkpoint.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for metric in ('relative_error', 'centered_cosine', 'retrieval_top1'):
        fig, axes = plt.subplots(3, 8, figsize=(22, 8), sharex=True)
        for mi, modality in enumerate(('A', 'T', 'V')):
            for ri in range(8):
                ax = axes[mi, ri]
                for branch in ('normal', 'decay_only', 'write_only', 'hold'):
                    values = sorted((r for r in rows if r['head'] == 'all' and r['phase'] == 'future'
                                     and r['modality'] == modality and r['rate'] == ri / 10
                                     and r['branch'] == branch and r.get(metric) is not None),
                                    key=lambda r: r['distance'])
                    ax.plot([r['distance'] for r in values], [r[metric] for r in values], marker='o', label=branch)
                ax.set_title(f'{modality}, miss={ri / 10:.1f}')
                if mi == 2:
                    ax.set_xlabel('Future distance (utterances)')
                if ri == 0:
                    ax.set_ylabel(metric)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=4)
        fig.tight_layout(rect=(0, 0, 1, .94))
        fig.savefig(args.input / f'{metric}.png', dpi=140)
        fig.savefig(args.input / f'{metric}.pdf')
        plt.close(fig)


if __name__ == '__main__':
    main()
