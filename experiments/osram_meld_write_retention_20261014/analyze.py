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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input', type=Path)
    source.add_argument('--csv', type=Path, help='Render an existing aggregate without rereading raw rows')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or (args.input if args.input else args.csv.parent)
    output.mkdir(parents=True, exist_ok=True)
    if args.csv:
        with args.csv.open() as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            row['rate'] = float(row['rate'])
            row['distance'] = int(row['distance'])
            row['head'] = 'all' if row['head'] == 'all' else int(row['head'])
            for name in METRICS:
                row[name] = float(row[name]) if row[name] else None
                row[name + '_n'] = int(row[name + '_n'])
    else:
        summary = json.loads((args.input / 'SUMMARY.json').read_text())
        if summary['status'] != 'complete':
            raise ValueError('Full eight-rate completed input required')
        rows = aggregate([args.input / value['rows_file'] for value in summary['rates'].values()])
    if not rows:
        raise ValueError('No audit rows')
    if not args.csv:
        with (output / 'AGGREGATE.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    # Curves remain per-rate: each rate uses its own retained checkpoint.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for metric in ('relative_error', 'error_change_from_write_after', 'centered_cosine', 'retrieval_top1'):
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
        fig.savefig(output / f'{metric}.png', dpi=140)
        fig.savefig(output / f'{metric}.pdf')
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for ax, modality in zip(axes, ('A', 'T', 'V')):
        for branch in ('write_before', 'write_after'):
            values = sorted((r for r in rows if r['head'] == 'all' and r['phase'] == 'write'
                             and r['modality'] == modality and r['branch'] == branch), key=lambda r: r['rate'])
            ax.plot([r['rate'] for r in values], [r['relative_error'] for r in values], marker='o', label=branch)
        ax.set_title(modality)
        ax.set_xlabel('Missing rate')
    axes[0].set_ylabel('Relative source-value error')
    axes[-1].legend()
    fig.tight_layout()
    fig.savefig(output / 'immediate_write.png', dpi=140)
    fig.savefig(output / 'immediate_write.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
