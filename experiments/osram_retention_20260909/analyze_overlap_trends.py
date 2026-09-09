#!/usr/bin/env python3
"""Descriptive overlap/damage association from existing retention logs only (stdlib)."""

import argparse
import csv
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gcnet_missing_m3.analyze_osram_memory_retention import (  # noqa: E402
    distance_bucket, number, overlap_bucket,
)

BUCKETS = ['[0,.1)', '[.1,.2)', '[.2,.4)', '[.4,.6)', '[.6,1]']


def ranks(values):
    result = [0.0] * len(values)
    order = sorted(range(len(values)), key=values.__getitem__)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        for index in order[start:end]:
            result[index] = (start + 1 + end) / 2
        start = end
    return result


def pearson(x, y):
    if len(x) < 2:
        return None
    mx, my = mean(x), mean(y)
    xx, yy = [v - mx for v in x], [v - my for v in y]
    denominator = math.sqrt(sum(v*v for v in xx) * sum(v*v for v in yy))
    return sum(a*b for a, b in zip(xx, yy)) / denominator if denominator else None


def correlations(rows):
    x, y = [r['x'] for r in rows], [r['y'] for r in rows]
    return dict(pearson=pearson(x, y), spearman=pearson(ranks(x), ranks(y)))


def adjusted(rows, ranked=False):
    """Demean raw variables (or global midranks) within joint categorical strata."""
    x, y = [r['x'] for r in rows], [r['y'] for r in rows]
    if ranked:
        x, y = ranks(x), ranks(y)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row['stratum']].append(i)
    for indices in groups.values():
        mx, my = mean(x[i] for i in indices), mean(y[i] for i in indices)
        for i in indices:
            x[i] -= mx
            y[i] -= my
    return pearson(x, y)


def summary(rows):
    return dict(n=len(rows), overlap_mean=mean(r['x'] for r in rows) if rows else None,
                damage_mean=mean(r['y'] for r in rows) if rows else None,
                damage_median=median(r['y'] for r in rows) if rows else None,
                damage_positive_fraction=mean(r['y'] > 0 for r in rows) if rows else None,
                **correlations(rows))


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def self_test():
    assert ranks([4, 1, 1, 2]) == [4, 1.5, 1.5, 3]
    assert pearson([1, 2, 3], [3, 2, 1]) == -1
    assert pearson([1, 1], [2, 3]) is None
    assert pearson([], []) is None
    rows = [dict(x=x, y=y, stratum=s) for x, y, s in
            [(0, 10, 'a'), (1, 11, 'a'), (10, 0, 'b'), (11, 1, 'b')]]
    assert adjusted(rows) == 1
    assert adjusted(rows, ranked=True) == 1
    assert number(None) is None and number(float('nan')) is None
    assert overlap_bucket(0.1) == '[.1,.2)' and overlap_bucket(1.0) == '[.6,1]'
    assert distance_bucket(8.0) == '8+'
    print('Self-tests passed: ties, constants, signed correlation, joint residuals, missing values, boundaries.')


def analyze(raw, output):
    runs, conversations, strata, buckets = [], [], [], []
    selected = {p.parent: p for p in raw.glob('*/retention.jsonl')}
    selected.update({p.parent: p for p in raw.glob('*/retention.jsonl.gz')})
    paths = sorted(selected.values())  # Prefer archived gzip; never double-count its plain copy.
    if len(paths) != 40:
        raise ValueError(f'Expected exactly existing 40 runs, found {len(paths)}')
    for path in paths:
        metadata = json.loads((path.parent / 'metadata.json').read_text())
        base = dict(run=path.parent.name, dataset=metadata['dataset'], seed=metadata['seed'], rate=metadata['rate'])
        rows, total, retention, missing_overlap, missing_damage = [], 0, 0, 0, 0
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt', encoding='utf-8') as handle:
            for line in handle:
                row = json.loads(line)
                total += 1
                if row['dataset'] != base['dataset'] or row['missing_rate'] != base['rate']:
                    raise ValueError(f'Metadata mismatch: {path}')
                if row['status'] != 'retention':
                    continue
                retention += 1
                x, y = number(row.get('max_key_overlap')), number(row.get('write_damage'))
                missing_overlap += x is None
                missing_damage += y is None
                if x is None or y is None:
                    continue
                bucket = overlap_bucket(x)
                distance = distance_bucket(number(row.get('history_distance')))
                if distance is None or row.get('head') is None or not row.get('target_modality'):
                    raise ValueError('Missing stratum identifier')
                rows.append(dict(x=x, y=y, conversation=row['sample_id'], bucket=bucket,
                                 stratum=(row['target_modality'], row['head'], distance)))
        if total != metadata['records']:
            raise ValueError(f'Record-count mismatch: {path}')
        by_conversation, by_stratum, by_bucket = defaultdict(list), defaultdict(list), defaultdict(list)
        for row in rows:
            by_conversation[row['conversation']].append(row)
            by_stratum[row['stratum']].append(row)
            by_bucket[row['bucket']].append(row)
        local_conversations = []
        for conversation, group in sorted(by_conversation.items()):
            result = dict(base, conversation=conversation, **summary(group))
            conversations.append(result)
            local_conversations.append(result)
        for (modality, head, distance), group in sorted(by_stratum.items()):
            strata.append(dict(base, modality=modality, head=head, distance_bucket=distance, **summary(group)))
        for bucket in BUCKETS:
            group = by_bucket[bucket]
            conv_means = defaultdict(list)
            for row in group:
                conv_means[row['conversation']].append(row['y'])
            buckets.append(dict(base, overlap_bucket=bucket, **summary(group),
                                conversation_count=len(conv_means),
                                conversation_equal_damage_mean=mean(mean(v) for v in conv_means.values()) if conv_means else None))
        result = dict(base, total_records=total, retention_records=retention,
                      missing_overlap=missing_overlap, missing_damage=missing_damage,
                      **summary(rows), residual_pearson=adjusted(rows),
                      residual_rank_pearson=adjusted(rows, ranked=True),
                      stratum_count=len(by_stratum), strata_n_lt_10=sum(len(v) < 10 for v in by_stratum.values()),
                      conversation_count=len(local_conversations))
        for metric in ('pearson', 'spearman'):
            vals = [c[metric] for c in local_conversations if c[metric] is not None]
            result['conversation_equal_' + metric + '_mean'] = mean(vals) if vals else None
            result['conversation_' + metric + '_defined'] = len(vals)
            result['conversation_' + metric + '_positive'] = sum(v > 0 for v in vals)
        conv_rows = [dict(x=c['overlap_mean'], y=c['damage_mean']) for c in local_conversations]
        result.update({'between_conversation_' + k: v for k, v in correlations(conv_rows).items()})
        runs.append(result)
        print(f"{base['run']}: n={len(rows)}, r={result['pearson']:.4f}, rho={result['spearman']:.4f}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    for suffix, table in [('runs', runs), ('conversations', conversations), ('strata', strata), ('buckets', buckets)]:
        write_csv(output / f'overlap_trends_{suffix}.csv', table)
    metrics = ['pearson', 'spearman', 'residual_pearson', 'residual_rank_pearson',
               'conversation_equal_pearson_mean', 'conversation_equal_spearman_mean',
               'between_conversation_pearson', 'between_conversation_spearman']
    aggregates = []
    for dataset in ['ALL'] + sorted({r['dataset'] for r in runs}):
        for metric in metrics:
            vals = [r[metric] for r in runs if (dataset == 'ALL' or r['dataset'] == dataset) and r[metric] is not None]
            aggregates.append(dict(dataset=dataset, metric=metric, defined_runs=len(vals),
                                   positive_runs=sum(v > 0 for v in vals), negative_runs=sum(v < 0 for v in vals),
                                   median=median(vals), minimum=min(vals), maximum=max(vals)))
    write_csv(output / 'overlap_trends_summary.csv', aggregates)
    lines = ['# Existing-log overlap versus signed write damage', '',
             f'Inputs: {len(runs)} existing runs; {sum(r["retention_records"] for r in runs)} retention records; '
             f'{sum(r["n"] for r in runs)} complete finite overlap/damage pairs. '
             f'Missing overlap: {sum(r["missing_overlap"] for r in runs)}; missing damage: {sum(r["missing_damage"] for r in runs)}.', '',
             'Descriptive only: no p-values, independence assumptions, causal claims, or justification for a protection mechanism. '
             'Negative write_damage means improvement; it is never clipped. Missing/non-finite overlap is excluded, never set to zero. '
             'Metadata dataset/rate and record counts are checked against each log. Checkpoint selection remains the metadata oracle protocol.', '',
             'Pearson/Spearman are computed within each dataset × seed × rate run on head-level records. '
             'Spearman uses average ranks for ties. Residual Pearson removes joint modality × head × history-distance-bucket means. '
             'Residual rank Pearson first globally ranks both variables within each run, then demeans ranks within the same joint strata; '
             'it is a categorical-adjusted rank association, not ordinary Spearman of raw residuals. Distance buckets: 1, 2–3, 4–7, 8+.', '',
             'Conversation-equal metrics average defined within-conversation correlations with one vote per conversation; '
             'constant/singleton groups yield NA, and defined/positive counts are exported. Between-conversation correlations use '
             'one (mean overlap, mean damage) pair per conversation and measure a different, ecological association. '
             'Runs share evaluation conversations across rates/seeds; run sign counts are descriptive, not independent replications.', '',
             'Bucket CSV retains all five fixed buckets, including empty ones. Its n is the number of complete head-level pairs; '
             'conversation_equal_damage_mean averages conversation means only among conversations present in that bucket. '
             'Sparse or empty strata/buckets do not support a universal monotonic claim. Strata with n < 10 are counted per run; '
             'singleton strata contribute zero residual variation; zero-variance correlations are NA.', '',
             '| Dataset | Metric | Positive / defined runs | Median | Range |',
             '| --- | --- | --- | --- | --- |']
    for row in aggregates:
        lines.append(f'| {row["dataset"]} | {row["metric"]} | {row["positive_runs"]}/{row["defined_runs"]} | '
                     f'{row["median"]:.5f} | {row["minimum"]:.5f} to {row["maximum"]:.5f} |')
    defined_conversations = [r for r in conversations if r['spearman'] is not None]
    defined_strata = [r for r in strata if r['spearman'] is not None]
    run_buckets = defaultdict(list)
    for row in buckets:
        run_buckets[row['run']].append(row)
    monotonic = {}
    for metric in ('damage_mean', 'damage_median', 'conversation_equal_damage_mean'):
        eligible = [group for group in run_buckets.values() if all(r[metric] is not None for r in group)]
        monotonic[metric] = (sum(all(a[metric] <= b[metric] for a, b in zip(group, group[1:]))
                                 for group in eligible), len(eligible))
    lines += ['', '## Stability and exceptions', '',
              f'Within-conversation Spearman is positive in {sum(r["spearman"] > 0 for r in defined_conversations)}'
              f'/{len(defined_conversations)} defined conversation–run groups (not unique independent conversations). '
              f'Among {len(strata)} joint strata, {sum(r["n"] < 10 for r in strata)} have n < 10; '
              f'{sum(r["spearman"] < 0 for r in defined_strata)}/{len(defined_strata)} defined stratum correlations are negative. '
              f'Among {len(buckets)} buckets, {sum(r["n"] == 0 for r in buckets)} are empty and '
              f'{sum(r["n"] < 10 for r in buckets)} have n < 10.', '',
              'Nondecreasing damage across all five ordered overlap buckets (complete-bucket runs only): ' +
              '; '.join(f'{key}: {value[0]}/{value[1]}' for key, value in monotonic.items()) + '.', '',
              'Overall and adjusted associations, including the conversation-equal within-conversation summaries, '
              'are directionally stable across these runs. Individual strata, bucket monotonicity, and especially '
              'MOSI between-conversation associations are not universal. This is evidence of association in these '
              'recorded diagnostics, not evidence that overlap causes damage or that a protection change will improve predictions.']
    lines += ['', 'Reproduce: `python experiments/osram_retention_20260909/analyze_overlap_trends.py --self-test` and '
              '`python experiments/osram_retention_20260909/analyze_overlap_trends.py`.', '']
    (output / 'overlap_trends.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=Path(__file__).parent / 'raw')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        analyze(args.raw, args.output)
