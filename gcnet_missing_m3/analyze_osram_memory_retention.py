#!/usr/bin/env python3
"""Summarize OSRAM retention diagnostics without third-party dependencies.

Usage: python analyze_osram_memory_retention.py INPUT --output-dir OUTPUT
INPUT is a JSON list, JSONL stream, or CSV of raw diagnostic records.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


GROUP = ('dataset', 'missing_rate', 'target_modality')
QUERY = ('dataset', 'missing_rate', 'sample_id', 'time_index', 'target_modality')
METRICS = ('err_pre', 'err_decay', 'err_post', 'decay_damage', 'write_damage',
           'cos_pre', 'cos_decay', 'cos_post', 'norm_ratio_pre',
           'norm_ratio_decay', 'norm_ratio_post')
HEADERS = {
    'coverage': [*GROUP, 'missing_count', 'no_history_count', 'no_history_ratio'],
    'distance': [*GROUP, 'history_query_count', 'distance_mean', 'distance_median', 'distance_p90'],
    'damage': [*GROUP, 'distance_bucket', 'head_count',
               *[m + '_mean' for m in METRICS], 'decay_damage_median', 'write_damage_median'],
    'overlap': ['dataset', 'missing_rate', 'overlap_bucket', 'head_count',
                'max_key_overlap_mean', 'mean_key_overlap_mean', 'write_damage_mean', 'write_damage_median', 'decay_damage_mean'],
}
NOTES = (
    'Coverage and distance count unique missing queries, deduplicated by dataset, '
    'missing_rate, sample_id, time_index, target_modality. Distance excludes NO_HISTORY. '
    'Damage and overlap counts are head-level retention records, not query counts. '
    'Overlap buckets use max_key_overlap. Missing/non-finite metrics are excluded '
    'individually, never replaced by zero; empty CSV cells and NA denote unavailable '
    'statistics. Damage is signed (negative means improvement). err_decay measures '
    'the current read; err_post measures retention for future reads only, not the '
    'current prediction. P90 uses linear interpolation between sorted observations.'
)


def number(value):
    """Return a finite float, preserving absent values as None."""
    if value is None or (isinstance(value, str) and value.strip().lower() in ('', 'null', 'none', 'na', 'nan')):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def load_records(path):
    path = Path(path)
    with path.open(encoding='utf-8-sig', newline='') as handle:
        if path.suffix.lower() == '.csv':
            return list(csv.DictReader(handle))
        text = handle.read()
    if text.lstrip().startswith('['):
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError('Expected a list/stream of diagnostic record objects')
    return rows


def statistic(rows, field, function=mean):
    values = [value for row in rows if (value := number(row.get(field))) is not None]
    return function(values) if values else None


def percentile90(values):
    values = sorted(values)
    position = (len(values) - 1) * 0.9
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def distance_bucket(value):
    if value is None:
        return None
    if value < 1 or not value.is_integer():
        raise ValueError('history_distance must be a positive integer')
    return '1' if value == 1 else '2-3' if value <= 3 else '4-7' if value <= 7 else '8+'


def overlap_bucket(value):
    if value is None:
        return None
    if not 0 <= value <= 1:
        raise ValueError('max_key_overlap must lie in [0, 1]')
    for upper, label in [(0.1, '[0,.1)'), (0.2, '[.1,.2)'), (0.4, '[.2,.4)'), (0.6, '[.4,.6)')]:
        if value < upper:
            return label
    return '[.6,1]'


def summarize(records):
    queries = {}
    damage = defaultdict(list)
    overlap = defaultdict(list)
    for source in records:
        row = dict(source)
        for field in QUERY:
            if row.get(field) is None or row.get(field) == '':
                raise ValueError('Missing query identifier: ' + field)
        row['missing_rate'] = number(row['missing_rate'])
        row['sample_id'] = str(row['sample_id'])
        row['time_index'] = str(row['time_index'])
        if row['status'] not in ('retention', 'NO_HISTORY'):
            raise ValueError('Unknown status: ' + str(row['status']))
        key = tuple(row[field] for field in QUERY)
        distance = number(row.get('history_distance'))
        bucket = distance_bucket(distance) if row['status'] == 'retention' else None
        query_info = (row['status'], distance if row['status'] == 'retention' else None)
        if key in queries and queries[key] != query_info:
            raise ValueError('Conflicting status or history distance for query: ' + str(key))
        queries[key] = query_info
        if row['status'] == 'retention':
            group = tuple(row[field] for field in GROUP)
            if bucket is not None:
                damage[(*group, bucket)].append(row)
            bucket = overlap_bucket(number(row.get('max_key_overlap')))
            if bucket is not None:
                overlap[(row['dataset'], row['missing_rate'], bucket)].append(row)

    query_groups = defaultdict(list)
    for key, info in queries.items():
        query_groups[(key[0], key[1], key[4])].append(info)
    tables = {name: [] for name in HEADERS}
    for group, values in sorted(query_groups.items()):
        base = dict(zip(GROUP, group))
        no_history = sum(status == 'NO_HISTORY' for status, _ in values)
        tables['coverage'].append(dict(base, missing_count=len(values), no_history_count=no_history,
                                       no_history_ratio=no_history / len(values)))
        distances = [distance for status, distance in values if status == 'retention' and distance is not None]
        tables['distance'].append(dict(base, history_query_count=sum(status == 'retention' for status, _ in values),
                                       distance_mean=mean(distances) if distances else None,
                                       distance_median=median(distances) if distances else None,
                                       distance_p90=percentile90(distances) if distances else None))
    for key, rows in sorted(damage.items()):
        result = dict(zip((*GROUP, 'distance_bucket'), key), head_count=len(rows))
        result.update({metric + '_mean': statistic(rows, metric) for metric in METRICS})
        result.update({metric + '_median': statistic(rows, metric, median) for metric in ('decay_damage', 'write_damage')})
        tables['damage'].append(result)
    for key, rows in sorted(overlap.items()):
        result = dict(zip(('dataset', 'missing_rate', 'overlap_bucket'), key), head_count=len(rows))
        result.update({metric + '_mean': statistic(rows, metric) for metric in
                       ('max_key_overlap', 'mean_key_overlap', 'write_damage', 'decay_damage')})
        result['write_damage_median'] = statistic(rows, 'write_damage', median)
        tables['overlap'].append(result)
    # Lexical order would put the zero-overlap bucket last.
    order = ['[0,.1)', '[.1,.2)', '[.2,.4)', '[.4,.6)', '[.6,1]']
    tables['overlap'].sort(key=lambda row: (row['dataset'], row['missing_rate'], order.index(row['overlap_bucket'])))
    return tables


def write_reports(tables, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, headers in HEADERS.items():
        with (output_dir / (name + '.csv')).open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            writer.writerows(tables[name])
        lines = ['# ' + name.title(), '', NOTES, '',
                 '| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
        for row in tables[name]:
            cells = []
            for field in headers:
                value = row.get(field)
                cell = 'NA' if value is None else format(value, '.6g') if isinstance(value, float) else str(value)
                cells.append(cell.replace('|', '\\|').replace('\n', ' '))
            lines.append('| ' + ' | '.join(cells) + ' |')
        (output_dir / (name + '.md')).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    write_reports(summarize(load_records(args.input)), args.output_dir)


if __name__ == '__main__':
    main()
