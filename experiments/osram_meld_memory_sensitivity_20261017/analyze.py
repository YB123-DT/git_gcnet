"""Read-only postprocessing of a completed frozen sensitivity artifact set.

No model is loaded, no mask is generated and no preset is selected. Statistics
describe saved diagnostic interventions, not an independently selected method.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def rows(path, *, allow_missing=False):
    if not path.is_file():
        if allow_missing:
            return
        raise FileNotFoundError(path)
    with gzip.open(path, 'rt') as stream:
        for number, line in enumerate(stream, 1):
            try:
                yield json.loads(line)
            except (ValueError, TypeError) as error:
                raise ValueError(f'Invalid JSON at {path}:{number}') from error


def stats(values):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError('Require finite scalar diagnostic values')
    if not array.size:
        return dict(N=0, mean=None, median=None, p50=None, p95=None, minimum=None, maximum=None)
    return dict(N=int(array.size), mean=float(array.mean()), median=float(np.median(array)),
                p50=float(np.percentile(array, 50)), p95=float(np.percentile(array, 95)),
                minimum=float(array.min()), maximum=float(array.max()))


def vector_stats(values):
    """Summarize each occurrence's mean over heads, and each head separately."""
    if not values:
        return {**stats([]), 'per_head': []}
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError('Require a consistent finite [occurrence, head] diagnostic')
    return {**stats(array.mean(axis=1)), 'per_head': [stats(array[:, h]) for h in range(array.shape[1])]}


def history_bin(prior):
    return 'first' if prior == 0 else '1-3' if prior < 4 else '4-7' if prior < 8 else '8-15' if prior < 16 else '16+'


def head_effect_profiles(labels, baseline, removed, patterns):
    """Pearson similarity of eight ablation CE effects; never compare bases."""
    labels, baseline, removed, patterns = map(np.asarray, (labels, baseline, removed, patterns))
    n = len(labels)
    if baseline.shape != (n, 7) or removed.shape != (n, 8, 7) or patterns.shape != (n,):
        raise ValueError('Require aligned [N,7] baseline and [N,8,7] head ablations')
    if not np.isfinite(baseline).all() or not np.isfinite(removed).all():
        raise ValueError('Nonfinite logits cannot define head effect profiles')
    def ce(logits):
        shifted = logits.astype(float) - logits.max(-1, keepdims=True)
        return np.log(np.exp(shifted).sum(-1)) - shifted[np.arange(n), labels.astype(int)]
    effects = np.stack([ce(removed[:, head]) - ce(baseline) for head in range(8)], -1)
    groups = [('all', np.ones(n, dtype=bool))]
    groups.extend((str(pattern), patterns == pattern) for pattern in sorted(set(patterns))
                  if np.count_nonzero(patterns == pattern) >= 20)
    result = []
    for pattern, selected in groups:
        values = effects[selected]
        centered = values - values.mean(0, keepdims=True)
        norms = np.linalg.norm(centered, axis=0)
        matrix, pairs = [], []
        for i in range(8):
            row = []
            for j in range(8):
                value = None if norms[i] == 0 or norms[j] == 0 else float(np.clip(
                    np.dot(centered[:, i], centered[:, j]) / (norms[i] * norms[j]), -1., 1.))
                row.append(value)
                if i < j and value is not None:
                    pairs.append(value)
            matrix.append(row)
        result.append(dict(availability=pattern, N=int(selected.sum()), correlation_matrix=matrix,
            effect_definition='CE(remove head)-CE(original full)',
            mean_ce_delta_per_head=values.mean(0).tolist(),
            zero_variance_heads=np.flatnonzero(norms == 0).tolist(),
            finite_off_diagonal_pairs=len(pairs), off_diagonal_mean=float(np.mean(pairs)) if pairs else None,
            off_diagonal_min=float(min(pairs)) if pairs else None, off_diagonal_max=float(max(pairs)) if pairs else None))
    return result


def summarize_state(source, prefix, rate, allow_incomplete):
    metadata = {}
    for row in rows(source / f'{prefix}_utterances.jsonl.gz', allow_missing=allow_incomplete):
        if row['condition'] != 'baseline':
            continue
        key = (row['conversation_id'], row['utterance_index'])
        if key in metadata:
            raise ValueError('Duplicate baseline sample identity in utterance stream')
        metadata[key] = int(row['prior_valid_utterances'])
    grouped = defaultdict(lambda: {'N': 0, 'excluded_first': 0, 'values': defaultdict(list)})
    for row in rows(source / f'{prefix}_state_changes.jsonl.gz', allow_missing=allow_incomplete):
        identity = (row['conversation_id'], row['utterance_index'])
        if identity not in metadata:
            raise ValueError('State row is not aligned to baseline sample metadata')
        prior = metadata[identity]
        # Keep exclusion counts on the condition's all-history row. No first
        # utterance norm ratio can establish sensitivity to historical memory.
        for bucket in ('all_history', history_bin(prior)):
            group = grouped[(row['condition'], bucket)]
            if prior == 0:
                group['excluded_first'] += 1
                continue
            group['N'] += 1
            for field in ('prestate_relative_change', 'poststate_relative_change', 'base_relative_change'):
                group['values'][field].append(row[field])
            if any(row['active_gap']):
                group['values']['active_gap_relative_change'].append(row['gap_relative_change'])
            for field, value in row.items():
                if ('absolute' in field or field.endswith('_norm')) and isinstance(value, list):
                    group['values'][field].append(value)
    return [dict(rate=rate, condition=condition, history_bin=bucket,
                 N=group['N'], excluded_first=group['excluded_first'],
                 **{name: vector_stats(values) for name, values in group['values'].items()})
            for (condition, bucket), group in grouped.items() if bucket != 'first']


def summarize_direct(source, prefix, rate, allow_incomplete):
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows(source / f'{prefix}_direct_beta.jsonl.gz', allow_missing=allow_incomplete):
        for field in ('relative_write_delta_change', 'absolute_poststate_change'):
            groups[float(row['beta'])][field].append(row[field])
    return [dict(rate=rate, beta=beta, N=len(group['relative_write_delta_change']),
                 **{field: vector_stats(values) for field, values in group.items()})
            for beta, group in sorted(groups.items())]


def summarize_spectra(source, prefix, rate, ridge, allow_incomplete):
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows(source / f'{prefix}_spectra.jsonl.gz', allow_missing=allow_incomplete):
        observed = len(row['active_modalities'])
        if observed < 1:
            raise ValueError('Spectral row must contain an active observed key')
        group = groups[(observed, int(row['head']))]
        if len(row['weighted_gram_eigenvalues']) != observed:
            raise ValueError('Active Gram spectrum size disagrees with observation count')
        group['minimum'].append(row['weighted_min_eigenvalue_over_ridge'])
        group['maximum'].append(row['weighted_max_eigenvalue_over_ridge'])
        group['eigenvalues'].extend(np.asarray(row['weighted_gram_eigenvalues']) / ridge)
        group['unweighted'].extend(np.asarray(row['gram_eigenvalues']) / ridge)
        group['ridge'].extend(row['effective_ridge_diagonal'])
        group['below'].append(int(row['weighted_eigenvalues_below_ridge']))
    return [dict(rate=rate, observed_modalities=observed, head=head, N=len(group['minimum']),
                 minimum_eigenvalue_over_ridge=stats(group['minimum']),
                 maximum_eigenvalue_over_ridge=stats(group['maximum']),
                 weighted_eigenvalues_over_ridge=stats(group['eigenvalues']),
                 unweighted_eigenvalues_over_ridge=stats(group['unweighted']),
                 fraction_eigenvalues_below_ridge=float(sum(group['below']) / (len(group['below']) * observed)),
                 fraction_writes_with_eigenvalue_below_ridge=float(np.mean(np.asarray(group['below']) > 0)),
                 effective_ridge_min=float(min(group['ridge'])), effective_ridge_max=float(max(group['ridge'])))
            for (observed, head), group in sorted(groups.items())]


def analyze(source, *, allow_incomplete=False):
    source = Path(source)
    summary_path = source / 'SUMMARY.json'
    summary = json.loads(summary_path.read_text())
    status = summary.get('status')
    if not allow_incomplete and (status != 'complete' or not summary.get('rates') or
            any(not entry.get('full_coverage', False) for entry in summary['rates'].values())):
        raise ValueError('Require a complete full-coverage SUMMARY; use --allow-incomplete only for explicit previews')
    report = dict(source=str(source.resolve()), source_status=status,
        partial_preview=status != 'complete' or any(not entry.get('full_coverage', False) for entry in summary.get('rates', {}).values()),
        summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        protocol=summary.get('protocol'), source_code_sha256=summary.get('code_sha256'),
        notes=[
            'All saved presets are reported; this analysis does not select a winning preset.',
            'Metric values and deltas are fractions; multiply by100 for percentage points.',
            'State vector means average each occurrence over heads; per_head retains individual-head summaries.',
            'First/no-history states are excluded. Other zero denominators cannot be identified because baseline denominator norms were not saved.',
            'Active Gap excludes fully observed rows; saved Gap vectors already mask observed modalities and summarize their remaining slots jointly.',
            'History bins count prior valid utterances; lagged alpha cohorts instead use actual source-target distance and may repeat targets.',
            'Absolute state/norm summaries are included only when present in the source streams.',
            'Head profile correlations compare per-utterance CE effects of ablations, not cross-head representation coordinates; zero-variance profiles have null correlation.',
        ], state_changes=[], direct_beta=[], spectra=[], performance=[], head_ablation=[], lagged_alpha=[],
        head_effect_correlations=[], unavailable_optional_artifacts=[])
    ridge = float(summary.get('source_config', {}).get('osram_write_ridge', .001))
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError('Require finite positive write ridge')
    for rate, entry in summary.get('rates', {}).items():
        prefix = f'miss_{float(rate):.1f}'.replace('.', 'p')
        report['state_changes'].extend(summarize_state(source, prefix, rate, allow_incomplete))
        report['direct_beta'].extend(summarize_direct(source, prefix, rate, allow_incomplete))
        report['spectra'].extend(summarize_spectra(source, prefix, rate, ridge, allow_incomplete))
        performance = entry['metrics']
        baseline = next(row for row in performance if row['condition'] == 'baseline' and row['grouping'] == 'all')
        for row in performance:
            if len(row['class_recall']) != 7 or len(row['class_support']) != 7:
                raise ValueError('Require fixed seven-class recall/support metrics')
            report['performance'].append(dict(rate=rate, **row))
            if row['grouping'] != 'all' or not row['condition'].startswith('remove_'):
                continue
            report['head_ablation'].append(dict(rate=rate, condition=row['condition'], N=row['N'],
                **{f'full_minus_removed_{metric}': float(baseline[metric] - row[metric])
                    for metric in ('accuracy', 'unweighted_accuracy', 'weighted_f1')},
                corrections=row['corrections'], harms=row['harms'], prediction_changes=row['prediction_changes'],
                removed_cross_entropy=row['cross_entropy'], full_cross_entropy=baseline['cross_entropy'],
                class_recall_removed=row['class_recall'], class_recall_full=baseline['class_recall'],
                class_recall_delta_full_minus_removed=(np.asarray(baseline['class_recall']) - np.asarray(row['class_recall'])).tolist(),
                class_support=row['class_support']))
        learned = {row['distance']: row for row in entry.get('lagged_metrics', []) if row['alpha'] == 'learned'}
        for row in entry.get('lagged_metrics', []):
            if row['distance'] not in learned or row['N'] != learned[row['distance']]['N']:
                raise ValueError('Lagged alpha cohort has no matched learned-alpha reference')
            reference = learned[row['distance']]
            report['lagged_alpha'].append(dict(rate=rate, **row,
                delta_ce_vs_learned=float(row['cross_entropy'] - reference['cross_entropy']),
                **{f'delta_{metric}_vs_learned': float(row[metric] - reference[metric])
                   for metric in ('accuracy', 'unweighted_accuracy', 'weighted_f1')}))
        predictions_path = source / f'{prefix}_predictions.npz'
        if predictions_path.is_file():
            with np.load(predictions_path, allow_pickle=False) as predictions:
                for kind in ('base', 'gap'):
                    removed = np.stack([predictions[f'remove_{kind}_head_{head}'] for head in range(8)], 1)
                    profiles = head_effect_profiles(predictions['labels'], predictions['baseline'], removed,
                                                    predictions['availability_patterns'])
                    report['head_effect_correlations'].extend(dict(rate=rate, kind=kind, **row) for row in profiles)
        else:
            report['unavailable_optional_artifacts'].append(str(predictions_path))
    return report


def save_report(report, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    temporary = output / 'ANALYSIS.json.tmp'
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(output / 'ANALYSIS.json')
    for name in ('performance', 'head_ablation', 'lagged_alpha'):
        entries = report[name]
        if not entries:
            continue
        fields = list(dict.fromkeys(field for row in entries for field in row))
        with (output / f'{name.upper()}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                              for key, value in row.items()} for row in entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = analyze(args.input, allow_incomplete=args.allow_incomplete)
    save_report(report, args.output)
    print(json.dumps(dict(output=str(args.output), source_status=report['source_status'],
                         state_groups=len(report['state_changes']), head_rows=len(report['head_ablation']))))


if __name__ == '__main__':
    main()
