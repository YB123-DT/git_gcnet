"""Pure aggregation for frozen Persistent7 internal diagnostics."""

import math
import statistics
from numbers import Real


PATTERNS = ('A', 'T', 'V', 'AT', 'AV', 'TV', 'ATV')
METRICS = ('ACC', 'UA', 'WF1', 'CE')


def _finite(value, context):
    if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f'{context} must be a finite number')


def _validate_metrics(metrics, context):
    if not isinstance(metrics, dict):
        raise ValueError(f'Missing metrics for {context}')
    for name in (*METRICS, 'N', 'recall', 'support'):
        if name not in metrics:
            raise ValueError(f'Missing {name} for {context}')
        value = metrics[name]
        if name in ('recall', 'support'):
            if not isinstance(value, (list, tuple)) or not value:
                raise ValueError(f'{context}/{name} must be a nonempty finite sequence')
            for entry in value:
                _finite(entry, f'{context}/{name}')
        else:
            _finite(value, f'{context}/{name}')


def _summary(per_seed):
    return {
        'mean': {metric: statistics.mean(row[metric] for row in per_seed.values())
                 for metric in METRICS},
        'sample_std': {metric: statistics.stdev(row[metric] for row in per_seed.values())
                       for metric in METRICS},
    }


def aggregate_runs(rows, folds, seeds=(66, 67, 68)):
    """Average folds equally per seed, then report seed mean and sample SD.

    Require every seed/fold pair and all seven finite pattern results. Counts,
    per-class recalls and supports remain in the input run records; they are
    validated but never turned into synthetic averaged counts or pooled scores.
    """
    folds, seeds = tuple(folds), tuple(seeds)
    if not folds or len(set(folds)) != len(folds):
        raise ValueError('folds must be nonempty and unique')
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError('seeds must contain at least two unique seeds for sample SD')
    expected = {(seed, fold) for seed in seeds for fold in folds}
    runs = {}
    for row in rows:
        key = (row.get('seed'), row.get('fold'))
        if key in runs:
            raise ValueError(f'Duplicate seed/fold run: {key}')
        if key not in expected:
            raise ValueError(f'Unexpected seed/fold run: {key}')
        patterns = row.get('patterns')
        if not isinstance(patterns, dict) or set(patterns) != set(PATTERNS):
            raise ValueError(f'Expected exactly seven patterns for seed/fold {key}')
        for pattern in PATTERNS:
            result = patterns[pattern]
            metrics = result.get('metrics') if isinstance(result, dict) else None
            _validate_metrics(metrics, f'{key}/{pattern}')
        runs[key] = row
    missing = expected - set(runs)
    if missing:
        raise ValueError(f'Missing seed/fold runs: {sorted(missing)}')

    per_seed = {
        str(seed): {
            pattern: {
                metric: statistics.mean(
                    runs[(seed, fold)]['patterns'][pattern]['metrics'][metric]
                    for fold in folds
                ) for metric in METRICS
            } for pattern in PATTERNS
        } for seed in seeds
    }
    result = {
        'per_seed': per_seed,
        'patterns': {
            pattern: _summary({seed: values[pattern] for seed, values in per_seed.items()})
            for pattern in PATTERNS
        },
    }
    for name, selected in (
        ('macro_six', PATTERNS[:-1]),
        ('macro_seven', PATTERNS),
        ('no_text', ('A', 'V', 'AV')),
    ):
        macro = {
            seed: {metric: statistics.mean(values[pattern][metric] for pattern in selected)
                   for metric in METRICS}
            for seed, values in per_seed.items()
        }
        result[name] = {'per_seed': macro, **_summary(macro)}
    return result


def render_report(datasets):
    """Render completed dataset aggregations without reference/performance claims."""
    lines = [
        '# Frozen Persistent7 diagnostics',
        '',
        'INTERNAL DIAGNOSTIC ONLY. Original Flat checkpoints (not Nested), frozen '
        'inference with no training or fine-tuning. BEST checkpoint selection retains '
        'the original test-oracle limitation.',
        '',
        'A = Audio; T = Text; V = Video. Singles use missing-rate 0.7 BEST; '
        'pairs use 0.3 BEST; full ATV uses 0.0 BEST. Availability is fixed over '
        'each conversation and its history.',
        '',
        'Complete seeds 66, 67, 68 are required for every dataset. For IEMOCAP4 '
        'and IEMOCAP6, first average the five folds equally within each seed; '
        'MELD has one fold. Then report the three-seed mean ± sample SD. '
        'No utterance pooling. macro6 averages the six non-full patterns within '
        'each seed; full7 includes all seven. Counts remain in the original run records.',
        '',
    ]
    columns = [f'{name} {metric}' for name in datasets for metric in METRICS[:3]]
    lines += ['| Pattern | ' + ' | '.join(columns) + ' |',
              '| --- | ' + ' | '.join('---:' for _ in columns) + ' |']
    for label, key in [(pattern, pattern) for pattern in PATTERNS] + [
        ('macro6', 'macro_six'), ('full7', 'macro_seven'),
    ]:
        cells = []
        for dataset in datasets.values():
            aggregate = dataset['aggregation']
            summary = aggregate['patterns'][key] if key in PATTERNS else aggregate[key]
            cells.extend(f"{summary['mean'][metric]:.3f} ± {summary['sample_std'][metric]:.3f}"
                         for metric in METRICS[:3])
        lines.append('| ' + label + ' | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines) + '\n'
