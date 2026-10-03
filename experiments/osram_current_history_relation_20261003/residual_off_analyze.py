"""Decompose saved F / R-off / R-on predictions on immutable Flat audit groups."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('relation_saved_analysis', HERE / 'analyze.py')
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)
COMPARISONS = ('off_minus_f', 'on_minus_off', 'on_minus_f')
SCORES = ('f_wf1', 'off_wf1', 'on_wf1') + COMPARISONS
COUNTS = tuple(f'{name}_{kind}' for name in COMPARISONS for kind in ('corrections', 'harms'))


def decompose(labels, flat, off, on):
    result = {}
    for name, old, new in (('off_minus_f', flat, off), ('on_minus_off', off, on), ('on_minus_f', flat, on)):
        row = common.score(labels, old, new, flat)
        result['n'] = row['n']
        result[name] = row['delta']
        for kind in ('corrections', 'harms'):
            result[f'{name}_{kind}'] = row[kind]
        if name == 'off_minus_f':
            result.update(f_wf1=row['baseline_wf1'], off_wf1=row['relation_wf1'])
        elif name == 'on_minus_f':
            result['on_wf1'] = row['relation_wf1']
    if result['n'] and not np.isclose(result['on_minus_f'], result['off_minus_f'] + result['on_minus_off'], atol=1e-10):
        raise AssertionError('Metric difference identity failed')
    return result


def aggregate(rows):
    valid = [row for row in rows if row['n']]
    seeds = sorted({row['seed'] for row in valid})
    result = {key: sum(row[key] for row in valid) for key in ('n',) + COUNTS}
    result['nonempty_seed_rate_cells'] = len(valid)
    for key in SCORES:
        means = [statistics.mean(row[key] for row in valid if row['seed'] == seed) for seed in seeds]
        result[key] = statistics.mean(means) if means else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--off', type=Path, default=HERE / 'residual_off_results')
    parser.add_argument('--on', type=Path, default=HERE / 'results')
    parser.add_argument('--output', type=Path, default=HERE / 'residual_off_analysis')
    parser.add_argument('--seed', type=int, default=66)
    args = parser.parse_args()
    sources = {}

    def track(path):
        sources[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def load(path):
        with np.load(track(path)) as data:
            return {key: data[key].copy() for key in data.files}

    audit = common.ROOT / 'experiments/osram_context_audit_20261003'
    original = common.read_csv(track(audit / 'results/utterances.csv'))
    fixed = common.read_csv(track(audit / 'adjacent_results/utterances.csv'))
    mapping = common.ROOT / 'experiments/osram_cfg84_history_scale_fine_20260929/prediction_analysis/reconstructed_sample_ids.csv'
    mappings = sorted((r for r in common.read_csv(track(mapping)) if int(r['seed']) == args.seed), key=lambda r: int(r['artifact_row']))
    if not mappings or [int(r['artifact_row']) for r in mappings] != list(range(len(mappings))):
        raise ValueError('Incomplete original sample mapping')
    ids = [r['utterance_id'] for r in mappings]
    indices = {uid: i for i, uid in enumerate(ids)}
    on_metrics = json.loads(track(args.on / 'metrics.json').read_text())
    off_rows = json.loads(track(args.off / 'rows.json').read_text())
    if len(off_rows) != 8 or sorted(float(r['rate']) for r in off_rows) != list(common.RATES):
        raise ValueError('Expected exactly eight residual-off rate results')
    off_metrics = {float(r['rate']): r['metrics'] for r in off_rows}
    baseline_dir = common.ROOT / 'experiments/osram_cfg84_history_scale_20260929/results'
    per_rate, cells, joined = [], [], []
    for rate in common.RATES:
        baseline = load(baseline_dir / f'seed_{args.seed}_miss_{rate:.1f}_alpha_1.0.npz')
        filename = f'predictions_miss_{rate:.1f}.npz'.replace(f'{rate:.1f}', f'{rate:.1f}'.replace('.', 'p'))
        off = common.align_predictions(load(args.off / filename), baseline, ids)
        on = common.align_predictions(load(args.on / filename), baseline, ids)
        flat, labels = baseline['predictions'], baseline['labels']
        old_rows = {r['utterance_id']: r for r in original if int(r['seed']) == args.seed and float(r['rate']) == rate}
        if set(old_rows) != set(ids):
            raise ValueError('Original audit IDs incomplete')
        for i, uid in enumerate(ids):
            old = old_rows[uid]
            pattern = ''.join(m for m, active in zip('ATV', baseline['availability'][i]) if active)
            if float(old['label']) != labels[i] or float(mappings[i]['label']) != labels[i] or float(old['memory_prediction']) != flat[i] or old['pattern'] != pattern:
                raise ValueError('Original audit differs from baseline arrays')
        row = dict(seed=args.seed, rate=rate, **decompose(labels, flat, off, on))
        for name, expected in (('on_wf1', on_metrics['test'][f'{rate:.1f}']['weighted_f1']), ('off_wf1', off_metrics[rate]['weighted_f1'])):
            if not np.isclose(row[name], 100 * expected, atol=1e-8, rtol=0):
                raise ValueError(f'{name} does not reproduce saved metrics')
        per_rate.append(row)
        selected = [r for r in fixed if int(r['seed']) == args.seed and float(r['rate']) == rate]
        if len({r['utterance_id'] for r in selected}) != len(selected):
            raise ValueError('Duplicate fixed audit rows')
        for r in selected:
            old = old_rows[r['utterance_id']]
            if (r['local_prediction'] != old['local_prediction'] or r['memory_prediction'] != old['memory_prediction']
                    or r['boundary'] != ('near' if abs(float(old['local_prediction'])) <= .25 else 'far')
                    or float(r['label']) == 0 or float(r['previous_label']) == 0
                    or r['relation'] != ('same' if (float(r['label']) > 0) == (float(r['previous_label']) > 0) else 'opposite')):
                raise ValueError('Fixed original group changed')
            idx = indices[r['utterance_id']]
            joined.append(dict(r, relation_off_prediction=float(off[idx]), relation_on_prediction=float(on[idx])))
        for boundary in ('near', 'far'):
            for relation in ('same', 'opposite'):
                idx = [indices[r['utterance_id']] for r in selected if r['boundary'] == boundary and r['relation'] == relation]
                cells.append(dict(seed=args.seed, rate=rate, boundary=boundary, relation=relation, **decompose(labels[idx], flat[idx], off[idx], on[idx])))
    groups = [dict(boundary=b, relation=r, **aggregate([c for c in cells if c['boundary'] == b and c['relation'] == r])) for b in ('near', 'far') for r in ('same', 'opposite')]
    summary = dict(seed=args.seed, overall=aggregate(per_rate), high_missing=aggregate([r for r in per_rate if r['rate'] >= .5]), fixed_four_cells=groups)
    report = ['# Relation residual-off decomposition', '', 'INTERNAL DIAGNOSTIC ONLY', '',
              'Per-rate BEST Test-oracle checkpoints; one seed. No retraining, threshold search, or label-conditioned inference.',
              'F = original Flat; R-off = trained Relation checkpoint with only residual forced to zero; R-on = same checkpoint with residual enabled.',
              'W-F1 (%), nonzero labels only, prediction > 0. Fixed near/far groups use original Flat Local-off predictions at 0.25; same/opposite is reporting only.',
              'Scores are computed within each seed/rate then macro-averaged; counts sum rate exposures, not independent samples.', '',
              '|Rate / group|F|R-off|R-on|Off−F|On−Off|On−F|N|', '|---|---:|---:|---:|---:|---:|---:|---:|']
    tables = [(f"rate {r['rate']:.1f}", r) for r in per_rate] + [('8-rate mean', summary['overall']), ('High mean', summary['high_missing'])] + [(f"{r['boundary']} + {r['relation']}", r) for r in groups]
    for name, row in tables:
        report.append('|' + name + '|' + '|'.join('NA' if row[k] is None else f'{row[k]:.3f}' for k in SCORES) + f"|{row['n']}|")
    report += ['', '|Rate / group|Off−F corrections / harms|On−Off corrections / harms|On−F corrections / harms|', '|---|---:|---:|---:|']
    for name, row in tables:
        report.append('|' + name + '|' + '|'.join(f"{row[k + '_corrections']} / {row[k + '_harms']}" for k in COMPARISONS) + '|')
    report += ['', 'The W-F1 difference decomposition is an algebraic identity on matched samples, not independent causal effects.',
               'Corrections and harms are separately recomputed for each paired comparison and are NOT individually additive across the two stages.',
               'R-off−F describes the trained original-path change; R-on−R-off is the conditional inference-time effect of enabling the residual at the trained checkpoint.',
               'It does not isolate what training the original Flat alone under an identical optimization trajectory would have produced.',
               'Adjacent same/opposite polarity is not a true emotion transition; no mechanism or significance claim is made.', '']
    for path, digest in sources.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Source changed during analysis')
    args.output.mkdir(parents=True, exist_ok=False)
    for name, rows in (('per_rate.csv', per_rate), ('four_cells_per_rate.csv', cells), ('four_cells.csv', groups), ('fixed_utterances.csv', joined)):
        common.write_csv(args.output / name, rows)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (args.output / 'AUDIT.json').write_text(json.dumps(dict(training=False, inference=False, source_sha256=sources, fixed_group_source=str(audit / 'adjacent_results/utterances.csv')), indent=2) + '\n')
    (args.output / 'RESULT.md').write_text('\n'.join(report))
    print('\n'.join(report))


if __name__ == '__main__':
    main()
