"""Offline comparison on immutable original-Flat sample groups; no model imports."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
RATES = tuple(i / 10 for i in range(8))


def align_predictions(new, baseline, ids):
    """Use explicit IDs when present; otherwise require canonical saved-array order."""
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate canonical IDs')
    order = np.arange(len(ids))
    if 'utterance_ids' in new:
        supplied = [str(x) for x in new['utterance_ids']]
        if len(set(supplied)) != len(supplied) or set(supplied) != set(ids):
            raise ValueError('New prediction IDs differ from canonical IDs')
        lookup = {uid: i for i, uid in enumerate(supplied)}
        order = np.array([lookup[uid] for uid in ids])
    for key in ('labels', 'availability'):
        if len(new[key]) != len(ids) or not np.array_equal(new[key][order], baseline[key]):
            raise ValueError(f'{key} changed or saved artifact order is not canonical')
    prediction = np.asarray(new['predictions']).reshape(-1)
    if len(prediction) != len(ids) or not np.isfinite(prediction).all():
        raise ValueError('Invalid new predictions')
    return prediction[order]


def score(labels, baseline, relation, local):
    labels, baseline, relation, local = [np.asarray(x) for x in (labels, baseline, relation, local)]
    keep = labels != 0
    y, old, new, loc = [x[keep] > 0 for x in (labels, baseline, relation, local)]
    result = dict(n=int(keep.sum()), corrections=int(((old != y) & (new == y)).sum()),
                  harms=int(((old == y) & (new != y)).sum()))
    for name, pred in [('baseline', old), ('relation', new), ('local', loc)]:
        result[name + '_wf1'] = float(100 * f1_score(y, pred, average='weighted', zero_division=0)) if len(y) else None
    result['delta'] = result['relation_wf1'] - result['baseline_wf1'] if len(y) else None
    return result


def macro(cells):
    valid = [c for c in cells if c['n']]
    seeds = sorted({c['seed'] for c in valid})
    result = {key: sum(c[key] for c in valid) for key in ('n', 'corrections', 'harms')}
    result['nonempty_seed_rate_cells'] = len(valid)
    for key in ('baseline_wf1', 'relation_wf1', 'local_wf1', 'delta'):
        means = [statistics.mean(c[key] for c in valid if c['seed'] == s) for s in seeds]
        result[key] = statistics.mean(means) if means else None
    return result


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seed', default=66, type=int)
    args = parser.parse_args()
    sources = {}

    def track(path):
        sources[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    audit_root = ROOT / 'experiments/osram_context_audit_20261003'
    original = read_csv(track(audit_root / 'results/utterances.csv'))
    fixed = read_csv(track(audit_root / 'adjacent_results/utterances.csv'))
    mapping_path = ROOT / 'experiments/osram_cfg84_history_scale_fine_20260929/prediction_analysis/reconstructed_sample_ids.csv'
    mappings = sorted((r for r in read_csv(track(mapping_path)) if int(r['seed']) == args.seed),
                      key=lambda r: int(r['artifact_row']))
    if [int(r['artifact_row']) for r in mappings] != list(range(len(mappings))) or not mappings:
        raise ValueError('Incomplete original sample index')
    ids = [r['utterance_id'] for r in mappings]
    id_index = {uid: i for i, uid in enumerate(ids)}
    metrics = json.loads(track(args.run / 'metrics.json').read_text())
    per_rate, cells, joined = [], [], []
    baseline_dir = ROOT / 'experiments/osram_cfg84_history_scale_20260929/results'
    for rate in RATES:
        with np.load(track(baseline_dir / f'seed_{args.seed}_miss_{rate:.1f}_alpha_1.0.npz')) as data:
            baseline = {k: data[k].copy() for k in data.files}
        rate_key = f'{rate:.1f}'.replace('.', 'p')
        with np.load(track(args.run / f'predictions_miss_{rate_key}.npz')) as data:
            new = {k: data[k].copy() for k in data.files}
        prediction = align_predictions(new, baseline, ids)
        old_rows = {r['utterance_id']: r for r in original if int(r['seed']) == args.seed and float(r['rate']) == rate}
        if set(old_rows) != set(ids):
            raise ValueError('Original audit IDs incomplete')
        local = np.array([float(old_rows[uid]['local_prediction']) for uid in ids])
        for i, uid in enumerate(ids):
            r = old_rows[uid]
            pattern = ''.join(m for m, active in zip('ATV', baseline['availability'][i]) if active)
            if (float(r['label']) != baseline['labels'][i] or float(mappings[i]['label']) != baseline['labels'][i]
                    or float(r['memory_prediction']) != baseline['predictions'][i] or r['pattern'] != pattern):
                raise ValueError('Original audit no longer matches baseline artifacts')
        full = dict(seed=args.seed, rate=rate, **score(baseline['labels'], baseline['predictions'], prediction, local))
        if not np.isclose(full['relation_wf1'], 100 * metrics['test'][f'{rate:.1f}']['weighted_f1'], atol=1e-8, rtol=0):
            raise ValueError('Recomputed W-F1 differs from saved run metrics')
        per_rate.append(full)
        selected = [r for r in fixed if int(r['seed']) == args.seed and float(r['rate']) == rate]
        if len({r['utterance_id'] for r in selected}) != len(selected):
            raise ValueError('Duplicate fixed-cell IDs')
        for r in selected:
            old = old_rows[r['utterance_id']]
            if (r['local_prediction'] != old['local_prediction'] or r['memory_prediction'] != old['memory_prediction']
                    or r['boundary'] != ('near' if abs(float(old['local_prediction'])) <= .25 else 'far')
                    or r['relation'] != ('same' if (float(r['label']) > 0) == (float(r['previous_label']) > 0) else 'opposite')):
                raise ValueError('Fixed original grouping changed')
            joined.append(dict(r, relation_model_prediction=float(prediction[id_index[r['utterance_id']]])))
        for boundary in ('near', 'far'):
            for relation in ('same', 'opposite'):
                indices = [id_index[r['utterance_id']] for r in selected if r['boundary'] == boundary and r['relation'] == relation]
                cells.append(dict(seed=args.seed, rate=rate, boundary=boundary, relation=relation,
                                  **score(baseline['labels'][indices], baseline['predictions'][indices], prediction[indices], local[indices])))
    aggregate = [dict(boundary=b, relation=r, **macro([c for c in cells if c['boundary'] == b and c['relation'] == r]))
                 for b in ('near', 'far') for r in ('same', 'opposite')]
    summary = dict(seed=args.seed, overall=macro(per_rate), high_missing=macro([r for r in per_rate if r['rate'] >= .5]),
                   fixed_four_cells=aggregate)
    report = ['# Offline Relation comparison', '', 'INTERNAL DIAGNOSTIC ONLY', '',
              f'Per-rate BEST Test-oracle checkpoints; seed{args.seed} screening, not a formal paper result.',
              'No training/inference here. Label != 0; prediction > 0. W-F1 in percent; differences in percentage points.',
              'Fixed cells use OLD Flat Local-only predictions, not Relation-model predictions.',
              'Same/opposite means adjacent utterance label polarity, not a true emotion transition.', '',
              '|Rate|Flat full|Relation full|Delta|Corrections|Harms|N|', '|---|---:|---:|---:|---:|---:|---:|']
    for row in per_rate + [dict(rate='8-rate mean', **summary['overall']), dict(rate='High mean', **summary['high_missing'])]:
        report.append(f"|{row['rate']}|{row['baseline_wf1']:.3f}|{row['relation_wf1']:.3f}|{row['delta']:+.3f}|{row['corrections']}|{row['harms']}|{row['n']}|")
    report += ['', '|Fixed boundary|Adjacent relation|Old Local-only|Flat full|Relation full|Delta vs Flat|Corrections|Harms|N exposures|',
               '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in aggregate:
        report.append(f"|{row['boundary']}|{row['relation']}|{row['local_wf1']:.3f}|{row['baseline_wf1']:.3f}|{row['relation_wf1']:.3f}|{row['delta']:+.3f}|{row['corrections']}|{row['harms']}|{row['n']}|")
    report += ['', 'Corrections/harms compare Flat full → Relation full on exactly the same samples.',
               'W-F1 is calculated per seed/rate then macro-averaged; never pooled across rates.',
               'Counts sum repeated rate exposures, not independent utterances. No significance claim from one seed.',
               'No-ID trainer artifacts retain canonical evaluation-loader order; exact label and availability arrays are checked against the audited original artifacts.', '']
    for path, digest in sources.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Source changed during offline analysis')
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / 'per_rate.csv', per_rate)
    write_csv(args.output / 'four_cells_per_rate.csv', cells)
    write_csv(args.output / 'four_cells.csv', aggregate)
    write_csv(args.output / 'fixed_utterances.csv', joined)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (args.output / 'AUDIT.json').write_text(json.dumps(dict(training=False, inference=False, source_sha256=sources,
        fixed_group_source=str(audit_root / 'adjacent_results/utterances.csv')), indent=2) + '\n')
    (args.output / 'COMPARISON.md').write_text('\n'.join(report))
    print('\n'.join(report))


if __name__ == '__main__':
    main()
