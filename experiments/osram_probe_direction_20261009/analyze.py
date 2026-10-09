"""Read-only descriptive analysis of frozen OSRAM probe-direction diagnostics."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np


PARTS = ('full', 'hist', 'other')
RATES = tuple(round(i / 10, 1) for i in range(8))
SEEDS = (66, 67, 68)
LABEL = 'TEST-ORACLE INTERNAL DIAGNOSTIC'


def mean(values):
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def correlation(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) <= 1e-12 or np.ptp(y) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def ranks(values):
    """Average ranks, including ties; no optional statistics dependency."""
    values = np.asarray(values)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    del unique
    return (np.cumsum(counts) - (counts - 1) / 2.)[inverse]


def partial_correlation(x, y, controls):
    x, y, controls = np.asarray(x, float), np.asarray(y, float), np.asarray(controls, float)
    if controls.ndim == 1:
        controls = controls[:, None]
    design = np.column_stack([np.ones(len(x)), controls])
    if len(x) - np.linalg.matrix_rank(design) < 2:
        return None
    rx = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    ry = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    return correlation(rx, ry)


def classification(y, prediction):
    use = y != 0
    truth, pred = y[use] > 0, prediction[use] > 0
    if not len(truth):
        return dict(acc_nonzero=None, weighted_f1_nonzero=None)
    weighted = 0.
    for label in (False, True):
        tp = np.sum((truth == label) & (pred == label))
        denominator = np.sum(truth == label) + np.sum(pred == label)
        weighted += (2*tp/denominator if denominator else 0.) * np.sum(truth == label)
    return dict(acc_nonzero=float(np.mean(truth == pred)), weighted_f1_nonzero=float(weighted/len(truth)))


def array(rows, field):
    return np.asarray([r[field] for r in rows], float)


def summarize_group(rows):
    if not rows:
        raise ValueError('Cannot summarize an empty group')
    y, s = array(rows, 'y'), array(rows, 's_true')
    yb, sb = array(rows, 'y_base'), array(rows, 's_base')
    result = dict(n=len(rows), conversations_n=len({r['conversation_id'] for r in rows}),
                  nonneutral_n=int(np.sum(y != 0)),
                  negligible_gradient_n=sum(not r['valid_gradient'] for r in rows),
                  base_mse_y=mean((yb-y)**2), base_mse_s=mean((sb-s)**2))
    result.update({'base_'+k: v for k, v in classification(y, yb).items()})
    for field in ('delta_r_norm', 'hist_norm', 'other_norm', 'gradient_norm', 'local_error', 'full_replay_error'):
        result[field+'_mean'] = mean(array(rows, field))
        result[field+'_max'] = float(np.max(array(rows, field)))
    result['local_reconstruction_failure_n'] = int(np.sum(array(rows, 'local_error') > 1e-5))
    result['full_replay_failure_n'] = int(np.sum(array(rows, 'full_replay_error') > 1e-5))
    for diagnostic in ('reconstruction', 'orthogonality', 'probe_parity'):
        field = diagnostic+'_error'
        values = np.asarray([r[field] for r in rows if field in r], float)
        result[diagnostic+'_checked_n'] = len(values)
        result[field+'_mean'] = mean(values)
        result[field+'_max'] = float(np.max(values)) if len(values) else None
        result[diagnostic+'_failure_n'] = int(np.sum(abs(values) > 1e-5))
    for part in PARTS:
        yp, sp = array(rows, 'y_'+part), array(rows, 's_'+part)
        dy, ds = yp-yb, sp-sb
        result.update({part+'_'+k: v for k, v in classification(y, yp).items()})
        result.update({part+'_delta_y_mean': mean(dy), part+'_abs_delta_y_mean': mean(abs(dy)),
                       part+'_delta_s_mean': mean(ds), part+'_abs_delta_s_mean': mean(abs(ds)),
                       part+'_mse_y': mean((yp-y)**2), part+'_mse_s': mean((sp-s)**2),
                       part+'_delta_mse_y': mean((yp-y)**2-(yb-y)**2),
                       part+'_delta_mse_s': mean((sp-s)**2-(sb-s)**2),
                       part+'_corrections': int(np.sum((y != 0) & ((yb > 0) != (y > 0)) & ((yp > 0) == (y > 0)))),
                       part+'_harms': int(np.sum((y != 0) & ((yb > 0) == (y > 0)) & ((yp > 0) != (y > 0))))})
    nonadditivity = array(rows, 'y_full')-array(rows, 'y_hist')-array(rows, 'y_other')+yb
    result['nonadditivity_y_mean'] = mean(nonadditivity)
    result['nonadditivity_y_abs_mean'] = mean(abs(nonadditivity))
    return result


def associations(rows):
    pattern = array(rows, 'pattern').astype(int)
    controls = np.column_stack([np.log1p(array(rows, 'delta_r_norm')), array(rows, 'lag'),
                                *[(pattern == k).astype(float) for k in range(1, 8)]])
    result = []
    for part in PARTS:
        ds = array(rows, 's_'+part)-array(rows, 's_base')
        dy = array(rows, 'y_'+part)-array(rows, 'y_base')
        for transform in ('signed', 'absolute'):
            x, y = (ds, dy) if transform == 'signed' else (abs(ds), abs(dy))
            result.append(dict(component=part, transform=transform, n=len(rows),
                               pearson=correlation(x, y), spearman=correlation(ranks(x), ranks(y)),
                               partial_pearson=partial_correlation(x, y, controls),
                               partial_spearman=partial_correlation(ranks(x), ranks(y), controls),
                               residual_df=int(len(rows)-np.linalg.matrix_rank(np.column_stack([np.ones(len(rows)), controls])))))
    return result


def quadrants(rows):
    ds = abs(array(rows, 's_full')-array(rows, 's_base'))
    dy = abs(array(rows, 'y_full')-array(rows, 'y_base'))
    ts, ty = float(np.median(ds)), float(np.median(dy))
    result = []
    for large_s in (False, True):
        for large_y in (False, True):
            # Equality belongs to small, making all-zero groups unambiguous.
            use = ((ds > ts) == large_s) & ((dy > ty) == large_y)
            result.append(dict(quadrant=('large' if large_s else 'small')+'_s_'+('large' if large_y else 'small')+'_y',
                               threshold_abs_ds=ts, threshold_abs_dy=ty, n=int(use.sum()),
                               fraction=float(np.mean(use)), tie_rule='large strictly greater than median',
                               mean_abs_ds=mean(ds[use]), mean_abs_dy=mean(dy[use])))
    return result


def cluster_intervals(rows, resamples=500):
    """Conversation-cluster bootstrap of target-weighted mean MSE changes."""
    if resamples < 1:
        return []
    groups = defaultdict(list)
    for row in rows:
        groups[str(row['conversation_id'])].append(row)
    ids = sorted(groups)
    counts = np.asarray([len(groups[k]) for k in ids])
    rng = np.random.default_rng(66)
    draws = rng.integers(len(ids), size=(resamples, len(ids)))
    result = []
    for part in PARTS:
        for target, truth in (('y', 'y'), ('s', 's_true')):
            totals = np.asarray([sum((r[target+'_'+part]-r[truth])**2-(r[target+'_base']-r[truth])**2 for r in groups[k]) for k in ids])
            sampled = totals[draws].sum(axis=1)/counts[draws].sum(axis=1)
            low, high = np.quantile(sampled, [.025, .975])
            result.append(dict(component=part, target=target, mean_delta_mse=float(totals.sum()/counts.sum()),
                               ci_low=float(low), ci_high=float(high), conversations_n=len(ids), resamples=resamples,
                               interval_identifiable=len(ids) > 1))
    return result


def macro_rates(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['probe_seed'], row['arm'], row['pattern']].append(row)
    result = []
    for (seed, arm, pattern), values in sorted(groups.items(), key=lambda item: str(item[0])):
        rates = sorted(v['rate'] for v in values)
        if len(set(rates)) != len(rates):
            raise ValueError('Duplicate rate in macro average')
        entry = dict(probe_seed=seed, arm=arm, pattern=pattern, rates=rates, rates_n=len(rates),
                     complete_rates=tuple(rates) == RATES)
        for key in values[0]:
            if key not in ('rate', 'probe_seed', 'arm', 'pattern'):
                entry[key] = mean(v.get(key) for v in values)
        result.append(entry)
    return result


def seed_descriptions(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['arm'], row['pattern']].append(row)
    result = []
    excluded = {'probe_seed', 'arm', 'pattern', 'rates', 'rates_n', 'complete_rates'}
    for (arm, pattern), values in sorted(groups.items(), key=lambda item: str(item[0])):
        entry = dict(arm=arm, pattern=pattern, seeds=sorted(v['probe_seed'] for v in values),
                     complete_three_probe_seeds=sorted(v['probe_seed'] for v in values) == list(SEEDS),
                     comparable_rate_coverage=all(v['rates'] == values[0]['rates'] for v in values),
                     rates_by_seed={str(v['probe_seed']): v['rates'] for v in values})
        for key in values[0].keys()-excluded:
            nums = [v[key] for v in values if v.get(key) is not None]
            entry[key+'_mean'] = mean(nums)
            entry[key+'_sd'] = float(np.std(nums, ddof=1)) if len(nums) > 1 else None
        result.append(entry)
    return result


def load_rows(root):
    rows, seen = [], set()
    for path in sorted(Path(root).glob('rate_*/rows.csv')):
        with path.open(newline='') as stream:
            for raw in csv.DictReader(stream):
                row = {}
                for key, value in raw.items():
                    if key in ('arm', 'conversation_id'):
                        row[key] = value
                    elif key == 'valid_gradient':
                        if value.lower() not in ('true', 'false', '1', '0'):
                            raise ValueError('Invalid valid_gradient value')
                        row[key] = value.lower() in ('true', '1')
                    else:
                        try:
                            row[key] = float(value)
                        except (ValueError, TypeError):
                            row[key] = value
                required = {'rate', 'probe_seed', 'conversation_id', 'utterance_index', 'arm', 'pattern', 'lag', 'y', 's_true',
                            'delta_r_norm', 'hist_norm', 'other_norm', 'gradient_norm', 'valid_gradient', 'local_error', 'full_replay_error'}
                required.update(f'{kind}_{part}' for kind in ('y', 's') for part in ('base', *PARTS))
                if required-row.keys():
                    raise ValueError(f'Missing row fields: {required-row.keys()}')
                for key in required-{'arm', 'conversation_id', 'valid_gradient'}:
                    if not isinstance(row[key], (float, int)) or not np.isfinite(row[key]):
                        raise ValueError(f'Nonfinite/non-numeric row field: {key}')
                for key in ('probe_seed', 'utterance_index', 'pattern', 'lag'):
                    if int(row[key]) != row[key]:
                        raise ValueError(f'Noninteger {key}')
                    row[key] = int(row[key])
                if row['arm'] not in ('delete', 'control') or row['pattern'] not in range(1, 8):
                    raise ValueError('Invalid arm or availability pattern')
                identity = tuple(row[k] for k in ('rate', 'probe_seed', 'arm', 'conversation_id', 'utterance_index'))
                if identity in seen:
                    raise ValueError(f'Duplicate diagnostic observation: {identity}')
                seen.add(identity)
                rows.append(row)
    return rows


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def analyze(root, output=None, bootstrap=500):
    root = Path(root)
    output = Path(output) if output else root/'summary'
    rows = load_rows(root)
    if not rows:
        raise ValueError('No diagnostic rows found')
    status = json.loads((root/'STATUS.json').read_text()) if (root/'STATUS.json').exists() else {'status': 'missing'}
    groups = defaultdict(list)
    for row in rows:
        groups[row['rate'], row['probe_seed'], row['arm']].append(row)
    summaries, association_rows, quadrant_rows, intervals = [], [], [], []
    for (rate, seed, arm), values in sorted(groups.items()):
        identity = dict(rate=rate, probe_seed=seed, arm=arm)
        summaries.append(dict(**identity, pattern='all', **summarize_group(values)))
        for pattern in sorted({r['pattern'] for r in values}):
            summaries.append(dict(**identity, pattern=str(pattern), **summarize_group([r for r in values if r['pattern'] == pattern])))
        association_rows.extend(dict(**identity, **v) for v in associations(values))
        quadrant_rows.extend(dict(**identity, **v) for v in quadrants(values))
        if seed == 66:
            intervals.extend(dict(**identity, **v) for v in cluster_intervals(values, bootstrap))
    macro = macro_rates(summaries)
    seed_stats = seed_descriptions(macro)
    expected = {(rate, seed, arm) for rate in RATES for seed in SEEDS for arm in ('delete', 'control')}
    missing = sorted(expected-set(groups))
    summary = dict(label=LABEL, source_status=status, primary_probe_seed=66, robustness_probe_seeds=[67, 68],
                   row_count=len(rows), complete_expected_groups=not missing, missing_groups=missing,
                   per_rate=summaries, macro_rates=macro, descriptive_probe_seeds=seed_stats,
                   associations=association_rows, quadrants=quadrant_rows, cluster_bootstrap=intervals)
    output.mkdir(parents=True, exist_ok=True)
    for name, values in [('per_rate', summaries), ('macro_rates', macro), ('probe_seed_descriptions', seed_stats),
                         ('associations', association_rows), ('quadrants', quadrant_rows), ('cluster_bootstrap', intervals)]:
        write_csv(output/(name+'.csv'), values)
    (output/'SUMMARY.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    lines = [f'# {LABEL}', '', f'Source status: {status.get("status", "unknown")}; rows: {len(rows)}; missing rate/seed/arm groups: {len(missing)}.', '',
             'Frozen oldOSRAM and the existing preceding3_mean ProbeB are evaluated without training or new checkpoint/configuration selection. Inherited test-oracle selection remains a limitation. Probe 66 is primary; 67 and 68 are robustness checks. All are reported, with no best-seed selection. Three probe seeds are not three independently trained backbones.', '',
             'Full uses actual historical observed-text deletion. Hist projects its real raw-R displacement onto the normalized local probe gradient; other is the complementary displacement. The basis uses raw active-forward R coordinates, so coordinate scaling determines this geometry. This is a rank-one local probe-sensitive direction, not a pure emotion direction. The orthogonal complement may still contain the same decodable information because D is nonlinear; other does not mean non-emotion. Hist/other states can be off-trajectory synthetic states. Nonlinear outputs need not add; nonadditivity is reported explicitly.', '',
             'Deltas are intervention minus base. Positive delta MSE means worse. Sentiment MSE includes neutral targets; ACC and weighted F1 exclude y=0. Reconstruction failure counts use absolute error >1e-5. Negligible gradients use the producer valid_gradient flag.', '',
             'Association tables include signed and absolute Pearson/Spearman, and residual correlations adjusting both variables for log1p(displacement norm), lag, and one-hot availability within each rate, probe seed, and arm. Partial Spearman ranks the two outcomes before this same adjustment. These are statistical adjustments, not causal identification. Constant or insufficient-residual-degree groups have null correlations.', '',
             'Four quadrants use fixed within-rate/seed/arm medians of absolute full deltas. Equality is small. Thresholds and all quadrants are reported descriptively; they are not selection criteria.', '',
             'Macro statistics weight rates equally within each probe seed before descriptive averaging across probe seeds. Duplicate observations across seeds are never pooled as independent samples. Missing rates and unequal coverage are explicit. Bootstrap intervals use 500 conversation-cluster resamples by default within each rate and arm for primary probe 66 only; one-conversation intervals are marked unidentifiable.', '',
             '| Primary probe 66, equal-rate macro | ΔMSE full | ΔMSE hist | ΔMSE other |', '|---|---:|---:|---:|']
    for row in macro:
        if row['probe_seed'] == 66 and row['pattern'] == 'all':
            lines.append(f'| {row["arm"]} ({row["rates_n"]}/8 rates) | {row["full_delta_mse_y"]:.6g} | {row["hist_delta_mse_y"]:.6g} | {row["other_delta_mse_y"]:.6g} |')
    lines.extend(['', 'Detailed tables: per_rate.csv, associations.csv, quadrants.csv, macro_rates.csv, probe_seed_descriptions.csv, cluster_bootstrap.csv. SUMMARY.json retains source STATUS and completeness metadata.', ''])
    (output/'RESULT.md').write_text('\n'.join(lines))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--bootstrap', type=int, default=500)
    args = parser.parse_args()
    summary = analyze(args.root, args.output, args.bootstrap)
    print(json.dumps(dict(rows=summary['row_count'], complete=summary['complete_expected_groups'])))


if __name__ == '__main__':
    main()
