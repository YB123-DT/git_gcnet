"""Summaries of frozen, temporally persistent projected-Value perturbations."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

import numpy as np


def binary_metrics(labels, predictions):
    y, p = np.asarray(labels), np.asarray(predictions)
    eligible = y != 0
    y, p = y[eligible] > 0, p[eligible] > 0
    n = len(y)
    if not n:
        return dict(N_binary=0, ACC=None, W_F1=None)
    weighted = 0.
    for category in (False, True):
        tp = int(((y == category) & (p == category)).sum())
        support = int((y == category).sum())
        denominator = int((p == category).sum()) + support
        weighted += support * (2 * tp / denominator if denominator else 0.)
    return dict(N_binary=n, ACC=float(100 * (y == p).mean()), W_F1=100 * weighted / n)


def summarize_group(rows):
    y = np.array([r['label'] for r in rows])
    clean = np.array([r['clean_prediction'] for r in rows])
    noisy = np.array([r['noisy_predictions'] for r in rows])
    eligible = y != 0
    good = (clean > 0) == (y > 0)
    noisy_good = (noisy > 0) == (y[:, None] > 0)
    corrections = ((~good[:, None]) & noisy_good & eligible[:, None]).sum(0)
    harms = (good[:, None] & ~noisy_good & eligible[:, None]).sum(0)
    flips = ((noisy > 0) != (clean[:, None] > 0)) & eligible[:, None]
    per_draw = [binary_metrics(y, noisy[:, d]) for d in range(noisy.shape[1])]
    mse = float(np.square(clean - y).mean())
    noisy_mse = np.square(noisy - y[:, None]).mean(0)
    result = dict(N=len(y), draws=noisy.shape[1], clean=dict(binary_metrics(y, clean), MSE=mse),
                  noisy={k: float(np.mean([r[k] for r in per_draw])) if eligible.any() else None
                         for k in ('ACC', 'W_F1')},
                  noisy_MSE_mean=float(noisy_mse.mean()), delta_MSE_mean=float(noisy_mse.mean() - mse),
                  prediction_abs_shift_mean=float(np.abs(noisy - clean[:, None]).mean()),
                  prediction_shift_rms=float(np.square(noisy - clean[:, None]).mean() ** .5),
                  flip_rate_percent=float(100 * flips.sum() / (eligible.sum() * noisy.shape[1])) if eligible.any() else None,
                  corrections_mean=float(corrections.mean()), harms_mean=float(harms.mean()),
                  N_clean_wrong=int((~good & eligible).sum()), N_clean_correct=int((good & eligible).sum()),
                  correction_rate_percent=float(100 * corrections.sum() / ((~good & eligible).sum() * noisy.shape[1])) if (~good & eligible).any() else None,
                  harm_rate_percent=float(100 * harms.sum() / ((good & eligible).sum() * noisy.shape[1])) if (good & eligible).any() else None,
                  corrections_per_draw=corrections.tolist(), harms_per_draw=harms.tolist())
    for field in ('kappa_mean', 'read_noise_rms', 'task_variance', 'value_rms'):
        result[field + '_mean'] = float(np.mean([r[field] for r in rows]))
    return result


def aggregate_rows(rows):
    groups, heads = defaultdict(list), defaultdict(list)
    # Kappa bins are diagnostic quantiles within a dataset/rate, never a predictor.
    strata = defaultdict(list)
    for row in rows:
        strata[(row['dataset'], row['rate'])].append(row['kappa_mean'])
    bounds = {key: np.quantile(values, [.25, .5, .75]) for key, values in strata.items()}
    for row in rows:
        prefix = (row['dataset'], row['rate'], row['sigma'])
        named = [('overall', 'all'), ('pattern', row['pattern']),
                 ('clean_correct', str(row['clean_correct'])),
                 ('clean_margin', '<=0.25' if abs(row['clean_prediction']) <= .25 else ('0.25-0.5' if abs(row['clean_prediction']) <= .5 else '>0.5')),
                 ('has_history', str(row['history_count'] > 0)),
                 ('kappa_quartile', str(int(np.searchsorted(bounds[prefix[:2]], row['kappa_mean'], side='right')) + 1))]
        for name, value in named:
            groups[prefix + (name, value)].append(row)
        if row['sigma'] == .01:
            for h, kappa in enumerate(row['head_kappa']):
                heads[prefix + (h,)].append((kappa, row['head_task_variance'][h], row['head_prediction_shift'][h]))
    summary = dict(label='INTERNAL DIAGNOSTIC ONLY', groups=[], heads=[],
        interpretation='kappa=diag(G), not uncertainty; actual nonlinear responses and local first-order variance are distinct; all counts average the same fixed noise draws')
    for (dataset, rate, sigma, group, value), selected in sorted(groups.items()):
        summary['groups'].append(dict(dataset=dataset, rate=rate, sigma=sigma, group=group, value=value,
                                      **summarize_group(selected)))
    for (dataset, rate, sigma, head), selected in sorted(heads.items()):
        x = np.asarray(selected)
        summary['heads'].append(dict(dataset=dataset, rate=rate, sigma=sigma, head=head,
            N=len(x), kappa_mean=float(x[:, 0].mean()), task_variance_mean=float(x[:, 1].mean()),
            head_intervention_sigma=.01, head_intervention_draw=0,
            one_draw_head_prediction_abs_shift_mean=float(x[:, 2].mean())))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    with gzip.open(args.rows, 'rt') as handle:
        rows = [json.loads(line) for line in handle]
    if not rows:
        raise ValueError('No valid utterance diagnostic rows')
    summary = aggregate_rows(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'AGGREGATE.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    lines = ['# Frozen Value-noise diagnostic', '', 'INTERNAL DIAGNOSTIC ONLY. No training.', '',
             '| Dataset | Rate | sigma | N | Clean ACC | Noisy ACC | ΔMSE | Mean absolute prediction shift | Flip % | Corrections / harms |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(x):
        return 'N/A' if x is None else f'{x:.6g}'
    for r in summary['groups']:
        if r['group'] == 'overall':
            lines.append(f"| {r['dataset']} | {r['rate']} | {r['sigma']} | {r['N']} | {fmt(r['clean']['ACC'])} | {fmt(r['noisy']['ACC'])} | {fmt(r['delta_MSE_mean'])} | {fmt(r['prediction_abs_shift_mean'])} | {fmt(r['flip_rate_percent'])} | {fmt(r['corrections_mean'])} / {fmt(r['harms_mean'])} |")
    lines += ['', 'Noise is applied to actual historical projected Values, not separately to Base/Gap. The same unit noise is reused at both sigmas.',
              'Binary metrics/counts exclude y=0 and retain threshold prediction>0; MSE includes all valid labels.',
              'The head-only intervention uses draw0; it is not a Monte Carlo estimate. Kappa is not task uncertainty.',
              'AGGREGATE.json contains pattern, correctness, history, kappa-quartile and head strata. STATUS.json contains parity and covariance errors.',
              'Full G matrices and per-draw predictions are retained in remote batch NPZ/ROWS artifacts, not committed to Git.', '']
    (args.output / 'RESULT.md').write_text('\n'.join(lines))
    return summary


if __name__ == '__main__':
    main()
