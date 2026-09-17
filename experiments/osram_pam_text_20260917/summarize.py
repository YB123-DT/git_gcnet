"""Summarize PAM-T metrics, pattern results, and minimal prediction diagnostics."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

RATES = tuple(f'{i / 10:.1f}' for i in range(8))
PATTERNS = ('A', 'T', 'V', 'AT', 'AV', 'TV', 'ATV')
HIGH = {'0.5', '0.6', '0.7'}


def weighted_f1(labels, predictions):
    labels = np.asarray(labels) > 0
    predictions = np.asarray(predictions) > 0
    return float(f1_score(labels, predictions, average='weighted'))


def pam_quality(prediction, target):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError('PAM prediction and target arrays must be [N,256]')
    if prediction.shape[0] == 0:
        return {'count': 0, 'centered_cosine': None, 'std_ratio': None}
    p = prediction - prediction.mean(0, keepdims=True)
    t = target - target.mean(0, keepdims=True)
    cosine = (p * t).sum(-1) / (np.linalg.norm(p, axis=-1) * np.linalg.norm(t, axis=-1) + 1e-8)
    std_ratio = float(prediction.std(0).mean() / max(target.std(0).mean(), 1e-12))
    return {'count': int(prediction.shape[0]), 'centered_cosine': float(cosine.mean()),
            'std_ratio': std_ratio}


def read_group(root, group):
    rows = []
    patterns = []
    quality = []
    for seed in (66, 67, 68, 69, 70):
        seed_root = root / 'mosi' / f'seed_{seed}'
        for rate in RATES:
            metrics = json.loads((seed_root / 'metrics.json').read_text())
            score = float(metrics['test'][rate]['weighted_f1']) * 100.0
            cp = int(metrics.get('selected_epoch_by_rate', {}).get(rate, -1))
            artifact = np.load(seed_root / f'predictions_miss_{rate.replace(".", "p")}.npz')
            labels, predictions, availability = artifact['labels'], artifact['predictions'], artifact['availability']
            rows.append(dict(group=group, seed=seed, rate=rate, weighted_f1=score,
                             selected_epoch=cp, selection_protocol=metrics['selection_protocol']))
            pattern_ids = availability[:, 0].astype(int) * 4 + availability[:, 1].astype(int) * 2 + availability[:, 2].astype(int)
            names = {1: 'V', 2: 'T', 3: 'TV', 4: 'A', 5: 'AV', 6: 'AT', 7: 'ATV'}
            for pattern, pattern_id in names.items():
                selected = pattern_ids == pattern_id
                patterns.append(dict(group=group, seed=seed, rate=rate, pattern=pattern,
                                     count=int(selected.sum()),
                                     weighted_f1=(weighted_f1(labels[selected], predictions[selected]) * 100.0
                                                  if bool(selected.any()) else None)))
            if group == 'PAM-T':
                quality.append(dict(seed=seed, rate=rate,
                                    **pam_quality(artifact['pam_prediction_text'], artifact['pam_target_text'])))
    return rows, patterns, quality


def aggregate(rows):
    result = {}
    for rate in RATES:
        values = [row['weighted_f1'] for row in rows if row['rate'] == rate]
        result[rate] = {'mean': float(np.mean(values)), 'std': float(np.std(values, ddof=1)),
                        'n': len(values)}
    overall = [row['weighted_f1'] for row in rows]
    high = [row['weighted_f1'] for row in rows if row['rate'] in HIGH]
    result['8-rate-mean'] = {'mean': float(np.mean(overall)), 'std': float(np.std(overall, ddof=1))}
    result['high-missing-mean'] = {'mean': float(np.mean(high)), 'std': float(np.std(high, ddof=1))}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pam-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    pam_rows, pattern_rows, quality = read_group(args.pam_root, 'PAM-T')
    # Existing reg-only summary is committed with this experiment branch and
    # is the locked comparison source, not retrained or reselected here.
    reg_summary = Path(__file__).resolve().parents[1] / 'osram_reg_only_20260916' / 'summary.json'
    summary = json.loads(reg_summary.read_text())
    comparisons = {
        'no-JEPA': summary['groups']['no-JEPA']['per_rate_mean_percent'],
        'reg-only': summary['groups']['reg-only']['per_rate_mean_percent'],
        'PAM-T': {rate: aggregate(pam_rows)[rate]['mean'] for rate in RATES},
    }
    payload = {
        'label': 'INTERNAL DIAGNOSTIC ONLY; NOT A FORMAL PAPER RESULT',
        'selection_protocol': 'per-rate-test-oracle',
        'overall': {name: aggregate(rows) for name, rows in [('PAM-T', pam_rows)]},
        'comparison_rate_mean_percent': comparisons,
        'comparison_overall_mean_percent': {
            'no-JEPA': summary['groups']['no-JEPA']['macro8_mean_percent'],
            'reg-only': summary['groups']['reg-only']['macro8_mean_percent'],
            'PAM-T': aggregate(pam_rows)['8-rate-mean']['mean'],
        },
        'comparison_high_missing_mean_percent': {
            'no-JEPA': summary['groups']['no-JEPA']['macro_high_mean_percent'],
            'reg-only': summary['groups']['reg-only']['macro_high_mean_percent'],
            'PAM-T': aggregate(pam_rows)['high-missing-mean']['mean'],
        },
        'pam_quality': quality,
    }
    (args.output / 'per_seed_rate.csv').write_text(
        'group,seed,rate,weighted_f1,selected_epoch,selection_protocol\n' +
        '\n'.join(','.join(str(row[key]) for key in ('group', 'seed', 'rate', 'weighted_f1', 'selected_epoch', 'selection_protocol')) for row in pam_rows) + '\n')
    (args.output / 'pattern_per_seed.csv').write_text(
        'group,seed,rate,pattern,count,weighted_f1\n' +
        '\n'.join(','.join(str(row[key]) for key in ('group', 'seed', 'rate', 'pattern', 'count', 'weighted_f1')) for row in pattern_rows) + '\n')
    (args.output / 'summary.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
