"""Summarize PAM-A metrics, patterns, paired comparisons, and mechanism stats."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

RATES = tuple(f'{i / 10:.1f}' for i in range(8))
SEEDS = (66, 67, 68, 69, 70)
PATTERNS = ('A', 'T', 'V', 'AT', 'AV', 'TV', 'ATV')
PATTERN_IDS = {1: 'V', 2: 'T', 3: 'TV', 4: 'A', 5: 'AV', 6: 'AT', 7: 'ATV'}
HIGH = {'0.5', '0.6', '0.7'}
T_MISSING = {'A', 'V', 'AV'}
T_PRESENT = {'T', 'AT', 'TV', 'ATV'}


def weighted_f1(labels, predictions):
    labels = np.asarray(labels) > 0
    predictions = np.asarray(predictions) > 0
    scores, weights = [], []
    for cls in (False, True):
        support = int(np.sum(labels == cls))
        if support == 0:
            continue
        tp = int(np.sum((labels == cls) & (predictions == cls)))
        fp = int(np.sum((labels != cls) & (predictions == cls)))
        fn = int(np.sum((labels == cls) & (predictions != cls)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores.append(f1)
        weights.append(support)
    return float(np.average(scores, weights=weights)) if weights else 0.0


def pam_prediction_quality(prediction, target):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError('PAM prediction and target arrays must be [N,256]')
    if prediction.shape[0] == 0:
        return {
            'count': 0,
            'centered_cosine': None,
            'std_ratio': None,
            'prediction_norm_mean': None,
        }
    p = prediction - prediction.mean(0, keepdims=True)
    t = target - target.mean(0, keepdims=True)
    cosine = (p * t).sum(-1) / (
        np.linalg.norm(p, axis=-1) * np.linalg.norm(t, axis=-1) + 1e-8
    )
    return {
        'count': int(prediction.shape[0]),
        'centered_cosine': float(cosine.mean()),
        'std_ratio': float(prediction.std(0).mean() / max(target.std(0).mean(), 1e-12)),
        'prediction_norm_mean': float(np.linalg.norm(prediction, axis=-1).mean()),
    }


def read_pam_e(root: Path):
    rows, patterns, quality, mechanism = [], [], [], []
    for seed in SEEDS:
        seed_root = root / 'mosi' / f'seed_{seed}'
        metrics = json.loads((seed_root / 'metrics.json').read_text())
        for rate in RATES:
            score = float(metrics['test'][rate]['weighted_f1']) * 100.0
            selected_epoch = int(metrics.get('selected_epoch_by_rate', {}).get(rate, -1))
            artifact_path = seed_root / f'predictions_miss_{rate.replace(".", "p")}.npz'
            artifact = np.load(artifact_path)
            labels = artifact['labels']
            predictions = artifact['predictions']
            availability = artifact['availability']
            rows.append({
                'group': 'PAM-A', 'seed': seed, 'rate': rate,
                'weighted_f1': score, 'selected_epoch': selected_epoch,
                'selection_protocol': metrics['selection_protocol'],
            })
            pattern_ids = (
                availability[:, 0].astype(int) * 4
                + availability[:, 1].astype(int) * 2
                + availability[:, 2].astype(int)
            )
            for pattern_id, pattern in PATTERN_IDS.items():
                selected = pattern_ids == pattern_id
                patterns.append({
                    'group': 'PAM-A', 'seed': seed, 'rate': rate,
                    'pattern': pattern, 'count': int(selected.sum()),
                    'weighted_f1': (
                        weighted_f1(labels[selected], predictions[selected]) * 100.0
                        if bool(selected.any()) else None
                    ),
                })
            if 'pam_prediction_text' in artifact.files and 'pam_target_text' in artifact.files:
                quality.append({
                    'seed': seed, 'rate': rate,
                    **pam_prediction_quality(
                        artifact['pam_prediction_text'], artifact['pam_target_text']
                    ),
                })
            if 'pam_history_text_count' in artifact.files:
                history_count = np.asarray(artifact['pam_history_text_count'])
                bank_nonempty = np.asarray(artifact['pam_bank_nonempty'])
                if history_count.size and bank_nonempty.size:
                    mechanism.append({
                        'seed': seed,
                        'rate': rate,
                        'history_text_count_mean': float(history_count.mean()),
                        'bank_nonempty_coverage': float(bank_nonempty.mean()),
                    })
    return rows, patterns, quality, mechanism


def aggregate_rows(rows):
    result = {}
    for rate in RATES:
        values = [row['weighted_f1'] for row in rows if row['rate'] == rate]
        result[rate] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values, ddof=1)),
            'n': len(values),
        }
    overall = [row['weighted_f1'] for row in rows]
    high = [row['weighted_f1'] for row in rows if row['rate'] in HIGH]
    result['8-rate-mean'] = {
        'mean': float(np.mean(overall)),
        'std': float(np.std(overall, ddof=1)),
    }
    result['high-missing-mean'] = {
        'mean': float(np.mean(high)),
        'std': float(np.std(high, ddof=1)),
    }
    return result


def per_seed_macro(rows):
    return {
        str(seed): float(np.mean([
            row['weighted_f1'] for row in rows if row['seed'] == seed
        ]))
        for seed in SEEDS
    }


def per_seed_high(rows):
    return {
        str(seed): float(np.mean([
            row['weighted_f1']
            for row in rows
            if row['seed'] == seed and row['rate'] in HIGH
        ]))
        for seed in SEEDS
    }


def aggregate_patterns(pattern_rows):
    result = {}
    for pattern in PATTERNS:
        vals = [
            row['weighted_f1'] for row in pattern_rows
            if row['pattern'] == pattern and row['weighted_f1'] is not None
        ]
        result[pattern] = float(np.mean(vals)) if vals else None
    # Group macro scores with equal weight per pattern, matching the PAM-T
    # result convention.  Rate 0.0 has no T-only/AT/TV rows, so row-weighted
    # grouping would otherwise over-weight ATV.
    missing = [result[p] for p in T_MISSING if result[p] is not None]
    present = [result[p] for p in T_PRESENT if result[p] is not None]
    result['T-missing'] = float(np.mean(missing)) if missing else None
    result['T-present'] = float(np.mean(present)) if present else None
    return result


def paired_delta(pam_rows, baseline_per_seed):
    pam_per_seed = per_seed_macro(pam_rows)
    delta = {
        str(seed): pam_per_seed[str(seed)] - float(baseline_per_seed[str(seed)])
        for seed in SEEDS
    }
    values = np.asarray([delta[str(seed)] for seed in SEEDS], dtype=np.float64)
    ttest = stats.ttest_rel(
        np.asarray([pam_per_seed[str(seed)] for seed in SEEDS], dtype=np.float64),
        np.asarray([float(baseline_per_seed[str(seed)]) for seed in SEEDS], dtype=np.float64),
    )
    try:
        wilcoxon = float(stats.wilcoxon(values).pvalue)
    except ValueError:
        wilcoxon = None
    return {
        'mean_delta_pp': float(values.mean()),
        'sd_delta_pp': float(values.std(ddof=1)),
        'positive': int((values > 0).sum()),
        'n': len(values),
        'paired_t_p': float(ttest.pvalue),
        'wilcoxon_p': wilcoxon,
        'per_seed_delta_pp': delta,
    }


def load_locked_summaries(repo_root: Path):
    pam_e = json.loads(
        (repo_root / 'experiments/osram_pam_episodic_text_20260917/results/summary.json').read_text()
    )
    pam_t = json.loads(
        (repo_root / 'experiments/osram_pam_text_20260917/results/summary.json').read_text()
    )
    controls = json.loads(
        (repo_root / 'experiments/osram_reg_only_20260916/summary.json').read_text()
    )
    return pam_e, pam_t, controls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pam-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows, pattern_rows, quality, mechanism = read_pam_e(args.pam_root)
    repo_root = Path(__file__).resolve().parents[2]
    pam_e_summary, pam_t_summary, controls_summary = load_locked_summaries(repo_root)

    task = aggregate_rows(rows)
    patterns = aggregate_patterns(pattern_rows)
    task['per_seed_macro8_percent'] = per_seed_macro(rows)
    task['per_seed_high_percent'] = per_seed_high(rows)

    controls_per_seed = {
        name: controls_summary['groups'][name]['per_seed_macro8_percent']
        for name in ('no-JEPA', 'reg-only', 'reg+NCE')
    }
    controls_per_rate = {
        name: controls_summary['groups'][name]['per_rate_mean_percent']
        for name in ('no-JEPA', 'reg-only', 'reg+NCE')
    }
    baseline_pam_e_seed = pam_e_summary['task']['per_seed_macro8_percent']
    baseline_pam_e_rate = pam_e_summary['comparison_per_rate_mean_percent']['PAM-E']
    baseline_pam_e_pattern = pam_e_summary['pattern']['macro_patterns_percent']
    baseline_pam_t_seed = pam_t_summary['task']['per_seed_macro8_percent']
    baseline_pam_t_rate = pam_t_summary['task']['per_rate_mean_percent']
    baseline_pam_t_pattern = pam_t_summary['pattern']['macro_patterns_percent']

    quality_valid = [row for row in quality if row['centered_cosine'] is not None]
    mechanism_payload = {
        'n_pam_prediction_samples': int(sum(row['count'] for row in quality_valid)),
        'mean_history_text_episode_count_T_missing': (
            float(np.mean([row['history_text_count_mean'] for row in mechanism]))
            if mechanism else None
        ),
        'T_missing_bank_nonempty_coverage': (
            float(np.mean([row['bank_nonempty_coverage'] for row in mechanism]))
            if mechanism else None
        ),
        'z_hat_T_norm_mean': (
            float(np.mean([row['prediction_norm_mean'] for row in quality_valid]))
            if quality_valid else None
        ),
        'prediction_target_centered_cosine': (
            float(np.mean([row['centered_cosine'] for row in quality_valid]))
            if quality_valid else None
        ),
        'prediction_target_std_ratio': (
            float(np.mean([row['std_ratio'] for row in quality_valid]))
            if quality_valid else None
        ),
    }

    pam_e_macro = task['8-rate-mean']['mean']
    pam_e_high = task['high-missing-mean']['mean']
    payload = {
        'experiment': 'PAM-A: Explicit Cross-Modal Episodic Text Memory on CMU-MOSI',
        'label': 'INTERNAL DIAGNOSTIC ONLY; NOT A FORMAL PAPER RESULT',
        'dataset': 'CMUMOSI',
        'fold': 1,
        'seeds': list(SEEDS),
        'missing_rates': list(RATES),
        'selection_protocol': 'per-rate-test-oracle',
        'task': {
            **task,
            'macro8_mean_percent': pam_e_macro,
            'macro8_sd_percent': task['8-rate-mean']['std'],
            'macro_high_mean_percent': pam_e_high,
            'macro_high_sd_percent': task['high-missing-mean']['std'],
        },
        'pattern': {
            'macro_patterns_percent': {
                pattern: patterns[pattern] for pattern in PATTERNS
            },
            'T_missing_macro_percent': patterns['T-missing'],
            'T_present_macro_percent': patterns['T-present'],
        },
        'comparison_per_rate_mean_percent': {
            'PAM-A': {rate: task[rate]['mean'] for rate in RATES},
            'PAM-E': baseline_pam_e_rate,
            'PAM-T': baseline_pam_t_rate,
            **controls_per_rate,
        },
        'comparison_overall_mean_percent': {
            'PAM-A': pam_e_macro,
            'PAM-E': float(pam_e_summary['task']['macro8_mean_percent']),
            'PAM-T': float(pam_t_summary['task']['macro8_mean_percent']),
            'no-JEPA': float(controls_summary['groups']['no-JEPA']['macro8_mean_percent']),
            'reg-only': float(controls_summary['groups']['reg-only']['macro8_mean_percent']),
            'reg+NCE': float(controls_summary['groups']['reg+NCE']['macro8_mean_percent']),
        },
        'comparison_high_missing_mean_percent': {
            'PAM-A': pam_e_high,
            'PAM-E': float(pam_e_summary['task']['macro_high_mean_percent']),
            'PAM-T': float(pam_t_summary['task']['macro_high_mean_percent']),
            'no-JEPA': float(controls_summary['groups']['no-JEPA']['macro_high_mean_percent']),
            'reg-only': float(controls_summary['groups']['reg-only']['macro_high_mean_percent']),
            'reg+NCE': float(controls_summary['groups']['reg+NCE']['macro_high_mean_percent']),
        },
        'pattern_comparison_percent': {
            'PAM-A': {pattern: patterns[pattern] for pattern in PATTERNS},
            'PAM-E': baseline_pam_e_pattern,
            'PAM-T': baseline_pam_t_pattern,
            'no-JEPA': {
                k: v * 100.0
                for k, v in pam_t_summary['pattern']['comparison_previous']['no-JEPA'].items()
            },
            'reg-only': {
                k: v * 100.0
                for k, v in pam_t_summary['pattern']['comparison_previous']['reg-only'].items()
            },
            'reg+NCE': {
                k: v * 100.0
                for k, v in pam_t_summary['pattern']['comparison_previous']['reg+NCE'].items()
            },
        },
        'paired': {
            'PAM-A - PAM-E': paired_delta(rows, baseline_pam_e_seed),
            'PAM-A - PAM-T': paired_delta(rows, baseline_pam_t_seed),
            **{
                f'PAM-A - {name}': paired_delta(rows, controls_per_seed[name])
                for name in ('no-JEPA', 'reg-only', 'reg+NCE')
            },
        },
        'mechanism': mechanism_payload,
        'per_seed_macro8_percent': task['per_seed_macro8_percent'],
        'per_seed_high_percent': task['per_seed_high_percent'],
    }

    (args.output / 'per_seed_rate.csv').write_text(
        'group,seed,rate,weighted_f1,selected_epoch,selection_protocol\n'
        + '\n'.join(
            ','.join(str(row[key]) for key in (
                'group', 'seed', 'rate', 'weighted_f1',
                'selected_epoch', 'selection_protocol'
            ))
            for row in rows
        ) + '\n'
    )
    (args.output / 'pattern_per_seed.csv').write_text(
        'group,seed,rate,pattern,count,weighted_f1\n'
        + '\n'.join(
            ','.join(str(row[key]) for key in (
                'group', 'seed', 'rate', 'pattern', 'count', 'weighted_f1'
            ))
            for row in pattern_rows
        ) + '\n'
    )
    (args.output / 'summary.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
