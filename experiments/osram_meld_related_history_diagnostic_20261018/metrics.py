"""Frozen diagnostic metrics; all rates are fractions, never percentages.

No thresholds or calibration functions are fitted. Relation metrics describe
all valid pairs unweighted; retrieval-weighted results are secondary only.
"""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _vector(values, name):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError(f'{name} must be a finite one-dimensional array')
    return result


def _binary_metrics(probability, target, weights):
    n = len(target)
    mass = float(weights.sum())

    def mean(values):
        return float(np.dot(values, weights) / mass) if mass else None

    predictions = probability >= 0.5
    positive_mass = float(weights[target == 1].sum())
    negative_mass = float(weights[target == 0].sum())
    tp = float(weights[predictions & (target == 1)].sum())
    fp = float(weights[predictions & (target == 0)].sum())
    fn = float(weights[~predictions & (target == 1)].sum())
    tn = float(weights[~predictions & (target == 0)].sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / positive_mass if positive_mass else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    two_classes = positive_mass > 0 and negative_mass > 0
    prevalence = positive_mass / mass if mass else None
    clipped = np.clip(probability, 1e-12, 1 - 1e-12)
    bins = []
    weighted_gap = 0.0
    indices = np.minimum((probability * 10).astype(np.int64), 9)
    for i in range(10):
        included = indices == i
        bin_weights = weights[included]
        bin_mass = float(bin_weights.sum())
        count = int(included.sum())
        mean_p = float(np.dot(probability[included], bin_weights) / bin_mass) if bin_mass else None
        frequency = float(np.dot(target[included], bin_weights) / bin_mass) if bin_mass else None
        gap = abs(mean_p - frequency) if bin_mass else None
        weighted_gap += bin_mass * gap if bin_mass else 0
        bins.append({
            'lower': i / 10, 'upper': (i + 1) / 10,
            'right_inclusive': i == 9, 'count': count,
            'weight_sum': bin_mass, 'mean_probability': mean_p,
            'positive_fraction': frequency, 'gap': gap,
        })
    return {
        'N': n, 'weight_sum': mass, 'prevalence': prevalence,
        'accuracy': mean(predictions == target),
        'balanced_accuracy': (tp / positive_mass + tn / negative_mass) / 2 if two_classes else None,
        'precision': precision if mass else None,
        'recall': recall if mass else None,
        'f1': f1 if mass else None,
        'auroc': float(roc_auc_score(target, probability, sample_weight=weights)) if two_classes else None,
        'auprc': float(average_precision_score(target, probability, sample_weight=weights)) if two_classes else None,
        'bce': mean(-target * np.log(clipped) - (1 - target) * np.log1p(-clipped)),
        'brier': mean((probability - target) ** 2),
        'ece': weighted_gap / mass if mass else None,
        'calibration_bins': bins,
        'constant_half': {
            'accuracy': prevalence, 'bce': float(np.log(2)) if mass else None,
            'brier': 0.25 if mass else None,
        },
        'majority_accuracy': max(prevalence, 1 - prevalence) if mass else None,
    }


def relation_metrics(probabilities, targets, weights=None):
    """Evaluate fixed same-class probabilities, optionally also weighted by R.

    BCE clips only its logarithm inputs to [1e-12,1-1e-12]. Calibration uses
    ten fixed equal-width probability bins and a prediction threshold of .5
    (ties positive). Single-class ranking/balanced accuracy is undefined.
    """
    probability = _vector(probabilities, 'probabilities')
    target = _vector(targets, 'targets')
    if len(probability) != len(target):
        raise ValueError('probabilities and targets must have equal lengths')
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError('probabilities must lie in [0,1]')
    if np.any((target != 0) & (target != 1)):
        raise ValueError('targets must be binary')
    result = _binary_metrics(probability, target, np.ones(len(target)))
    result['positives'] = int(target.sum())
    result['threshold'] = 0.5
    result['bce_epsilon'] = 1e-12
    quantiles = np.percentile(probability, [5, 25, 50, 75, 95]) if len(target) else [None] * 5
    result['probability'] = {
        'mean': float(probability.mean()) if len(target) else None,
        'std': float(probability.std()) if len(target) else None,
        'quantiles': {f'p{p:02d}': float(q) if q is not None else None
                      for p, q in zip([5, 25, 50, 75, 95], quantiles)},
        'min': float(probability.min()) if len(target) else None,
        'max': float(probability.max()) if len(target) else None,
        'fraction_le_0p1': float((probability <= .1).mean()) if len(target) else None,
        'fraction_ge_0p9': float((probability >= .9).mean()) if len(target) else None,
    }
    if weights is not None:
        retrieval_weights = _vector(weights, 'weights')
        if (len(retrieval_weights) != len(target) or np.any(retrieval_weights < 0)
                or not np.isfinite(retrieval_weights.sum())
                or retrieval_weights.sum() <= 0):
            raise ValueError('weights need matching length, nonnegative entries and positive finite sum')
        result['weighted_metrics'] = _binary_metrics(probability, target, retrieval_weights)
    return result


def paired_comparison(labels, reference_logits, candidate_logits):
    """Compare identical ordered examples; CE delta is candidate minus reference."""
    target = _vector(labels, 'labels')
    reference = np.asarray(reference_logits, dtype=np.float64)
    candidate = np.asarray(candidate_logits, dtype=np.float64)
    if (reference.ndim != 2 or reference.shape != candidate.shape
            or reference.shape[0] != len(target) or reference.shape[1] < 1
            or not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidate))):
        raise ValueError('logits must have equal finite [N,C] shapes matching labels')
    if np.any(target != np.floor(target)) or np.any((target < 0) | (target >= reference.shape[1])):
        raise ValueError('labels must be integer class indices within logits')
    target = target.astype(np.int64)
    ref_prediction = reference.argmax(axis=1)
    new_prediction = candidate.argmax(axis=1)
    ref_correct = ref_prediction == target
    new_correct = new_prediction == target

    def ce(logits):
        shifted = logits - logits.max(axis=1, keepdims=True)
        return np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(target)), target]

    changes = int((ref_prediction != new_prediction).sum())
    return {
        'N': len(target),
        'corrections': int((~ref_correct & new_correct).sum()),
        'harms': int((ref_correct & ~new_correct).sum()),
        'both_correct': int((ref_correct & new_correct).sum()),
        'both_wrong': int((~ref_correct & ~new_correct).sum()),
        'prediction_changes': changes, 'class_disagreement_count': changes,
        'mean_delta_ce': float(np.mean(ce(candidate) - ce(reference))) if len(target) else None,
    }
