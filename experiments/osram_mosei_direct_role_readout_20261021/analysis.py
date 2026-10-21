"""Same-cohort evidence increments, following original MOSEI polarity metrics."""
import numpy as np
from experiments.osram_mosi_source_role_neutral_audit_20261020.analysis import metrics, paired

MODELS = ('L', 'LB', 'LG', 'Full')
EDGES = (('L', 'LB'), ('L', 'LG'), ('LB', 'Full'), ('LG', 'Full'), ('L', 'Full'))


def analyze(arrays):
    labels = np.asarray(arrays['labels'])
    if labels.ndim != 1 or not labels.size or not np.isfinite(labels).all():
        raise ValueError('Nonempty finite labels required')
    preds = {m: np.asarray(arrays['prediction_' + m]) for m in MODELS}
    if any(p.shape != labels.shape or not np.isfinite(p).all() for p in preds.values()):
        raise ValueError('Predictions must be finite and aligned')
    models = {m: metrics(labels, p) for m, p in preds.items()}
    comparisons = {a + '_to_' + b: paired(labels, preds[b], preds[a]) for a, b in EDGES}
    # Shared loader's selected-prediction guard expects this legacy key.
    return dict(models=models, comparisons=comparisons, groups={'all': {'learned': models['Full']}})
