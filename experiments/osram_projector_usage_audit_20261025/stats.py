"""Descriptive single-checkpoint statistics; no persistent/dead-unit claim."""
import numpy as np


class Stats:
    def __init__(self, width):
        self.count = np.zeros(8, dtype=np.int64)
        self.activity = np.zeros((8, width), dtype=np.float64)
        self.gradient = np.zeros_like(self.activity)
        self.zero = np.zeros_like(self.activity, dtype=np.int64)

    def update(self, ids, activity, gradient):
        ids = np.asarray(ids)
        a, g = np.asarray(activity), np.asarray(gradient)
        if a.shape != g.shape or a.shape != (len(ids), self.activity.shape[1]):
            raise ValueError('shape mismatch')
        if not np.isfinite(a).all() or not np.isfinite(g).all():
            raise ValueError('nonfinite observations')
        if not np.isin(ids, np.arange(1, 8)).all():
            raise ValueError('invalid pattern')
        for p in np.unique(ids):
            selected = ids == p
            self.count[p] += selected.sum()
            self.activity[p] += np.abs(a[selected]).sum(0)
            self.gradient[p] += np.abs(g[selected]).sum(0)
            self.zero[p] += (a[selected] == 0).sum(0)

    def report(self, patterns, min_samples=32):
        a = self.activity / np.maximum(self.count[:, None], 1)
        g = self.gradient / np.maximum(self.count[:, None], 1)
        na = a / np.maximum(a.mean(1, keepdims=True), 1e-30)
        ng = g / np.maximum(g.mean(1, keepdims=True), 1e-30)
        enough = bool((self.count[patterns] >= min_samples).all())
        thresholds = {}
        for threshold in (.01, .05, .1):
            low = (na < threshold) & (ng < threshold)
            indices = np.flatnonzero(low[patterns].all(0)).tolist() if enough else None
            thresholds[str(threshold)] = dict(all_conditions_indices=indices,
                all_conditions_count=len(indices) if enough else None,
                per_pattern_count={str(p): int(low[p].sum()) if self.count[p] >= min_samples else None for p in patterns})
        return dict(count=self.count.tolist(), min_samples=min_samples,
            enough_support=enough, thresholds=thresholds,
            mean_abs_activity=a.tolist(), mean_abs_gradient=g.tolist(),
            relative_activity=na.tolist(), relative_gradient=ng.tolist(),
            exact_zero_channels={str(p): int((self.zero[p] == self.count[p]).sum()) if self.count[p] else None for p in patterns})
