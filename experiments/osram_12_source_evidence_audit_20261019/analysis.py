"""Pure NumPy paired statistics for the frozen 12-Source Evidence Audit.

Rows are read destinations, columns are joint-write Value-slot sources. Their
norms are not semantic attribution. Removed-source and coalition predictions
are frozen final-readout interventions, potentially off the training manifold.
No labels enter model inputs; labels are used only here for offline scoring.
"""
from itertools import combinations

import numpy as np


ROWS = ("Base", "Gap-A", "Gap-T", "Gap-V")
SOURCES = ("A", "T", "V")
COALITIONS = ("none", "A", "T", "AT", "V", "AV", "TV", "ATV")
EPS = 1e-12


def _class_ids(classes):
    ids = np.arange(classes) if isinstance(classes, (int, np.integer)) else np.asarray(classes)
    if ids.ndim != 1 or ids.size < 2 or len(np.unique(ids)) != ids.size:
        raise ValueError("classes must specify at least two distinct class IDs")
    return ids


def _cross_entropy(labels, logits, classes):
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    log_normalizer = np.log(np.exp(shifted).sum(axis=-1))
    columns = np.argmax(labels[:, None] == classes[None, :], axis=1)
    return log_normalizer - shifted[np.arange(len(labels)), columns]


def _metrics(labels, logits, classes):
    n = len(labels)
    if not n:
        return dict(N=0, ACC=None, UA=None, WF1=None, CE=None,
                    class_recall=[None] * len(classes), class_support=[0] * len(classes))
    prediction = classes[np.argmax(logits, axis=-1)]
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for index, class_id in enumerate(classes):
        mask = labels == class_id
        confusion[index] = [(prediction[mask] == candidate).sum() for candidate in classes]
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    correct = np.diag(confusion)
    recall = np.divide(correct, support, out=np.zeros(len(classes)), where=support > 0)
    f1 = np.divide(2 * correct, support + predicted, out=np.zeros(len(classes)),
                   where=(support + predicted) > 0)
    return dict(N=n, ACC=float(100 * correct.sum() / n), UA=float(100 * recall.mean()),
                WF1=float(100 * np.dot(support / n, f1)),
                CE=float(_cross_entropy(labels, logits, classes).mean()),
                class_recall=(100 * recall).tolist(), class_support=support.tolist())


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return dict(N=0, mean=None, std=None, min=None, q25=None, median=None, q75=None, max=None)
    return dict(N=int(values.size), mean=float(values.mean()), std=float(values.std()),
                min=float(values.min()), q25=float(np.quantile(values, .25)),
                median=float(np.median(values)), q75=float(np.quantile(values, .75)),
                max=float(values.max()))


def _paired(labels, full, intervened, classes):
    full_metrics = _metrics(labels, full, classes)
    intervention_metrics = _metrics(labels, intervened, classes)
    if len(labels):
        full_correct = classes[full.argmax(axis=-1)] == labels
        intervention_correct = classes[intervened.argmax(axis=-1)] == labels
        corrections = int((~full_correct & intervention_correct).sum())
        harms = int((full_correct & ~intervention_correct).sum())
        unchanged_correct = int((full_correct & intervention_correct).sum())
        unchanged_wrong = int((~full_correct & ~intervention_correct).sum())
    else:
        corrections = harms = unchanged_correct = unchanged_wrong = None
    return dict(N=len(labels), full_metrics=full_metrics, removed_metrics=intervention_metrics,
                **{f"delta_{key}": (intervention_metrics[key] - full_metrics[key]) if len(labels) else None
                   for key in ("ACC", "UA", "WF1", "CE")},
                corrections=corrections, harms=harms, both_correct=unchanged_correct,
                both_wrong=unchanged_wrong)


def _validate(arrays, classes):
    required = ("labels", "logits_full", "logits_removed", "logits_coalitions", "source_reads",
                "row_active", "trajectory", "state_source_norms", "state_joint_norm", "availability",
                "conversation_ids", "utterance_indices")
    data = {name: np.asarray(arrays[name]) for name in required}
    n, c = len(data["labels"]), len(classes)
    shapes = dict(labels=(n,), logits_full=(n, c), logits_removed=(n, 4, 3, c),
                  logits_coalitions=(n, 8, c), row_active=(n, 4), trajectory=(n, 3, 4),
                  state_source_norms=(n, 3), state_joint_norm=(n,), availability=(n, 3),
                  conversation_ids=(n,), utterance_indices=(n,))
    for name, shape in shapes.items():
        if data[name].shape != shape:
            raise ValueError(f"{name} shape {data[name].shape} != {shape}")
    if data["source_reads"].ndim != 4 or data["source_reads"].shape[:3] != (n, 4, 3):
        raise ValueError("source_reads must have shape [N,4,3,D]")
    for name in required:
        if name not in ("conversation_ids",) and not np.isfinite(data[name]).all():
            raise ValueError(f"{name} must be finite")
    if not np.isin(data["labels"], classes).all():
        raise ValueError("labels not in explicit classes")
    if not np.isin(data["availability"], [0, 1]).all() or np.any(data["availability"].sum(axis=1) < 1):
        raise ValueError("availability must be binary, with at least one observed modality")
    expected_active = np.column_stack((np.ones(n, dtype=bool), ~data["availability"].astype(bool)))
    if not np.array_equal(data["row_active"], expected_active):
        raise ValueError("row_active must enable Base and only currently missing Gap rows")
    if not np.allclose(data["logits_coalitions"][:, 7], data["logits_full"], rtol=1e-6, atol=1e-6):
        raise ValueError("full coalition logits do not reproduce original full logits")
    trajectory = data["trajectory"]
    if (np.any(trajectory[:, :, 0] < 0) or np.any(trajectory[:, :, 1] < 0)
            or np.any(trajectory[:, :, 1] > 1) or not np.isin(trajectory[:, :, 3], [0, 1]).all()
            or not np.array_equal(trajectory[:, :, 0] > 0, trajectory[:, :, 3].astype(bool))):
        raise ValueError("invalid count/coverage/ever_seen trajectory")
    for name in ("state_source_norms", "state_joint_norm"):
        if np.any(data[name] < 0):
            raise ValueError(f"{name} must be nonnegative")
    return data


def _shapley_values(ce):
    """Exact three-player Shapley for value=-CE; bit order A1/T2/V4."""
    value = -ce
    result = np.zeros((len(ce), 3), dtype=np.float64)
    for source in range(3):
        other = [index for index in range(3) if index != source]
        for size in range(3):
            coefficient = (1 / 3, 1 / 6, 1 / 3)[size]
            for subset in combinations(other, size):
                mask = sum(1 << index for index in subset)
                result[:, source] += coefficient * (value[:, mask | (1 << source)] - value[:, mask])
    return result


def analyze_arrays(arrays, classes):
    """Summarize one fixed checkpoint/condition, without training or inference.

    Metrics use percent; removal deltas are intervention minus original. UA is
    macro recall over *all explicit classes* (unsupported classes recall=0).
    Read-strength, cancellation, and removal statistics exclude inactive rows.
    Coverage and ever_seen come from strictly past availability supplied by the
    runner; this function never derives history from future observations.
    """
    classes = _class_ids(classes)
    data = _validate(arrays, classes)
    labels, full = data["labels"], data["logits_full"]
    active = data["row_active"].astype(bool)
    norms = np.linalg.norm(data["source_reads"].astype(np.float64), axis=-1)
    norm_sum = norms.sum(axis=-1)
    joint_norm = np.linalg.norm(data["source_reads"].sum(axis=2, dtype=np.float64), axis=-1)
    cancellation = np.clip(1 - joint_norm / np.maximum(norm_sum, EPS), 0, 1)
    ce = np.column_stack([_cross_entropy(labels, data["logits_coalitions"][:, index], classes)
                          for index in range(8)]) if len(labels) else np.empty((0, 8))
    shapley = _shapley_values(ce)

    def cohort(mask):
        n = int(mask.sum())
        slots, cancelled = {}, {}
        for row_index, row in enumerate(ROWS):
            eligible = mask & active[:, row_index]
            slots[row] = {}
            for source_index, source in enumerate(SOURCES):
                item = _paired(labels[eligible], full[eligible],
                               data["logits_removed"][eligible, row_index, source_index], classes)
                item["read_norm"] = _summary(norms[eligible, row_index, source_index])
                slots[row][source] = item
            cancelled[row] = _summary(cancellation[mask & active[:, row_index]
                                                  & (norm_sum[:, row_index] > EPS), row_index])
        gain = ce[mask, 0] - ce[mask, 7]
        credits = shapley[mask]
        return dict(N=n, full_metrics=_metrics(labels[mask], full[mask], classes), slots=slots,
                    cancellation=cancelled,
                    source_states={source: _summary(data["state_source_norms"][mask, index])
                                   for index, source in enumerate(SOURCES)},
                    joint_state_norm=_summary(data["state_joint_norm"][mask]),
                    coalitions={name: _metrics(labels[mask], data["logits_coalitions"][mask, index], classes)
                                for index, name in enumerate(COALITIONS)},
                    shapley=dict(mean={source: float(credits[:, index].mean()) if n else None
                                       for index, source in enumerate(SOURCES)},
                                 summary={source: _summary(credits[:, index])
                                          for index, source in enumerate(SOURCES)},
                                 full_minus_zero_gain=float(gain.mean()) if n else None,
                                 efficiency_max_abs_error=float(np.abs(credits.sum(axis=1) - gain).max()) if n else None))

    output = cohort(np.ones(len(labels), dtype=bool))
    current_bits = data["availability"].astype(int) @ np.array([1, 2, 4])
    ever = data["trajectory"][:, :, 3].astype(bool)
    history_bits = ever.astype(int) @ np.array([1, 2, 4])
    strata = dict(availability={COALITIONS[mask]: cohort(current_bits == mask) for mask in range(1, 8)},
                  history_ever={COALITIONS[mask]: cohort(history_bits == mask) for mask in range(8)},
                  source_history={})
    for source_index, source in enumerate(SOURCES):
        seen = ever[:, source_index]
        coverage = data["trajectory"][:, source_index, 1]
        bins = {"never": ~seen}
        for lower, upper, name in ((0, .25, "(0,0.25]"), (.25, .5, "(0.25,0.5]"),
                                   (.5, .75, "(0.5,0.75]"), (.75, 1, "(0.75,1]")):
            bins[name] = seen & (coverage > lower) & (coverage <= upper)
        strata["source_history"][source] = dict(
            ever_seen={"never": cohort(~seen), "seen": cohort(seen)},
            coverage={name: cohort(mask) for name, mask in bins.items()})
    output["strata"] = strata
    output["definitions"] = dict(
        metrics="ACC/UA/WF1 in percent; UA uses every explicit class, absent-class recall=0; CE in nats",
        delta="removed minus original on exactly the same active-row cohort",
        corrections_harms="removed prediction vs original prediction; corrections are original-wrong -> removed-correct",
        shapley="Exact A/T/V source-group Shapley; payoff=-per-utterance CE; A1/T2/V4",
        source="Joint Block Write Value-slot source contribution, not pure modality semantics or causal attribution",
        limitation="Frozen final-readout zeroing can be out-of-distribution; norms and cancellation are not sentiment conflict",
        inactive="Inactive Gap and empty cohorts are N/A (None), never evidence of zero contribution")
    return output
