"""Summarize the MOSI binary-task no-JEPA diagnostic."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path("/data2/yb/remote_experiments/osram_mosi_binary_20260918")
BASE = Path("/data2/yb/remote_experiments/osram_causal_nojepa_20260910/mosi")
SEEDS = (66, 67, 68, 69, 70)
RATES = tuple(f"{value / 10:.1f}" for value in range(8))
PATTERNS = {1: "V", 2: "T", 3: "TV", 4: "A", 5: "AV", 6: "AT", 7: "ATV"}
PATTERN_ORDER = ("A", "T", "V", "AT", "AV", "TV", "ATV")
GROUP_PATTERNS = {"T-missing": {"A", "V", "AV"}, "T-present": {"T", "AT", "TV", "ATV"}}


def read_json(path: Path):
    return json.loads(path.read_text())


def mean_std(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1)) if values.size > 1 else 0.0


def load_predictions(root: Path, seed: int, rate: str):
    path = root / f"seed_{seed}" / f"predictions_miss_{rate.replace('.', 'p')}.npz"
    with np.load(path) as data:
        return {name: data[name].copy() for name in data.files}


def binary_wf1(labels, predictions, *, filter_zero=True):
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    selected = np.isfinite(labels) & np.isfinite(predictions)
    if filter_zero:
        selected &= labels != 0
    if not selected.any():
        return None
    return float(
        f1_score(labels[selected] > 0, predictions[selected] > 0, average="weighted", zero_division=0)
    )


def rate_rows():
    rows = []
    for rate in RATES:
        for seed in SEEDS:
            baseline = read_json(BASE / f"seed_{seed}" / "metrics.json")["test"][rate]
            binary_metrics = read_json(ROOT / "mosi" / f"seed_{seed}" / "metrics.json")
            current = binary_metrics["test"][rate]
            rows.append(
                {
                    "seed": seed,
                    "rate": rate,
                    "no_jepa_regression_wf1": baseline["weighted_f1"],
                    "binary_wf1": current["weighted_f1"],
                    "delta": current["weighted_f1"] - baseline["weighted_f1"],
                    "selected_epoch": binary_metrics.get("selected_epoch_by_rate", {}).get(rate),
                }
            )
    return rows


def pattern_rows():
    rows = []
    stores = {
        variant: {name: {"labels": [], "predictions": [], "count": 0, "excluded": 0} for name in PATTERN_ORDER}
        for variant in ("no-JEPA-regression", "Binary")
    }
    for variant, root in (("no-JEPA-regression", BASE), ("Binary", ROOT / "mosi")):
        for seed in SEEDS:
            for rate in RATES:
                data = load_predictions(root, seed, rate)
                availability = data["availability"]
                metric_labels = data["labels"]
                # Binary evaluation stores class labels (0/1) in ``labels``;
                # neutral MOSI labels must be filtered using the preserved
                # continuous labels, otherwise every negative example would
                # be mistaken for label==0 and removed from the audit.
                selection_labels = (
                    data.get("continuous_labels", metric_labels)
                    if variant == "Binary"
                    else metric_labels
                )
                predictions = data["predictions"]
                pattern_ids = (
                    availability[:, 0].astype(int) * 4
                    + availability[:, 1].astype(int) * 2
                    + availability[:, 2].astype(int)
                )
                for pattern_id, name in PATTERNS.items():
                    selected = pattern_ids == pattern_id
                    if not selected.any():
                        continue
                    stores[variant][name]["count"] += int(selected.sum())
                    stores[variant][name]["excluded"] += int((selected & (selection_labels == 0)).sum())
                    kept = selected & (selection_labels != 0)
                    stores[variant][name]["labels"].extend(metric_labels[kept].tolist())
                    stores[variant][name]["predictions"].extend(predictions[kept].tolist())
                    score = binary_wf1(metric_labels[kept], predictions[kept], filter_zero=False)
                    if score is not None:
                        rows.append(
                            {
                                "variant": variant,
                                "seed": seed,
                                "rate": rate,
                                "pattern": name,
                                "total_count": int(selected.sum()),
                                "excluded_label_zero": int((selected & (selection_labels == 0)).sum()),
                                "count": int(kept.sum()),
                                "weighted_f1": score,
                            }
                        )
    return rows, stores


def aggregate_store(store, names):
    labels = np.asarray([value for name in names for value in store[name]["labels"]])
    predictions = np.asarray([value for name in names for value in store[name]["predictions"]])
    sample_pooled = binary_wf1(labels, predictions, filter_zero=False)
    pattern_scores = [
        binary_wf1(store[name]["labels"], store[name]["predictions"], filter_zero=False)
        for name in names
        if store[name]["labels"]
    ]
    return {
        "pattern_macro": float(np.mean(pattern_scores)) if pattern_scores else None,
        "sample_pooled": sample_pooled,
        "sample_count": int(labels.size),
    }


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with path.open("w", newline="") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows = rate_rows()
    pattern_data, stores = pattern_rows()
    per_rate = []
    for rate in RATES:
        current = [row["binary_wf1"] for row in rows if row["rate"] == rate]
        baseline = [row["no_jepa_regression_wf1"] for row in rows if row["rate"] == rate]
        m0, s0 = mean_std(baseline)
        m1, s1 = mean_std(current)
        per_rate.append(
            {
                "rate": rate,
                "no_jepa_regression_mean": m0,
                "no_jepa_regression_std": s0,
                "binary_mean": m1,
                "binary_std": s1,
                "delta": m1 - m0,
                "positive_seeds": sum(value > base for value, base in zip(current, baseline)),
            }
        )
    def grouped(rates):
        by_seed = []
        for seed in SEEDS:
            current = [row["binary_wf1"] for row in rows if row["seed"] == seed and row["rate"] in rates]
            baseline = [row["no_jepa_regression_wf1"] for row in rows if row["seed"] == seed and row["rate"] in rates]
            by_seed.append((float(np.mean(baseline)), float(np.mean(current))))
        return {
            "no_jepa_mean": float(np.mean([pair[0] for pair in by_seed])),
            "binary_mean": float(np.mean([pair[1] for pair in by_seed])),
            "delta": float(np.mean([pair[1] - pair[0] for pair in by_seed])),
            "positive_seeds": sum(pair[1] > pair[0] for pair in by_seed),
        }
    text_groups = {}
    for group, names in GROUP_PATTERNS.items():
        text_groups[group] = {
            variant: aggregate_store(stores[variant], names)
            for variant in ("no-JEPA-regression", "Binary")
        }
    summary = {
        "status": "complete",
        "selection_protocol": "per-rate-test-oracle",
        "configuration_delta": {"mosi_task_mode": ["regression", "binary"]},
        "rate": per_rate,
        "overall": grouped(RATES),
        "high_missing": grouped(("0.5", "0.6", "0.7")),
        "text_groups": text_groups,
        "patterns": {
            name: {
                variant: aggregate_store(stores[variant], (name,))
                for variant in ("no-JEPA-regression", "Binary")
            }
            for name in PATTERN_ORDER
        },
        "pattern_rows": pattern_data,
    }
    write_csv(ROOT / "results" / "per_seed_rate.csv", rows)
    write_csv(ROOT / "results" / "pattern_per_seed.csv", pattern_data)
    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    lines = [
        "# MOSI binary-task no-JEPA diagnostic",
        "",
        "Internal diagnostic only; the sole configuration change from the inherited no-JEPA causal OSRAM is `mosi_task_mode: regression -> binary`.",
        "Test masks, cyclic training schedule, eta=0.6, Flat readout, optimizer, features, and per-seed × per-rate Test-oracle selection are unchanged.",
        "",
        "## Per-rate weighted F1",
        "",
        "| rate | no-JEPA regression | binary | delta | positive seeds |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in per_rate:
        lines.append(
            f"| {row['rate']} | {row['no_jepa_regression_mean']*100:.2f}±{row['no_jepa_regression_std']*100:.2f} | {row['binary_mean']*100:.2f}±{row['binary_std']*100:.2f} | {row['delta']*100:+.2f} | {row['positive_seeds']}/5 |"
        )
    lines += [
        "",
        f"8-rate mean: no-JEPA regression {summary['overall']['no_jepa_mean']*100:.2f}%, binary {summary['overall']['binary_mean']*100:.2f}% ({summary['overall']['delta']*100:+.2f} pp).",
        f"High-missing (.5/.6/.7): no-JEPA regression {summary['high_missing']['no_jepa_mean']*100:.2f}%, binary {summary['high_missing']['binary_mean']*100:.2f}% ({summary['high_missing']['delta']*100:+.2f} pp).",
        "",
        "## T-missing vs T-present",
        "",
        "Scores exclude label==0. Pattern-macro averages the three/four pattern scores; sample-pooled concatenates all valid samples.",
        "",
        "| group | aggregation | no-JEPA regression | binary | delta |",
        "|---|---|---:|---:|---:|",
    ]
    for group, values in text_groups.items():
        for aggregation in ("pattern_macro", "sample_pooled"):
            c0 = values["no-JEPA-regression"][aggregation]
            c1 = values["Binary"][aggregation]
            lines.append(f"| {group} | {aggregation} | {c0*100:.2f} | {c1*100:.2f} | {(c1-c0)*100:+.2f} |")
    lines += ["", "## Seven-pattern sample-pooled W-F1", "", "| pattern | no-JEPA regression | binary | delta |", "|---|---:|---:|---:|"]
    for name in PATTERN_ORDER:
        c0 = summary["patterns"][name]["no-JEPA-regression"]["sample_pooled"]
        c1 = summary["patterns"][name]["Binary"]["sample_pooled"]
        lines.append(f"| {name} | {c0*100:.2f} | {c1*100:.2f} | {(c1-c0)*100:+.2f} |")
    (ROOT / "RESULT.md").write_text("\n".join(lines) + "\n")
    print((ROOT / "RESULT.md").read_text())


if __name__ == "__main__":
    main()
