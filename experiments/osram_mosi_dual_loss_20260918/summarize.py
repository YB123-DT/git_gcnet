"""Summarize the MOSI regression-plus-sign-classification diagnostic."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from experiments.osram_mosi_binary_20260918.summarize import (
    aggregate_store,
    binary_wf1,
    load_predictions,
    mean_std,
    read_json,
    write_csv,
)

ROOT = Path("/data2/yb/remote_experiments/osram_mosi_dual_loss_20260918")
BASE = Path("/data2/yb/remote_experiments/osram_causal_nojepa_20260910/mosi")
SEEDS = (66, 67, 68, 69, 70)
RATES = tuple(f"{value / 10:.1f}" for value in range(8))
PATTERNS = {1: "V", 2: "T", 3: "TV", 4: "A", 5: "AV", 6: "AT", 7: "ATV"}
PATTERN_ORDER = ("A", "T", "V", "AT", "AV", "TV", "ATV")
GROUP_PATTERNS = {"T-missing": {"A", "V", "AV"}, "T-present": {"T", "AT", "TV", "ATV"}}


def rate_rows():
    rows = []
    for rate in RATES:
        for seed in SEEDS:
            baseline = read_json(BASE / f"seed_{seed}" / "metrics.json")["test"][rate]
            current_metrics = read_json(ROOT / "mosi" / f"seed_{seed}" / "metrics.json")
            current = current_metrics["test"][rate]
            rows.append(
                {
                    "seed": seed,
                    "rate": rate,
                    "no_jepa_regression_wf1": baseline["weighted_f1"],
                    "dual_wf1": current["weighted_f1"],
                    "no_jepa_regression_accuracy": baseline["accuracy"],
                    "dual_accuracy": current["accuracy"],
                    "delta": current["weighted_f1"] - baseline["weighted_f1"],
                    "accuracy_delta": current["accuracy"] - baseline["accuracy"],
                    "selected_epoch": current_metrics.get("selected_epoch_by_rate", {}).get(rate),
                }
            )
    return rows


def pattern_rows():
    rows = []
    variants = ("no-JEPA-regression", "Dual-loss")
    stores = {
        variant: {name: {"labels": [], "predictions": [], "count": 0, "excluded": 0} for name in PATTERN_ORDER}
        for variant in variants
    }
    for variant, root in ((variants[0], BASE), (variants[1], ROOT / "mosi")):
        for seed in SEEDS:
            for rate in RATES:
                data = load_predictions(root, seed, rate)
                availability = data["availability"]
                metric_labels = data["labels"]
                selection_labels = data.get("continuous_labels", metric_labels)
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
                    kept = selected & (selection_labels != 0)
                    stores[variant][name]["count"] += int(selected.sum())
                    stores[variant][name]["excluded"] += int((selected & (selection_labels == 0)).sum())
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
                                "excluded_original_neutral": int((selected & (selection_labels == 0)).sum()),
                                "count": int(kept.sum()),
                                "weighted_f1": score,
                            }
                        )
    return rows, stores


def grouped(rows, rates):
    pairs = []
    for seed in SEEDS:
        current = [row["dual_wf1"] for row in rows if row["seed"] == seed and row["rate"] in rates]
        baseline = [row["no_jepa_regression_wf1"] for row in rows if row["seed"] == seed and row["rate"] in rates]
        pairs.append((float(np.mean(baseline)), float(np.mean(current))))
    return {
        "no_jepa_mean": float(np.mean([pair[0] for pair in pairs])),
        "dual_mean": float(np.mean([pair[1] for pair in pairs])),
        "delta": float(np.mean([pair[1] - pair[0] for pair in pairs])),
        "positive_seeds": sum(pair[1] > pair[0] for pair in pairs),
    }


def main():
    rows = rate_rows()
    pattern_data, stores = pattern_rows()
    per_rate = []
    for rate in RATES:
        current = [row["dual_wf1"] for row in rows if row["rate"] == rate]
        baseline = [row["no_jepa_regression_wf1"] for row in rows if row["rate"] == rate]
        current_acc = [row["dual_accuracy"] for row in rows if row["rate"] == rate]
        baseline_acc = [row["no_jepa_regression_accuracy"] for row in rows if row["rate"] == rate]
        m0, s0 = mean_std(baseline)
        m1, s1 = mean_std(current)
        a0, _ = mean_std(baseline_acc)
        a1, _ = mean_std(current_acc)
        per_rate.append(
            {
                "rate": rate,
                "no_jepa_regression_mean": m0,
                "no_jepa_regression_std": s0,
                "dual_mean": m1,
                "dual_std": s1,
                "delta": m1 - m0,
                "no_jepa_regression_accuracy_mean": a0,
                "dual_accuracy_mean": a1,
                "accuracy_delta": a1 - a0,
                "positive_seeds": sum(value > base for value, base in zip(current, baseline)),
            }
        )
    text_groups = {
        group: {
            variant: aggregate_store(stores[variant], names)
            for variant in ("no-JEPA-regression", "Dual-loss")
        }
        for group, names in GROUP_PATTERNS.items()
    }
    summary = {
        "status": "complete",
        "selection_protocol": "per-rate-test-oracle",
        "configuration_delta": {"mosi_task_mode": ["regression", "dual"]},
        "loss_definition": "MSE(all valid continuous labels) + BCEWithLogits(nonzero sign labels)",
        "rate": per_rate,
        "overall": grouped(rows, RATES),
        "high_missing": grouped(rows, ("0.5", "0.6", "0.7")),
        "text_groups": text_groups,
        "patterns": {
            name: {
                variant: aggregate_store(stores[variant], (name,))
                for variant in ("no-JEPA-regression", "Dual-loss")
            }
            for name in PATTERN_ORDER
        },
        "pattern_rows": pattern_data,
    }
    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    write_csv(ROOT / "results" / "per_seed_rate.csv", rows)
    write_csv(ROOT / "results" / "pattern_per_seed.csv", pattern_data)
    (ROOT / "results" / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    lines = [
        "# MOSI dual-loss diagnostic",
        "",
        "Internal diagnostic only. The scalar output is trained with MSE on all valid continuous labels plus BCEWithLogits on nonzero sign labels; test reporting uses the existing Non0 binary protocol.",
        "The causal no-JEPA OSRAM, eta=0.6, Flat readout, cyclic missing-rate schedule, features, optimizer, and per-seed × per-rate Test-oracle selection are unchanged.",
        "",
        "## Per-rate weighted F1 and accuracy",
        "",
        "| rate | regression W-F1 | dual-loss W-F1 | Δ F1 | regression Acc | dual-loss Acc | Δ Acc |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in per_rate:
        lines.append(
            f"| {row['rate']} | {row['no_jepa_regression_mean']*100:.2f}±{row['no_jepa_regression_std']*100:.2f} | {row['dual_mean']*100:.2f}±{row['dual_std']*100:.2f} | {row['delta']*100:+.2f} | {row['no_jepa_regression_accuracy_mean']*100:.2f} | {row['dual_accuracy_mean']*100:.2f} | {row['accuracy_delta']*100:+.2f} |"
        )
    lines += [
        "",
        f"8-rate mean: regression {summary['overall']['no_jepa_mean']*100:.2f}%, dual-loss {summary['overall']['dual_mean']*100:.2f}% ({summary['overall']['delta']*100:+.2f} pp).",
        f"High-missing (.5/.6/.7): regression {summary['high_missing']['no_jepa_mean']*100:.2f}%, dual-loss {summary['high_missing']['dual_mean']*100:.2f}% ({summary['high_missing']['delta']*100:+.2f} pp).",
        "",
        "## T-missing vs T-present",
        "",
        "Original continuous neutral labels are excluded; binary negative/positive labels are retained.",
        "",
        "| group | aggregation | regression | dual-loss | delta |",
        "|---|---|---:|---:|---:|",
    ]
    for group, values in text_groups.items():
        for aggregation in ("pattern_macro", "sample_pooled"):
            c0 = values["no-JEPA-regression"][aggregation]
            c1 = values["Dual-loss"][aggregation]
            lines.append(f"| {group} | {aggregation} | {c0*100:.2f} | {c1*100:.2f} | {(c1-c0)*100:+.2f} |")
    lines += ["", "## Seven-pattern sample-pooled W-F1", "", "| pattern | regression | dual-loss | delta |", "|---|---:|---:|---:|"]
    for name in PATTERN_ORDER:
        c0 = summary["patterns"][name]["no-JEPA-regression"]["sample_pooled"]
        c1 = summary["patterns"][name]["Dual-loss"]["sample_pooled"]
        lines.append(f"| {name} | {c0*100:.2f} | {c1*100:.2f} | {(c1-c0)*100:+.2f} |")
    (ROOT / "RESULT.md").write_text("\n".join(lines) + "\n")
    print((ROOT / "RESULT.md").read_text())


if __name__ == "__main__":
    main()
