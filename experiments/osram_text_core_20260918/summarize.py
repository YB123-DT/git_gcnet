"""Summarize the Text-Core MOSI diagnostic against the existing no-JEPA runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path("/data2/yb/remote_experiments/osram_text_core_20260918")
BASE = Path("/data2/yb/remote_experiments/osram_causal_nojepa_20260910/mosi")
SEEDS = (66, 67, 68, 69, 70)
RATES = tuple(f"{value / 10:.1f}" for value in range(8))
PATTERNS = {1: "V", 2: "T", 3: "TV", 4: "A", 5: "AV", 6: "AT", 7: "ATV"}
PATTERN_ORDER = ("A", "T", "V", "AT", "AV", "TV", "ATV")
GROUP_PATTERNS = {
    "T-missing": {"A", "V", "AV"},
    "T-present": {"T", "AT", "TV", "ATV"},
}


def read_json(path: Path):
    return json.loads(path.read_text())


def mean_std(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1)) if values.size > 1 else 0.0


def weighted_binary_f1(labels, predictions):
    """MOSI W-F1 on non-zero labels, matching the task metric contract."""
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    selected = np.isfinite(labels) & np.isfinite(predictions) & (labels != 0)
    if not selected.any():
        return None
    return float(
        f1_score(
            labels[selected] > 0,
            predictions[selected] > 0,
            average="weighted",
            zero_division=0,
        )
    )


def old_centered_cosine(prediction, target):
    """Match the previous audit: center channels, then mean per-sample cosine."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[0] == 0:
        return None
    prediction = prediction - prediction.mean(axis=0, keepdims=True)
    target = target - target.mean(axis=0, keepdims=True)
    numerator = np.sum(prediction * target, axis=1)
    denominator = np.linalg.norm(prediction, axis=1) * np.linalg.norm(target, axis=1)
    cosine = numerator / np.maximum(denominator, 1e-8)
    return float(np.mean(cosine))


def old_std_ratio(prediction, target):
    """Match the previous audit's mean-channel population-std ratio."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[0] == 0:
        return None
    return float(
        prediction.std(axis=0, ddof=0).mean()
        / (target.std(axis=0, ddof=0).mean() + 1e-8)
    )


def load_predictions(root, seed, rate):
    path = root / f"seed_{seed}" / f"predictions_miss_{rate.replace('.', 'p')}.npz"
    with np.load(path) as data:
        return {name: data[name].copy() for name in data.files}


def rate_rows():
    rows = []
    for rate in RATES:
        for seed in SEEDS:
            c0 = read_json(BASE / f"seed_{seed}" / "metrics.json")["test"][rate]
            c1 = read_json(ROOT / "mosi" / f"seed_{seed}" / "metrics.json")["test"][rate]
            rows.append(
                {
                    "seed": seed,
                    "rate": rate,
                    "no_jepa_wf1": c0["weighted_f1"],
                    "text_core_wf1": c1["weighted_f1"],
                    "delta": c1["weighted_f1"] - c0["weighted_f1"],
                    "selected_epoch": read_json(ROOT / "mosi" / f"seed_{seed}" / "metrics.json").get("selected_epoch_by_rate", {}).get(rate),
                    "text_core_centered_cosine": c1.get("text_core_centered_cosine"),
                    "text_core_pred_target_std_ratio": c1.get("text_core_pred_target_std_ratio"),
                }
            )
    return rows


def pattern_rows():
    rows = []
    group_samples = {
        variant: {
            group: {"labels": [], "predictions": []}
            for group in GROUP_PATTERNS
        }
        for variant in ("no-JEPA", "Text-Core")
    }
    pattern_samples = {
        variant: {
            name: {"labels": [], "predictions": []}
            for name in PATTERN_ORDER
        }
        for variant in ("no-JEPA", "Text-Core")
    }
    for variant, root in (("no-JEPA", BASE), ("Text-Core", ROOT / "mosi")):
        for seed in SEEDS:
            for rate in RATES:
                data = load_predictions(root, seed, rate)
                availability = data["availability"]
                labels = data["labels"]
                predictions = data["predictions"]
                pattern_ids = (
                    availability[:, 0].astype(int) * 4
                    + availability[:, 1].astype(int) * 2
                    + availability[:, 2].astype(int)
                )
                for pattern_id, name in PATTERNS.items():
                    all_selected = pattern_ids == pattern_id
                    selected = all_selected & np.isfinite(labels) & (labels != 0)
                    if not selected.any():
                        continue
                    binary_labels = labels[selected] > 0
                    binary_predictions = predictions[selected] > 0
                    for group, names in GROUP_PATTERNS.items():
                        if name in names:
                            group_samples[variant][group]["labels"].append(labels[selected])
                            group_samples[variant][group]["predictions"].append(predictions[selected])
                    pattern_samples[variant][name]["labels"].append(labels[selected])
                    pattern_samples[variant][name]["predictions"].append(predictions[selected])
                    rows.append(
                        {
                            "variant": variant,
                            "seed": seed,
                            "rate": rate,
                            "pattern": name,
                            "total_count": int(all_selected.sum()),
                            "excluded_label_zero": int((all_selected & (labels == 0)).sum()),
                            "count": int(selected.sum()),
                            "weighted_f1": float(f1_score(binary_labels, binary_predictions, average="weighted", zero_division=0)),
                        }
                    )
    return rows, group_samples, pattern_samples


def write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    rows = rate_rows()
    write_csv(ROOT / "per_seed_rate.csv", rows)
    p_rows, group_samples, pattern_samples = pattern_rows()
    write_csv(ROOT / "pattern_per_seed.csv", p_rows)

    summary = {
        "rate": [], "overall": {}, "high_missing": {}, "text_groups": {},
        "patterns": {}, "task_slots": {},
        "metric_contract": {
            "pattern_wf1": "exclude label==0 before binary W-F1",
            "group_wf1": "sample-pooled over all valid samples, excluding label==0",
            "centered_cosine": "old protocol: channel-center across samples, then mean per-sample cosine",
            "std_ratio": "old mean-channel population-std ratio, computed only on T-missing samples",
        },
    }
    for rate in RATES:
        current = [row for row in rows if row["rate"] == rate]
        c0_mean, c0_std = mean_std([row["no_jepa_wf1"] for row in current])
        c1_mean, c1_std = mean_std([row["text_core_wf1"] for row in current])
        summary["rate"].append(
            {
                "rate": rate,
                "no_jepa_mean": c0_mean,
                "no_jepa_std": c0_std,
                "text_core_mean": c1_mean,
                "text_core_std": c1_std,
                "delta": c1_mean - c0_mean,
                "positive_seeds": sum(row["delta"] > 0 for row in current),
            }
        )
    for name, selected in (
        ("overall", rows),
        ("high_missing", [row for row in rows if row["rate"] in {"0.5", "0.6", "0.7"}]),
    ):
        summary[name] = {
            "no_jepa_mean": float(np.mean([row["no_jepa_wf1"] for row in selected])),
            "text_core_mean": float(np.mean([row["text_core_wf1"] for row in selected])),
            "delta": float(np.mean([row["delta"] for row in selected])),
        }
    for pattern_name in PATTERN_ORDER:
        summary["patterns"][pattern_name] = {}
        for variant in ("no-JEPA", "Text-Core"):
            sample_labels = np.concatenate(pattern_samples[variant][pattern_name]["labels"])
            sample_predictions = np.concatenate(pattern_samples[variant][pattern_name]["predictions"])
            per_group = [
                row["weighted_f1"]
                for row in p_rows
                if row["variant"] == variant and row["pattern"] == pattern_name
            ]
            macro_mean, macro_std = mean_std(per_group)
            summary["patterns"][pattern_name][variant] = {
                "pattern_macro_mean": macro_mean,
                "pattern_macro_std": macro_std,
                "sample_pooled_wf1": weighted_binary_f1(sample_labels, sample_predictions),
                "sample_count": int(sample_labels.size),
            }
    for group, names in GROUP_PATTERNS.items():
        summary["text_groups"][group] = {
            "pattern_macro": {},
            "sample_pooled": {},
            "sample_count": {},
        }
        for variant in ("no-JEPA", "Text-Core"):
            pattern_means = [
                summary["patterns"][name][variant]["pattern_macro_mean"]
                for name in PATTERN_ORDER
                if name in names
            ]
            labels = np.concatenate(group_samples[variant][group]["labels"])
            predictions = np.concatenate(group_samples[variant][group]["predictions"])
            summary["text_groups"][group]["pattern_macro"][variant] = float(np.mean(pattern_means))
            summary["text_groups"][group]["sample_pooled"][variant] = weighted_binary_f1(labels, predictions)
            summary["text_groups"][group]["sample_count"][variant] = int(labels.size)
    slot_values = {"real": [], "predicted": [], "centered_cosine": [], "std_ratio": []}
    for seed in SEEDS:
        metrics = read_json(ROOT / "mosi" / f"seed_{seed}" / "metrics.json")["test"]
        for rate in RATES:
            current = metrics[rate]
            for key, target in (("text_core_real_slot", "real"), ("text_core_predicted_slot", "predicted")):
                if current.get(key) is not None:
                    slot_values[target].append(current[key]["weighted_f1"])
            data = load_predictions(ROOT / "mosi", seed, rate)
            missing_text = ~data["text_core_text_observed"].astype(bool)
            if int(missing_text.sum()) >= 2:
                prediction_u = data["text_core_pred_u"][missing_text]
                target_u = data["text_core_target_u"][missing_text]
                slot_values["centered_cosine"].append(old_centered_cosine(prediction_u, target_u))
                slot_values["std_ratio"].append(old_std_ratio(prediction_u, target_u))
    summary["task_slots"] = {
        "real_wf1_mean": float(np.mean(slot_values["real"])),
        "predicted_wf1_mean": float(np.mean(slot_values["predicted"])),
        "centered_cosine_mean": float(np.mean(slot_values["centered_cosine"])),
        "std_ratio_mean": float(np.mean(slot_values["std_ratio"])),
        "centered_cosine_std": float(np.std(slot_values["centered_cosine"], ddof=1)),
        "std_ratio_std": float(np.std(slot_values["std_ratio"], ddof=1)),
        "centered_cosine_group_count": len(slot_values["centered_cosine"]),
        "std_ratio_group_count": len(slot_values["std_ratio"]),
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    lines = [
        "# Text-Core MOSI diagnostic",
        "",
        "Internal diagnostic only; each seed × missing rate selects its own Test W-F1 epoch.",
        "",
        "## Per-rate weighted F1",
        "",
        "| rate | no-JEPA | Text-Core | delta | positive seeds |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rate"]:
        lines.append(
            f"| {row['rate']} | {row['no_jepa_mean'] * 100:.2f}±{row['no_jepa_std'] * 100:.2f} | "
            f"{row['text_core_mean'] * 100:.2f}±{row['text_core_std'] * 100:.2f} | "
            f"{row['delta'] * 100:+.2f} | {row['positive_seeds']}/5 |"
        )
    lines.extend(
        [
            "",
            f"8-rate mean: no-JEPA {summary['overall']['no_jepa_mean'] * 100:.2f}%, Text-Core {summary['overall']['text_core_mean'] * 100:.2f}%.",
            f"High-missing (.5/.6/.7): no-JEPA {summary['high_missing']['no_jepa_mean'] * 100:.2f}%, Text-Core {summary['high_missing']['text_core_mean'] * 100:.2f}%.",
            "",
            "## T-missing vs T-present",
            "",
            "W-F1 excludes label==0. `pattern-macro` first averages valid W-F1 within each pattern, then averages patterns; `sample-pooled` concatenates the underlying valid samples before computing W-F1.",
            "",
            "| group | aggregation | no-JEPA | Text-Core | delta | valid samples (no-JEPA/Text-Core) |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for group, values in summary["text_groups"].items():
        for aggregation in ("pattern_macro", "sample_pooled"):
            no_jepa = values[aggregation]["no-JEPA"]
            text_core = values[aggregation]["Text-Core"]
            lines.append(
                f"| {group} | {aggregation} | {no_jepa * 100:.2f} | {text_core * 100:.2f} | "
                f"{(text_core - no_jepa) * 100:+.2f} | "
                f"{values['sample_count']['no-JEPA']}/{values['sample_count']['Text-Core']} |"
            )
    lines.extend(["", "## Seven-pattern summary", "", "| pattern | aggregation | no-JEPA | Text-Core | delta |", "|---|---|---:|---:|---:|"])
    for pattern_name in PATTERN_ORDER:
        for aggregation, key in (("pattern-macro", "pattern_macro_mean"), ("sample-pooled", "sample_pooled_wf1")):
            no_jepa = summary["patterns"][pattern_name]["no-JEPA"][key]
            text_core = summary["patterns"][pattern_name]["Text-Core"][key]
            lines.append(f"| {pattern_name} | {aggregation} | {no_jepa * 100:.2f} | {text_core * 100:.2f} | {(text_core - no_jepa) * 100:+.2f} |")
    lines.extend(
        [
            "",
            "## Task-slot diagnostics",
            "",
            f"Real Text slot W-F1: {summary['task_slots']['real_wf1_mean'] * 100:.2f}%",
            f"Predicted Text slot W-F1: {summary['task_slots']['predicted_wf1_mean'] * 100:.2f}%",
            f"Centered cosine (old per-sample protocol; T-missing only): {summary['task_slots']['centered_cosine_mean']:.4f}±{summary['task_slots']['centered_cosine_std']:.4f}",
            f"Prediction/target std ratio (T-missing only; old mean-channel population-std protocol): {summary['task_slots']['std_ratio_mean']:.4f}±{summary['task_slots']['std_ratio_std']:.4f}",
            "",
            "`pattern_per_seed.csv` excludes label==0 from `count` and W-F1 and records the excluded count explicitly.",
        ]
    )
    (ROOT / "RESULT.md").write_text("\n".join(lines) + "\n")
    print((ROOT / "RESULT.md").read_text())


if __name__ == "__main__":
    main()
