"""Summarize the continuous-rate forced-Text no-JEPA diagnostic."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path("/data2/yb/remote_experiments/osram_uniform_forced_text_20260918")
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


def load_predictions(root, seed, rate):
    path = root / f"seed_{seed}" / f"predictions_miss_{rate.replace('.', 'p')}.npz"
    with np.load(path) as data:
        return {name: data[name].copy() for name in data.files}


def binary_wf1(labels, predictions):
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    selected = np.isfinite(labels) & np.isfinite(predictions) & (labels != 0)
    if not selected.any():
        return None
    return float(f1_score(labels[selected] > 0, predictions[selected] > 0, average="weighted", zero_division=0))


def rate_rows():
    rows = []
    for rate in RATES:
        for seed in SEEDS:
            c0 = read_json(BASE / f"seed_{seed}" / "metrics.json")["test"][rate]
            c1 = read_json(ROOT / "mosi" / f"seed_{seed}" / "metrics.json")["test"][rate]
            rows.append({
                "seed": seed,
                "rate": rate,
                "no_jepa_wf1": c0["weighted_f1"],
                "uniform_wf1": c1["weighted_f1"],
                "delta": c1["weighted_f1"] - c0["weighted_f1"],
                "selected_epoch": read_json(ROOT / "mosi" / f"seed_{seed}" / "metrics.json").get("selected_epoch_by_rate", {}).get(rate),
            })
    return rows


def pattern_rows():
    rows = []
    groups = {variant: {name: {"labels": [], "predictions": []} for name in GROUP_PATTERNS} for variant in ("no-JEPA", "Uniform")}
    patterns = {variant: {name: {"labels": [], "predictions": []} for name in PATTERN_ORDER} for variant in ("no-JEPA", "Uniform")}
    for variant, root in (("no-JEPA", BASE), ("Uniform", ROOT / "mosi")):
        for seed in SEEDS:
            for rate in RATES:
                data = load_predictions(root, seed, rate)
                availability = data["availability"]
                labels = data["labels"]
                predictions = data["predictions"]
                pattern_ids = availability[:, 0].astype(int) * 4 + availability[:, 1].astype(int) * 2 + availability[:, 2].astype(int)
                for pattern_id, name in PATTERNS.items():
                    all_selected = pattern_ids == pattern_id
                    selected = all_selected & np.isfinite(labels) & (labels != 0)
                    if not selected.any():
                        continue
                    for group, names in GROUP_PATTERNS.items():
                        if name in names:
                            groups[variant][group]["labels"].append(labels[selected])
                            groups[variant][group]["predictions"].append(predictions[selected])
                    patterns[variant][name]["labels"].append(labels[selected])
                    patterns[variant][name]["predictions"].append(predictions[selected])
                    rows.append({
                        "variant": variant,
                        "seed": seed,
                        "rate": rate,
                        "pattern": name,
                        "total_count": int(all_selected.sum()),
                        "excluded_label_zero": int((all_selected & (labels == 0)).sum()),
                        "count": int(selected.sum()),
                        "weighted_f1": binary_wf1(labels[selected], predictions[selected]),
                    })
    return rows, groups, patterns


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    rows = rate_rows()
    write_csv(ROOT / "per_seed_rate.csv", rows)
    pattern_data, group_data, pattern_data_raw = pattern_rows()
    write_csv(ROOT / "pattern_per_seed.csv", pattern_data)

    summary = {"rate": [], "overall": {}, "high_missing": {}, "text_groups": {}, "patterns": {}}
    for rate in RATES:
        current = [row for row in rows if row["rate"] == rate]
        c0_mean, c0_std = mean_std([row["no_jepa_wf1"] for row in current])
        c1_mean, c1_std = mean_std([row["uniform_wf1"] for row in current])
        summary["rate"].append({
            "rate": rate,
            "no_jepa_mean": c0_mean,
            "no_jepa_std": c0_std,
            "uniform_mean": c1_mean,
            "uniform_std": c1_std,
            "delta": c1_mean - c0_mean,
            "positive_seeds": sum(row["delta"] > 0 for row in current),
        })
    for name, selected in (("overall", rows), ("high_missing", [row for row in rows if row["rate"] in {"0.5", "0.6", "0.7"}])):
        summary[name] = {
            "no_jepa_mean": float(np.mean([row["no_jepa_wf1"] for row in selected])),
            "uniform_mean": float(np.mean([row["uniform_wf1"] for row in selected])),
            "delta": float(np.mean([row["delta"] for row in selected])),
        }

    for pattern in PATTERN_ORDER:
        summary["patterns"][pattern] = {}
        for variant in ("no-JEPA", "Uniform"):
            labels = np.concatenate(pattern_data_raw[variant][pattern]["labels"])
            predictions = np.concatenate(pattern_data_raw[variant][pattern]["predictions"])
            per_group = [row["weighted_f1"] for row in pattern_data if row["variant"] == variant and row["pattern"] == pattern]
            macro_mean, macro_std = mean_std(per_group)
            summary["patterns"][pattern][variant] = {
                "pattern_macro_mean": macro_mean,
                "pattern_macro_std": macro_std,
                "sample_pooled_wf1": binary_wf1(labels, predictions),
                "sample_count": int(labels.size),
            }
    for group, names in GROUP_PATTERNS.items():
        summary["text_groups"][group] = {"pattern_macro": {}, "sample_pooled": {}, "sample_count": {}}
        for variant in ("no-JEPA", "Uniform"):
            labels = np.concatenate(group_data[variant][group]["labels"])
            predictions = np.concatenate(group_data[variant][group]["predictions"])
            summary["text_groups"][group]["pattern_macro"][variant] = float(np.mean([
                summary["patterns"][name][variant]["pattern_macro_mean"] for name in PATTERN_ORDER if name in names
            ]))
            summary["text_groups"][group]["sample_pooled"][variant] = binary_wf1(labels, predictions)
            summary["text_groups"][group]["sample_count"][variant] = int(labels.size)

    train_audits = []
    for seed in SEEDS:
        history = read_json(ROOT / "mosi" / f"seed_{seed}" / "history.json")
        train_audits.extend(record["train"] for record in history)
    summary["train_mask"] = {
        "sampled_rate_mean": float(np.mean([record["uniform_sampled_rate_mean"] for record in train_audits if record.get("uniform_sampled_rate_mean") is not None])),
        "realized_missing_fraction_mean": float(np.mean([record["uniform_realized_missing_fraction"] for record in train_audits if record.get("uniform_realized_missing_fraction") is not None])),
        "forced_text_fraction_mean": float(np.mean([record["uniform_forced_text_fraction"] for record in train_audits if record.get("uniform_forced_text_fraction") is not None])),
        "target_probability": 0.25,
        "condition": "at least one of A/T/V observed for every valid utterance",
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Continuous-rate forced-Text no-JEPA MOSI diagnostic",
        "",
        "Internal diagnostic only; training samples use per-utterance r~Uniform(0,1), P(Text forced missing)=0.25, and at least one observed modality. Test checkpoints use the inherited per-seed × per-rate Test-oracle protocol.",
        "",
        "## Per-rate weighted F1",
        "",
        "| rate | no-JEPA | uniform-forced-text | delta | positive seeds |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rate"]:
        lines.append(f"| {row['rate']} | {row['no_jepa_mean']*100:.2f}±{row['no_jepa_std']*100:.2f} | {row['uniform_mean']*100:.2f}±{row['uniform_std']*100:.2f} | {row['delta']*100:+.2f} | {row['positive_seeds']}/5 |")
    lines += [
        "",
        f"8-rate mean: no-JEPA {summary['overall']['no_jepa_mean']*100:.2f}%, uniform-forced-text {summary['overall']['uniform_mean']*100:.2f}% ({summary['overall']['delta']*100:+.2f} pp).",
        f"High-missing (.5/.6/.7): no-JEPA {summary['high_missing']['no_jepa_mean']*100:.2f}%, uniform-forced-text {summary['high_missing']['uniform_mean']*100:.2f}% ({summary['high_missing']['delta']*100:+.2f} pp).",
        "",
        "## Training mask audit",
        "",
        f"Mean sampled r: {summary['train_mask']['sampled_rate_mean']:.4f}; realized missing fraction: {summary['train_mask']['realized_missing_fraction_mean']:.4f}; forced-Text fraction: {summary['train_mask']['forced_text_fraction_mean']:.4f}; target probability: 0.25.",
        "",
        "## T-missing vs T-present",
        "",
        "W-F1 excludes label==0. Pattern-macro averages per-pattern W-F1; sample-pooled concatenates underlying valid samples before computing W-F1.",
        "",
        "| group | aggregation | no-JEPA | uniform-forced-text | delta | valid samples |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for group, values in summary["text_groups"].items():
        for aggregation in ("pattern_macro", "sample_pooled"):
            c0 = values[aggregation]["no-JEPA"]
            c1 = values[aggregation]["Uniform"]
            lines.append(f"| {group} | {aggregation} | {c0*100:.2f} | {c1*100:.2f} | {(c1-c0)*100:+.2f} | {values['sample_count']['no-JEPA']}/{values['sample_count']['Uniform']} |")
    lines += ["", "## Seven-pattern sample-pooled W-F1", "", "| pattern | no-JEPA | uniform-forced-text | delta |", "|---|---:|---:|---:|"]
    for pattern in PATTERN_ORDER:
        c0 = summary["patterns"][pattern]["no-JEPA"]["sample_pooled_wf1"]
        c1 = summary["patterns"][pattern]["Uniform"]["sample_pooled_wf1"]
        lines.append(f"| {pattern} | {c0*100:.2f} | {c1*100:.2f} | {(c1-c0)*100:+.2f} |")
    (ROOT / "RESULT.md").write_text("\n".join(lines) + "\n")
    print((ROOT / "RESULT.md").read_text())


if __name__ == "__main__":
    main()
