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


def read_json(path: Path):
    return json.loads(path.read_text())


def mean_std(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1)) if values.size > 1 else 0.0


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
    for variant, root in (("no-JEPA", BASE), ("Text-Core", ROOT / "mosi")):
        for seed in SEEDS:
            for rate in RATES:
                data = np.load(root / f"seed_{seed}" / f"predictions_miss_{rate.replace('.', 'p')}.npz")
                availability = data["availability"]
                labels = data["labels"]
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
                    binary_labels = labels[selected] > 0
                    binary_predictions = predictions[selected] > 0
                    rows.append(
                        {
                            "variant": variant,
                            "seed": seed,
                            "rate": rate,
                            "pattern": name,
                            "count": int(selected.sum()),
                            "weighted_f1": float(
                                f1_score(binary_labels, binary_predictions, average="weighted")
                            ),
                        }
                    )
    return rows


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
    p_rows = pattern_rows()
    write_csv(ROOT / "pattern_per_seed.csv", p_rows)

    summary = {"rate": [], "overall": {}, "high_missing": {}, "text_groups": {}, "task_slots": {}}
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
    for group, names in (("T-missing", {"A", "V", "AV"}), ("T-present", {"T", "AT", "TV", "ATV"})):
        selected = [row for row in p_rows if row["pattern"] in names]
        summary["text_groups"][group] = {
            variant: float(np.mean([row["weighted_f1"] for row in selected if row["variant"] == variant]))
            for variant in ("no-JEPA", "Text-Core")
        }
    slot_values = {"real": [], "predicted": [], "centered_cosine": [], "std_ratio": []}
    for seed in SEEDS:
        metrics = read_json(ROOT / "mosi" / f"seed_{seed}" / "metrics.json")["test"]
        for rate in RATES:
            current = metrics[rate]
            for key, target in (("text_core_real_slot", "real"), ("text_core_predicted_slot", "predicted")):
                if current.get(key) is not None:
                    slot_values[target].append(current[key]["weighted_f1"])
            if current.get("text_core_centered_cosine") is not None:
                slot_values["centered_cosine"].append(current["text_core_centered_cosine"])
            if current.get("text_core_pred_target_std_ratio") is not None:
                slot_values["std_ratio"].append(current["text_core_pred_target_std_ratio"])
    summary["task_slots"] = {
        "real_wf1_mean": float(np.mean(slot_values["real"])),
        "predicted_wf1_mean": float(np.mean(slot_values["predicted"])),
        "centered_cosine_mean": float(np.mean(slot_values["centered_cosine"])),
        "std_ratio_mean": float(np.mean(slot_values["std_ratio"])),
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
            "| group | no-JEPA | Text-Core |",
            "|---|---:|---:|",
        ]
    )
    for group, values in summary["text_groups"].items():
        lines.append(f"| {group} | {values['no-JEPA'] * 100:.2f} | {values['Text-Core'] * 100:.2f} |")
    lines.extend(
        [
            "",
            "## Task-slot diagnostics",
            "",
            f"Real Text slot W-F1: {summary['task_slots']['real_wf1_mean'] * 100:.2f}%",
            f"Predicted Text slot W-F1: {summary['task_slots']['predicted_wf1_mean'] * 100:.2f}%",
            f"Centered cosine (predicted vs complete Text task slot): {summary['task_slots']['centered_cosine_mean']:.4f}",
            f"Prediction/target std ratio: {summary['task_slots']['std_ratio_mean']:.4f}",
            "",
            "## Seven-pattern details",
            "",
            "See `pattern_per_seed.csv`; the key Text-missing patterns are A, V and AV.",
        ]
    )
    (ROOT / "RESULT.md").write_text("\n".join(lines) + "\n")
    print((ROOT / "RESULT.md").read_text())


if __name__ == "__main__":
    main()
