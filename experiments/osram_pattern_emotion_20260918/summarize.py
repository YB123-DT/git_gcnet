"""Aggregate pattern-loss MOSI runs and matched no-JEPA/reg-only controls."""

import argparse
import csv
import json
from pathlib import Path
import statistics

import numpy as np
from sklearn.metrics import f1_score


SEEDS = (66, 67, 68, 69, 70)
RATES = ("0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7")
PATTERNS = (("A", 4), ("T", 2), ("V", 1), ("AT", 6), ("AV", 5), ("TV", 3), ("ATV", 7))
VARIANTS = {
    "no-jepa": "baseline no-JEPA",
    "no-jepa-pb": "no-JEPA + pattern-balanced",
    "no-jepa-groupdro": "no-JEPA + GroupDRO",
    "reg-only": "baseline reg-only JEPA",
    "reg-only-pb": "reg-only JEPA + pattern-balanced",
    "reg-only-groupdro": "reg-only JEPA + GroupDRO",
}


def _read_json(path: Path):
    return json.loads(path.read_text())


def _source(root: Path, variant: str, seed: int) -> Path:
    if variant == "no-jepa":
        return root / "osram_causal_nojepa_20260910" / "mosi" / f"seed_{seed}"
    if variant == "reg-only":
        return root / "osram_supervised_teacher_20260914" / "student-reg-only" / f"seed_{seed}"
    return root / "osram_pattern_emotion_20260918" / variant / f"seed_{seed}"


def _score(labels: np.ndarray, predictions: np.ndarray) -> float:
    valid = labels != 0
    return float(
        f1_score(labels[valid] > 0, predictions[valid] > 0, average="weighted", zero_division=0)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote-root", type=Path, default=Path("/data2/yb/remote_experiments"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rate_rows = []
    pattern_rows = []
    for variant in VARIANTS:
        for seed in SEEDS:
            source = _source(args.remote_root, variant, seed)
            metrics = _read_json(source / "metrics.json")
            selected = metrics.get("selected_epoch_by_rate", {})
            for rate in RATES:
                test = metrics["test"][rate]
                rate_rows.append({
                    "variant": variant,
                    "label": VARIANTS[variant],
                    "seed": seed,
                    "rate": rate,
                    "weighted_f1": test["weighted_f1"],
                    "macro_f1": test["macro_f1"],
                    "accuracy": test["accuracy"],
                    "mae": test.get("mae"),
                    "correlation": test.get("correlation"),
                    "selected_epoch": selected.get(rate, metrics.get("best_epoch")),
                    "selection_protocol": metrics["selection_protocol"],
                })
                npz = source / f"predictions_miss_{rate.replace('.', 'p')}.npz"
                if not npz.exists():
                    continue
                with np.load(npz, allow_pickle=False) as data:
                    labels = data["labels"].reshape(-1)
                    predictions = data["predictions"].reshape(-1)
                    availability = data["availability"].reshape(-1, 3)
                code = availability @ np.array([4, 2, 1])
                groups = list(PATTERNS) + [("NO_TEXT", -1), ("TEXT_PRESENT", -2), ("ALL", None)]
                for pattern, pattern_id in groups:
                    if pattern_id is None:
                        selected_mask = np.ones(labels.shape[0], dtype=bool)
                    elif pattern_id == -1:
                        selected_mask = availability[:, 1] == 0
                    elif pattern_id == -2:
                        selected_mask = availability[:, 1] == 1
                    else:
                        selected_mask = code == pattern_id
                    scored = selected_mask & (labels != 0)
                    if not scored.any():
                        continue
                    pattern_rows.append({
                        "variant": variant,
                        "label": VARIANTS[variant],
                        "seed": seed,
                        "rate": rate,
                        "pattern": pattern,
                        "n": int(scored.sum()),
                        "weighted_f1": _score(labels[scored], predictions[scored]),
                    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("per_seed_rate.csv", rate_rows), ("pattern_per_seed.csv", pattern_rows)):
        with (args.output_dir / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summaries = []
    for variant in VARIANTS:
        rows = [r for r in rate_rows if r["variant"] == variant]
        for rate in RATES:
            values = [float(r["weighted_f1"]) for r in rows if r["rate"] == rate]
            summaries.append({
                "variant": variant,
                "label": VARIANTS[variant],
                "rate": rate,
                "mean_weighted_f1": statistics.mean(values),
                "std_weighted_f1": statistics.stdev(values),
                "positive_seed_count": None,
            })
        all_values = [float(r["weighted_f1"]) for r in rows]
        for group, rates in (("8-rate-mean", RATES), ("high-missing-mean", ("0.5", "0.6", "0.7"))):
            by_seed = [
                statistics.mean(float(r["weighted_f1"]) for r in rows if r["seed"] == seed and r["rate"] in rates)
                for seed in SEEDS
            ]
            summaries.append({
                "variant": variant,
                "label": VARIANTS[variant],
                "rate": group,
                "mean_weighted_f1": statistics.mean(by_seed),
                "std_weighted_f1": statistics.stdev(by_seed),
                "positive_seed_count": sum(value > 0 for value in by_seed),
            })
    with (args.output_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (args.output_dir / "summary.json").write_text(json.dumps(summaries, indent=2))
    lines = [
        "# Pattern-balanced / GroupDRO MOSI diagnostic",
        "",
        "Internal diagnostic only. Each seed×missing-rate checkpoint was selected independently by Test W-F1; JEPA loss weighting was unchanged.",
        "",
        "## Overall",
        "",
        "| Variant | 8-rate mean | High-missing mean (0.5/0.6/0.7) |",
        "|---|---:|---:|",
    ]
    for variant in VARIANTS:
        lookup = {(r["variant"], r["rate"]): r for r in summaries}
        overall = lookup[(variant, "8-rate-mean")]
        high = lookup[(variant, "high-missing-mean")]
        lines.append(
            f"| {VARIANTS[variant]} | {overall['mean_weighted_f1']:.4f} ± {overall['std_weighted_f1']:.4f} | {high['mean_weighted_f1']:.4f} ± {high['std_weighted_f1']:.4f} |"
        )
    lines += ["", "## Per-rate means", "", "| Variant | " + " | ".join(RATES) + " |", "|---|" + "---:|" * len(RATES)]
    for variant in VARIANTS:
        lookup = {(r["variant"], r["rate"]): r for r in summaries}
        lines.append("| " + VARIANTS[variant] + " | " + " | ".join(f"{lookup[(variant, rate)]['mean_weighted_f1']:.4f}" for rate in RATES) + " |")
    lines += ["", "## Pattern files", "", "Pattern-level rows are in `pattern_per_seed.csv`; the primary comparison remains per-rate Test-oracle W-F1.", ""]
    (args.output_dir / "RESULT.md").write_text("\n".join(lines))
    print((args.output_dir / "RESULT.md").read_text())


if __name__ == "__main__":
    main()

