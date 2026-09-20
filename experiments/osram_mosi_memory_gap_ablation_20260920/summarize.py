"""Summarize the four-way MOSI memory/base/gap ablation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

RATES = tuple(f"{index / 10:.1f}" for index in range(8))
VARIANTS = ("local-only", "local-base", "raw-gap", "full")
SEEDS = (66, 67, 68, 69, 70)


def best_per_rate(history: list[dict]) -> dict[str, dict[str, float | int]]:
    best: dict[str, dict[str, float | int]] = {}
    for row in history:
        for rate, metrics in row.get("test_oracle", {}).items():
            score = metrics.get("weighted_f1")
            if rate not in RATES or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
                continue
            if rate not in best or float(score) > float(best[rate]["weighted_f1"]):
                best[rate] = {"epoch": int(row["epoch"]), "weighted_f1": float(score)}
    if set(best) != set(RATES):
        raise ValueError("history is missing one or more official test rates")
    return best


def collect(root: Path) -> dict:
    rows = []
    pending = []
    for variant in VARIANTS:
        for seed in SEEDS:
            path = root / variant / f"seed_{seed}"
            required = ("PROVENANCE.json", "config.json", "history.json", "metrics.json")
            if any(not (path / name).exists() for name in required):
                pending.append({"variant": variant, "seed": seed, "reason": "missing artifacts"})
                continue
            provenance = json.loads((path / "PROVENANCE.json").read_text())
            if provenance.get("status") != "complete":
                pending.append({"variant": variant, "seed": seed, "reason": provenance.get("status")})
                continue
            best = best_per_rate(json.loads((path / "history.json").read_text()))
            for rate in RATES:
                rows.append({"variant": variant, "seed": seed, "rate": rate, **best[rate]})
    aggregates = []
    if not pending:
        for variant in VARIANTS:
            for rate in RATES:
                values = [row["weighted_f1"] for row in rows if row["variant"] == variant and row["rate"] == rate]
                aggregates.append({"variant": variant, "rate": rate, "mean": statistics.mean(values),
                                   "std": statistics.stdev(values)})
    overall = []
    for variant in VARIANTS:
        seed_means = []
        for seed in SEEDS:
            values = [row["weighted_f1"] for row in rows if row["variant"] == variant and row["seed"] == seed]
            if len(values) == len(RATES):
                seed_means.append(statistics.mean(values))
        if len(seed_means) == len(SEEDS):
            high = [statistics.mean(row["weighted_f1"] for row in rows
                                    if row["variant"] == variant and row["seed"] == seed and row["rate"] in {"0.5", "0.6", "0.7"})
                    for seed in SEEDS]
            overall.append({"variant": variant, "all8_mean": statistics.mean(seed_means),
                            "all8_std": statistics.stdev(seed_means),
                            "high_mean": statistics.mean(high), "high_std": statistics.stdev(high)})
    return {"status": "complete" if not pending else "partial", "rows": rows,
            "aggregates": aggregates, "overall": overall, "pending": pending,
            "selection_protocol": "per-rate-test-oracle",
            "label": "INTERNAL DIAGNOSTIC ONLY; NOT A FORMAL PAPER RESULT"}


def write_report(report: dict, root: Path) -> None:
    summary = root / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    (summary / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    with (summary / "per_seed_rate.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("variant", "seed", "rate", "epoch", "weighted_f1"))
        writer.writeheader()
        writer.writerows(report["rows"])
    lines = ["# MOSI memory/base/gap ablation", "", report["label"], "",
             "Protocol: cfg84 causal OSRAM (eta=.6, Flat, no-JEPA), trained from scratch; each seed/rate selects its own Test-oracle epoch.", "",
             "| Variant | Rate | Mean W-F1 | SD |", "|---|---:|---:|---:|"]
    for row in report["aggregates"]:
        lines.append(f"| {row['variant']} | {row['rate']} | {100*row['mean']:.3f} | {100*row['std']:.3f} |")
    lines += ["", "## Overall", "", "| Variant | 8-rate mean | High-missing mean (.5/.6/.7) |", "|---|---:|---:|"]
    for row in report["overall"]:
        lines.append(f"| {row['variant']} | {100*row['all8_mean']:.3f} ± {100*row['all8_std']:.3f} | {100*row['high_mean']:.3f} ± {100*row['high_std']:.3f} |")
    if report["pending"]:
        lines += ["", "## Pending", ""] + [f"- {item['variant']} seed {item['seed']}: {item['reason']}" for item in report["pending"]]
    (summary / "RESULT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    report = collect(args.root)
    write_report(report, args.root)
    print(f"status={report['status']} rows={len(report['rows'])} pending={len(report['pending'])}")


if __name__ == "__main__":
    main()
