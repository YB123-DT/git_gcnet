"""Summarize the write-gate / Gap-strength follow-up."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

RATES = tuple(f"{index / 10:.1f}" for index in range(8))
VARIANTS = ("external-beta-r1", "embedded-r05", "external-beta-r05")
SEEDS = (66, 67, 68, 69, 70)


def best_per_rate(history):
    best = {}
    for row in history:
        for rate, metrics in row.get("test_oracle", {}).items():
            score = metrics.get("weighted_f1")
            if rate in RATES and isinstance(score, (int, float)) and math.isfinite(float(score)):
                if rate not in best or float(score) > best[rate]["weighted_f1"]:
                    best[rate] = {"epoch": int(row["epoch"]), "weighted_f1": float(score)}
    if set(best) != set(RATES):
        raise ValueError("incomplete history")
    return best


def _load(root, variant, seed):
    path = root / variant / f"seed_{seed}"
    if not all((path / name).exists() for name in ("PROVENANCE.json", "history.json")):
        return None
    if json.loads((path / "PROVENANCE.json").read_text()).get("status") != "complete":
        return None
    return best_per_rate(json.loads((path / "history.json").read_text()))


def collect(root: Path, previous: Path):
    rows, pending = [], []
    baseline = {}
    for seed in SEEDS:
        value = _load(previous, "full", seed)
        if value is None:
            pending.append({"variant": "full-reference", "seed": seed})
        else:
            baseline[seed] = value
    for variant in VARIANTS:
        for seed in SEEDS:
            value = _load(root, variant, seed)
            if value is None:
                pending.append({"variant": variant, "seed": seed})
            else:
                rows.extend({"variant": variant, "seed": seed, "rate": rate, **value[rate]} for rate in RATES)
    aggregates = []
    overall = []
    if not pending:
        for variant in VARIANTS:
            for rate in RATES:
                values = [row["weighted_f1"] for row in rows if row["variant"] == variant and row["rate"] == rate]
                ref = [baseline[seed][rate]["weighted_f1"] for seed in SEEDS]
                aggregates.append({"variant": variant, "rate": rate, "mean": statistics.mean(values),
                                   "std": statistics.stdev(values), "delta_full": statistics.mean(a-b for a,b in zip(values,ref))})
            for label, selected in (("all8", RATES), ("high", ("0.5", "0.6", "0.7"))):
                means = [statistics.mean(next(row["weighted_f1"] for row in rows if row["variant"] == variant and row["seed"] == seed and row["rate"] == rate) for rate in selected) for seed in SEEDS]
                refs = [statistics.mean(baseline[seed][rate]["weighted_f1"] for rate in selected) for seed in SEEDS]
                overall.append({"variant": variant, "rates": label, "mean": statistics.mean(means),
                                "std": statistics.stdev(means), "delta_full": statistics.mean(a-b for a,b in zip(means,refs)),
                                "positive_seed_count": sum(a>b for a,b in zip(means,refs))})
    return {"status": "complete" if not pending else "partial", "rows": rows,
            "aggregates": aggregates, "overall": overall, "pending": pending,
            "selection_protocol": "per-rate-test-oracle", "label": "INTERNAL DIAGNOSTIC ONLY; NOT A FORMAL PAPER RESULT"}


def write_report(report, root: Path):
    out = root / "summary"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    with (out / "per_seed_rate.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("variant", "seed", "rate", "epoch", "weighted_f1"))
        writer.writeheader(); writer.writerows(report["rows"])
    lines = ["# MOSI write-gate / Gap residual-strength follow-up", "", report["label"], "",
             "Protocol: cfg84 causal OSRAM eta=.6, per-seed × per-rate Test-oracle selection.", "",
             "| Variant | Rate | Mean W-F1 | SD | Δ previous Full |", "|---|---:|---:|---:|---:|"]
    for row in report["aggregates"]:
        lines.append(f"| {row['variant']} | {row['rate']} | {100*row['mean']:.3f} | {100*row['std']:.3f} | {100*row['delta_full']:+.3f} |")
    lines += ["", "## Overall", "", "| Variant | Rates | Mean ± SD | Δ Full | Positive seeds |", "|---|---|---:|---:|---:|"]
    for row in report["overall"]:
        lines.append(f"| {row['variant']} | {row['rates']} | {100*row['mean']:.3f} ± {100*row['std']:.3f} | {100*row['delta_full']:+.3f} | {row['positive_seed_count']}/5 |")
    if report["pending"]:
        lines += ["", "## Pending", ""] + [f"- {item['variant']} seed {item['seed']}" for item in report["pending"]]
    (out / "RESULT.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    args = parser.parse_args()
    report = collect(args.root, args.previous)
    write_report(report, args.root)
    print(f"status={report['status']} rows={len(report['rows'])} pending={len(report['pending'])}")


if __name__ == "__main__":
    main()
