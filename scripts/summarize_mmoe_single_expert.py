"""Summarize the single-expert MMoE weak-point diagnostic."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


RATES = [f"{i / 10:.1f}" for i in range(8)]
SEEDS = [66, 67, 68, 69, 70]


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def full_rows(root: Path) -> list[dict[str, object]]:
    source = root.parent / "osram_heads8_out700_20260906" / "per_seed_rate.csv"
    rows: list[dict[str, object]] = []
    with source.open(newline="") as handle:
        for row in csv.DictReader(handle):
            row["model"] = "Full-4E"
            rows.append(row)
    return rows


def single_rows(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        history = json.loads(
            (root / "raw" / f"seed_{seed}" / "history.json").read_text()
        )
        mean_selected = max(
            history, key=lambda item: item["test_oracle_mean_weighted_f1"]
        )
        for rate in RATES:
            selected = max(
                history, key=lambda item: item["test_oracle"][rate]["weighted_f1"]
            )
            metric = selected["test_oracle"][rate]
            rows.append(
                {
                    "model": "Single-1E",
                    "seed": seed,
                    "rate": rate,
                    "weighted_f1": metric["weighted_f1"],
                    "macro_f1": metric["macro_f1"],
                    "accuracy": metric["accuracy"],
                    "mae": metric["mae"],
                    "correlation": metric["correlation"],
                    "selected_epoch": selected["epoch"],
                    "selection_8rate_mean": mean_selected[
                        "test_oracle_mean_weighted_f1"
                    ],
                    "selection_policy": "per-rate-test-oracle",
                }
            )
    return rows


def main() -> None:
    root = Path("experiments/mmoe_single_expert_mosi_20260907")
    rows = full_rows(root) + single_rows(root)
    fields = [
        "model",
        "seed",
        "rate",
        "weighted_f1",
        "macro_f1",
        "accuracy",
        "mae",
        "correlation",
        "selected_epoch",
        "selection_8rate_mean",
        "selection_policy",
    ]
    with (root / "per_seed_rate.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["model"]), str(row["rate"])), []).append(row)

    full_lookup = {
        (int(row["seed"]), str(row["rate"])): float(row["weighted_f1"])
        for row in rows
        if row["model"] == "Full-4E"
    }
    summary_rows: list[dict[str, object]] = []
    for rate in RATES:
        full = [
            float(row["weighted_f1"]) for row in grouped[("Full-4E", rate)]
        ]
        single = [
            float(row["weighted_f1"]) for row in grouped[("Single-1E", rate)]
        ]
        summary_rows.append(
            {
                "rate": rate,
                "Full_4E_mean": mean(full),
                "Full_4E_std": std(full),
                "Single_1E_mean": mean(single),
                "Single_1E_std": std(single),
                "delta": mean(single) - mean(full),
                "positive_seed_count": sum(
                    value > full_lookup[(int(row["seed"]), rate)]
                    for value, row in zip(single, grouped[("Single-1E", rate)])
                ),
            }
        )
    with (root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=summary_rows[0], lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    def values(model: str, rates: list[str] = RATES) -> list[float]:
        return [
            float(row["weighted_f1"])
            for rate in rates
            for row in grouped[(model, rate)]
        ]

    def seed_means(model: str) -> dict[int, float]:
        return {
            seed: mean(
                [
                    float(row["weighted_f1"])
                    for row in rows
                    if row["model"] == model and int(row["seed"]) == seed
                ]
            )
            for seed in SEEDS
        }

    full_seed = seed_means("Full-4E")
    single_seed = seed_means("Single-1E")
    strict_selection: dict[str, dict[str, object]] = {}
    for label, directory in (
        ("Full-4E", root.parent / "osram_heads8_out700_20260906" / "raw"),
        ("Single-1E", root / "raw"),
    ):
        per_seed = []
        epochs = []
        for seed in SEEDS:
            history = json.loads(
                (directory / f"seed_{seed}" / "history.json").read_text()
            )
            selected = max(
                history, key=lambda item: item["test_oracle_mean_weighted_f1"]
            )
            per_seed.append(selected["test_oracle_mean_weighted_f1"])
            epochs.append(selected["epoch"])
        strict_selection[label] = {
            "per_seed_mean": per_seed,
            "mean": mean(per_seed),
            "selected_epochs": epochs,
        }
    routing_summary: dict[str, object] = {}
    for label, directory in (
        ("Full-4E", root.parent / "osram_heads8_out700_20260906" / "raw"),
        ("Single-1E", root / "raw"),
    ):
        entries = []
        for seed in SEEDS:
            history = json.loads(
                (directory / (f"seed_{seed}") / "history.json").read_text()
            )
            entries.append(history[-1]["train"]["routing"])
        routing_summary[label] = {
            "num_experts": 4 if label == "Full-4E" else 1,
            "top_k": 2 if label == "Full-4E" else 1,
            "mmoe_parameter_count": 924680 if label == "Full-4E" else 528386,
            "entropy_mean": {
                branch: mean([entry[branch]["entropy"] for entry in entries])
                for branch in ("regression", "contrastive")
            },
            "usage_mean": {
                branch: [
                    mean([entry[branch]["usage"][index] for entry in entries])
                    for index in range(
                        len(entries[0][branch]["usage"])
                    )
                ]
                for branch in ("regression", "contrastive")
            },
        }
    (root / "diagnostics.json").write_text(
        json.dumps(
            {
                "selection_protocol": "per-rate-test-oracle",
                "routing": routing_summary,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    high = ["0.5", "0.6", "0.7"]
    lines = [
        "# MMoE routing weak-point diagnostic",
        "",
        "**Internal diagnostic only; not a formal paper result.**",
        "",
        "Full-4E is inherited from the H8/32/32/700 OSRAM run. Single-1E was",
        "trained with the same cyclic mixed-rate protocol and changes only",
        "`num_experts=1, top_k=1`. The table uses the same per-rate Test-oracle",
        "extraction convention as the current OSRAM diagnostic; each seed's",
        "eight-rate Test-oracle checkpoint mean is also recorded.",
        "",
        "| Condition | 8-rate mean | High-missing mean (0.5/0.6/0.7) |",
        "|---|---:|---:|",
        f"| Full-4E | {100 * mean(values('Full-4E')):.3f}% | {100 * mean(values('Full-4E', high)):.3f}% |",
        f"| Single-1E | {100 * mean(values('Single-1E')):.3f}% | {100 * mean(values('Single-1E', high)):.3f}% |",
        "",
        "## Strict one-checkpoint cross-rate check",
        "",
        "The table above intentionally gives each rate its own best epoch,",
        "matching the requested per-rate diagnostic. As a guard against that",
        "optimistic view, the same histories were also scored with one",
        "eight-rate-mean Test-oracle checkpoint per seed:",
        "",
        "| Condition | mean over 5 one-checkpoint scores | selected epochs (66,67,68,69,70) |",
        "|---|---:|---|",
        f"| Full-4E | {100 * strict_selection['Full-4E']['mean']:.3f}% | {strict_selection['Full-4E']['selected_epochs']} |",
        f"| Single-1E | {100 * strict_selection['Single-1E']['mean']:.3f}% | {strict_selection['Single-1E']['selected_epochs']} |",
        f"| Δ | {100 * (strict_selection['Single-1E']['mean'] - strict_selection['Full-4E']['mean']):+.3f} percentage points | — |",
        "",
        "| Rate | Full-4E | Single-1E | Δ | positive seeds |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['rate']} | {100 * float(row['Full_4E_mean']):.3f}% ± {100 * float(row['Full_4E_std']):.3f} | "
            f"{100 * float(row['Single_1E_mean']):.3f}% ± {100 * float(row['Single_1E_std']):.3f} | "
            f"{100 * float(row['delta']):+.3f} | {row['positive_seed_count']}/5 |"
        )
    lines += [
        "",
        "## Per-seed eight-rate means",
        "",
        "| Seed | Full-4E | Single-1E | Δ |",
        "|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        lines.append(
            f"| {seed} | {100 * full_seed[seed]:.3f}% | {100 * single_seed[seed]:.3f}% | "
            f"{100 * (single_seed[seed] - full_seed[seed]):+.3f} |"
        )
    lines += [
        "",
        "## Diagnosis",
        "",
        "The single-expert control is a mechanism screen rather than a",
        "parameter-matched final model. A drop indicates that expert",
        "specialization/routing contributes useful capacity; parity or an",
        "improvement indicates that routing is a likely weak point and that a",
        "later parameter-matched shared predictor would be warranted.",
        "",
        "## Routing diagnostics",
        "",
        "The final training histories show that the Full-4E condition uses",
        "nonzero expert specialization, while Single-1E has zero routing",
        "entropy by construction.  This is evidence about the mechanism, not",
        "a claim that the lower-capacity control is a fair final model; the",
        "parameter counts and averaged routing statistics are in",
        "`diagnostics.json`.",
        "",
    ]
    (root / "RESULT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
