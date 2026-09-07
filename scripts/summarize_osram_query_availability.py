"""Summarize the OSRAM explicit-query-availability ablation."""

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


def inherited_rows(root: Path) -> list[dict[str, object]]:
    source = root.parent / "osram_heads8_out700_20260906" / "per_seed_rate.csv"
    rows: list[dict[str, object]] = []
    with source.open(newline="") as handle:
        for row in csv.DictReader(handle):
            row["model"] = "Explicit-a_t"
            rows.append(row)
    return rows


def no_availability_rows(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        history = json.loads(
            (root / "raw" / f"seed_{seed}" / "history.json").read_text()
        )
        strict_selected = max(
            history, key=lambda item: item["test_oracle_mean_weighted_f1"]
        )
        for rate in RATES:
            selected = max(
                history, key=lambda item: item["test_oracle"][rate]["weighted_f1"]
            )
            metric = selected["test_oracle"][rate]
            rows.append(
                {
                    "model": "No-explicit-a_t",
                    "seed": seed,
                    "rate": rate,
                    "weighted_f1": metric["weighted_f1"],
                    "macro_f1": metric["macro_f1"],
                    "accuracy": metric["accuracy"],
                    "mae": metric["mae"],
                    "correlation": metric["correlation"],
                    "selected_epoch": selected["epoch"],
                    "selection_8rate_mean": strict_selected[
                        "test_oracle_mean_weighted_f1"
                    ],
                    "selection_policy": "per-rate-test-oracle",
                }
            )
    return rows


def strict_scores(directory: Path) -> tuple[list[float], list[int]]:
    scores: list[float] = []
    epochs: list[int] = []
    for seed in SEEDS:
        history = json.loads(
            (directory / f"seed_{seed}" / "history.json").read_text()
        )
        selected = max(
            history, key=lambda item: item["test_oracle_mean_weighted_f1"]
        )
        scores.append(selected["test_oracle_mean_weighted_f1"])
        epochs.append(selected["epoch"])
    return scores, epochs


def main() -> None:
    root = Path("experiments/osram_query_no_availability_mosi_20260907")
    rows = inherited_rows(root) + no_availability_rows(root)
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
    explicit_lookup = {
        (int(row["seed"]), str(row["rate"])): float(row["weighted_f1"])
        for row in rows
        if row["model"] == "Explicit-a_t"
    }
    summary_rows: list[dict[str, object]] = []
    for rate in RATES:
        explicit = [
            float(row["weighted_f1"])
            for row in grouped[("Explicit-a_t", rate)]
        ]
        no_availability = [
            float(row["weighted_f1"])
            for row in grouped[("No-explicit-a_t", rate)]
        ]
        summary_rows.append(
            {
                "rate": rate,
                "Explicit_a_t_mean": mean(explicit),
                "Explicit_a_t_std": std(explicit),
                "No_explicit_a_t_mean": mean(no_availability),
                "No_explicit_a_t_std": std(no_availability),
                "delta": mean(no_availability) - mean(explicit),
                "positive_seed_count": sum(
                    value > explicit_lookup[(int(row["seed"]), rate)]
                    for value, row in zip(
                        no_availability,
                        grouped[("No-explicit-a_t", rate)],
                    )
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

    explicit_scores, explicit_epochs = strict_scores(
        root.parent / "osram_heads8_out700_20260906" / "raw"
    )
    no_availability_scores, no_availability_epochs = strict_scores(root / "raw")
    high = ["0.5", "0.6", "0.7"]
    lines = [
        "# OSRAM Query availability diagnostic",
        "",
        "**Internal diagnostic only; not a formal paper result.**",
        "",
        "The only newly trained condition removes explicit `a_t` from the Query.",
        "Key/Value conditioning, hard gap selection, memory write, predictor and",
        "all training settings remain unchanged.",
        "",
        "| Condition | 8-rate mean | High-missing mean (0.5/0.6/0.7) |",
        "|---|---:|---:|",
        f"| Explicit `a_t` | {100 * mean(values('Explicit-a_t')):.3f}% | {100 * mean(values('Explicit-a_t', high)):.3f}% |",
        f"| No explicit `a_t` | {100 * mean(values('No-explicit-a_t')):.3f}% | {100 * mean(values('No-explicit-a_t', high)):.3f}% |",
        "",
        "## Strict one-checkpoint cross-rate check",
        "",
        "| Condition | mean | selected epochs (66,67,68,69,70) |",
        "|---|---:|---|",
        f"| Explicit `a_t` | {100 * mean(explicit_scores):.3f}% | {explicit_epochs} |",
        f"| No explicit `a_t` | {100 * mean(no_availability_scores):.3f}% | {no_availability_epochs} |",
        f"| Δ (no explicit − explicit) | {100 * (mean(no_availability_scores) - mean(explicit_scores)):+.3f} percentage points | — |",
        "",
        "| Rate | Explicit `a_t` | No explicit `a_t` | Δ | positive seeds |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['rate']} | {100 * float(row['Explicit_a_t_mean']):.3f}% ± {100 * float(row['Explicit_a_t_std']):.3f} | "
            f"{100 * float(row['No_explicit_a_t_mean']):.3f}% ± {100 * float(row['No_explicit_a_t_std']):.3f} | "
            f"{100 * float(row['delta']):+.3f} | {row['positive_seed_count']}/5 |"
        )
    lines += [
        "",
        "The per-rate rows use the requested diagnostic convention in which each",
        "rate is allowed to select its own Test-oracle epoch. The strict table is",
        "the guard using one eight-rate-mean Test-oracle checkpoint per seed.",
        "",
    ]
    (root / "RESULT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
