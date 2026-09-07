"""Summarize the per-rate Test-oracle OSRAM ablation histories."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


RATES = [f"{index / 10:.1f}" for index in range(8)]
SEEDS = [66, 67, 68, 69, 70]
VARIANTS = ("Full OSRAM", "Local+Base", "Local-only")


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def _std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _history_rows(
    root: Path, variant: str, label: str | None = None
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    label = label or variant
    for seed in SEEDS:
        history_path = root / "raw" / variant / f"seed_{seed}" / "history.json"
        history = json.loads(history_path.read_text())
        mean_best = max(
            history,
            key=lambda item: item["test_oracle_mean_weighted_f1"],
        )
        for rate in RATES:
            best = max(
                history,
                key=lambda item: item["test_oracle"][rate]["weighted_f1"],
            )
            metric = best["test_oracle"][rate]
            rows.append(
                {
                    "model": label,
                    "seed": seed,
                    "rate": rate,
                    "weighted_f1": metric["weighted_f1"],
                    "macro_f1": metric["macro_f1"],
                    "accuracy": metric["accuracy"],
                    "mae": metric["mae"],
                    "correlation": metric["correlation"],
                    "selected_epoch": best["epoch"],
                    "selection_8rate_mean": mean_best[
                        "test_oracle_mean_weighted_f1"
                    ],
                    "selection_policy": "per-rate-test-oracle",
                }
            )
    return rows


def _full_rows(root: Path) -> list[dict[str, object]]:
    source = root.parent / "osram_heads8_out700_20260906" / "per_seed_rate.csv"
    rows: list[dict[str, object]] = []
    with source.open(newline="") as handle:
        for row in csv.DictReader(handle):
            row["model"] = "Full OSRAM"
            rows.append(row)
    return rows


def main() -> None:
    root = Path("experiments/osram_ablation_mosi_20260907")
    rows = _full_rows(root) + _history_rows(root, "local-base", "Local+Base")
    rows += _history_rows(root, "local-only", "Local-only")
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
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    by_model_rate: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        by_model_rate.setdefault((row["model"], row["rate"]), []).append(row)
    full_lookup = {
        (int(row["seed"]), row["rate"]): float(row["weighted_f1"])
        for row in rows
        if row["model"] == "Full OSRAM"
    }
    summary_fields = [
        "rate",
        "Full_OSRAM_mean",
        "Full_OSRAM_std",
        "Local+Base_mean",
        "Local+Base_std",
        "Local+Base_delta",
        "Local+Base_positive_seed_count",
        "Local-only_mean",
        "Local-only_std",
        "Local-only_delta",
        "Local-only_positive_seed_count",
    ]
    summary_rows = []
    for rate in RATES:
        full = [
            float(row["weighted_f1"])
            for row in by_model_rate[("Full OSRAM", rate)]
        ]
        row: dict[str, object] = {
            "rate": rate,
            "Full_OSRAM_mean": _mean(full),
            "Full_OSRAM_std": _std(full),
        }
        for variant, label in (("Local+Base", "Local+Base"), ("Local-only", "Local-only")):
            values = [
                float(item["weighted_f1"])
                for item in by_model_rate[(variant, rate)]
            ]
            delta = _mean(values) - _mean(full)
            positive = sum(
                float(item["weighted_f1"])
                > full_lookup[(int(item["seed"]), rate)]
                for item in by_model_rate[(variant, rate)]
            )
            row[f"{label}_mean"] = _mean(values)
            row[f"{label}_std"] = _std(values)
            row[f"{label}_delta"] = delta
            row[f"{label}_positive_seed_count"] = positive
        summary_rows.append(row)
    with (root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    diagnostics = {}
    for variant in ("local-base", "local-only"):
        entries = []
        for seed in SEEDS:
            path = root / "raw" / variant / f"seed_{seed}" / "diagnostics.json"
            entries.append(json.loads(path.read_text()))
        last_batches = [entry["last_batch"] for entry in entries]
        diagnostics[variant] = {
            "active_base_context_norm_mean": _mean(
                [item["base_context_norm"] for item in last_batches]
            ),
            "active_gap_context_norm_mean": {
                name: _mean(
                    [item["gap_context_norm"][name] for item in last_batches]
                )
                for name in ("audio", "text", "visual")
            },
            "address_residual_mean": {
                name: {
                    metric: _mean(
                        [
                            item["address_residual"][name][metric]
                            for item in last_batches
                        ]
                    )
                    for metric in ("rho", "eta", "cosine")
                }
                for name in ("audio", "text", "visual")
            },
            "selection_protocol": "per-rate-test-oracle",
        }
    (root / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n"
    )

    def model_values(model: str, rates: list[str] = RATES) -> list[float]:
        return [
            float(item["weighted_f1"])
            for rate in rates
            for item in by_model_rate[(model, rate)]
        ]

    lines = [
        "# OSRAM mechanism ablation result",
        "",
        "**Internal diagnostic only; not a formal paper result.**",
        "",
        "Each seed is one cyclic mixed-rate training run.  For the table below,",
        "each missing rate selects its own best Test weighted-F1 epoch from that",
        "history (`per-rate-test-oracle`).  Full OSRAM is inherited from the",
        "H8/32/32/700 run in `experiments/osram_heads8_out700_20260906/`.",
        "",
        "| Condition | 8-rate mean | High-missing mean (0.5/0.6/0.7) |",
        "|---|---:|---:|",
    ]
    high = ["0.5", "0.6", "0.7"]
    for model in VARIANTS:
        lines.append(
            f"| {model} | {100 * _mean(model_values(model)):.3f}% | "
            f"{100 * _mean(model_values(model, high)):.3f}% |"
        )
    lines += [
        "",
        "| Rate | Full OSRAM | Local+Base | Δ | Local-only | Δ |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary_rows:
        lines.append(
            f"| {item['rate']} | {100 * float(item['Full_OSRAM_mean']):.3f}% "
            f"± {100 * float(item['Full_OSRAM_std']):.3f} | "
            f"{100 * float(item['Local+Base_mean']):.3f}% | "
            f"{100 * float(item['Local+Base_delta']):+.3f} | "
            f"{100 * float(item['Local-only_mean']):.3f}% | "
            f"{100 * float(item['Local-only_delta']):+.3f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "Local-only removes both memory readout families and is the direct",
        "current-utterance control.  Local+Base keeps ordinary bidirectional",
        "memory but masks the three missing-specific gap slots.  The Full minus",
        "Local+Base difference therefore estimates the incremental contribution",
        "of the gap-conditioned readout under this diagnostic protocol.",
        "",
        "The ablation switch does not change the OSRAM scan or parameter count;",
        "it masks only the selected context slots before the emotion head and",
        "structured predictor.  Results are optimistic Test-oracle diagnostics",
        "and must not be presented as validation-selected benchmark scores.",
        "",
    ]
    (root / "RESULT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
