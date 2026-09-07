"""Summarize the OSRAM emotion-only objective diagnostic."""

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
            row["model"] = "Joint"
            rows.append(row)
    return rows


def emotion_only_rows(root: Path) -> list[dict[str, object]]:
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
                    "model": "Emotion-only",
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
    root = Path("experiments/osram_emotion_only_mosi_20260907")
    rows = inherited_rows(root) + emotion_only_rows(root)
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
    summary_rows: list[dict[str, object]] = []
    for rate in RATES:
        joint = [float(row["weighted_f1"]) for row in grouped[("Joint", rate)]]
        emotion = [
            float(row["weighted_f1"])
            for row in grouped[("Emotion-only", rate)]
        ]
        summary_rows.append(
            {
                "rate": rate,
                "Joint_mean": mean(joint),
                "Joint_std": std(joint),
                "Emotion_only_mean": mean(emotion),
                "Emotion_only_std": std(emotion),
                "delta": mean(emotion) - mean(joint),
                "positive_seed_count": sum(
                    value > joint[index]
                    for index, value in enumerate(emotion)
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

    joint_scores, joint_epochs = strict_scores(
        root.parent / "osram_heads8_out700_20260906" / "raw"
    )
    emotion_scores, emotion_epochs = strict_scores(root / "raw")
    high = ["0.5", "0.6", "0.7"]
    lines = [
        "# OSRAM JEPA coupling diagnostic",
        "",
        "**Internal diagnostic only; not a formal paper result.**",
        "",
        "Joint is inherited Full-4E OSRAM with classification plus JEPA. The",
        "Emotion-only condition changes only `training_objective` and removes",
        "the JEPA gradient from the training objective.",
        "",
        "| Condition | 8-rate mean | High-missing mean (0.5/0.6/0.7) |",
        "|---|---:|---:|",
        f"| Joint | {100 * mean(values('Joint')):.3f}% | {100 * mean(values('Joint', high)):.3f}% |",
        f"| Emotion-only | {100 * mean(values('Emotion-only')):.3f}% | {100 * mean(values('Emotion-only', high)):.3f}% |",
        "",
        "## Strict one-checkpoint cross-rate check",
        "",
        "| Condition | mean | selected epochs (66,67,68,69,70) |",
        "|---|---:|---|",
        f"| Joint | {100 * mean(joint_scores):.3f}% | {joint_epochs} |",
        f"| Emotion-only | {100 * mean(emotion_scores):.3f}% | {emotion_epochs} |",
        f"| Δ | {100 * (mean(emotion_scores) - mean(joint_scores)):+.3f} percentage points | — |",
        "",
        "| Rate | Joint | Emotion-only | Δ | positive seeds |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['rate']} | {100 * float(row['Joint_mean']):.3f}% ± {100 * float(row['Joint_std']):.3f} | "
            f"{100 * float(row['Emotion_only_mean']):.3f}% ± {100 * float(row['Emotion_only_std']):.3f} | "
            f"{100 * float(row['delta']):+.3f} | {row['positive_seed_count']}/5 |"
        )
    lines += [
        "",
        "The result distinguishes objective coupling from predictor architecture.",
        "A positive Emotion-only delta means the current JEPA gradient is noisy",
        "for OSRAM; a positive Joint delta means JEPA is useful regularization.",
        "",
    ]
    (root / "RESULT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
