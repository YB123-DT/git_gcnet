"""Summarize completed seed-66 screening records without changing selection."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

RATES = tuple(f"{i / 10:.1f}" for i in range(8))
ROOT = Path("/data2/yb/remote_experiments/osram_mosi_hparam_sweep_20260918_parallel")


def _best_record(path: Path) -> dict[str, object]:
    history = json.loads((path / "history.json").read_text())
    best: dict[str, dict[str, object]] = {}
    for row in history:
        metrics = row.get("test_oracle", {})
        for rate in RATES:
            score = metrics.get(rate, {}).get("weighted_f1")
            if isinstance(score, (int, float)) and math.isfinite(float(score)):
                if rate not in best or float(score) > float(best[rate]["weighted_f1"]):
                    best[rate] = {"epoch": int(row["epoch"]), "weighted_f1": float(score)}
    if set(best) != set(RATES):
        raise ValueError(f"missing best rate in {path}")
    return best


def collect(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted((root / "screen_seed66").glob("cfg*")):
        if not all((path / name).exists() for name in ("config.json", "history.json", "metrics.json", "PROVENANCE.json")):
            continue
        provenance = json.loads((path / "PROVENANCE.json").read_text())
        if provenance.get("status") != "complete":
            continue
        config = json.loads((path / "config.json").read_text())
        metrics = json.loads((path / "metrics.json").read_text())
        best = _best_record(path)
        scores = [float(best[rate]["weighted_f1"]) for rate in RATES]
        high = [float(best[rate]["weighted_f1"]) for rate in ("0.5", "0.6", "0.7")]
        rows.append({
            "spec_id": path.name,
            "group": provenance.get("spec", {}).get("group"),
            "learning_rate": config.get("learning_rate"),
            "batch_size": config.get("batch_size"),
            "optimizer": config.get("optimizer", "adam"),
            "lr_schedule": config.get("lr_schedule", "constant"),
            "weight_decay": config.get("weight_decay"),
            "dropout": config.get("dropout"),
            "projector_dropout": config.get("projector_dropout"),
            "epochs": config.get("epochs"),
            "latent_dim": config.get("latent_dim"),
            "output_dim": config.get("osram_output_dim"),
            "heads": config.get("osram_num_heads"),
            "key_dim": config.get("osram_key_dim"),
            "value_dim": config.get("osram_value_dim"),
            "all8_mean": statistics.mean(scores),
            "high_missing_mean": statistics.mean(high),
            "parameter_count": metrics.get("parameter_count"),
            "trainable_parameter_count": metrics.get("trainable_parameter_count"),
            "selected_epochs": json.dumps({rate: best[rate]["epoch"] for rate in RATES}, sort_keys=True),
        })
    return sorted(rows, key=lambda row: (-float(row["all8_mean"]), -float(row["high_missing_mean"])))


def write_report(rows: list[dict[str, object]], root: Path) -> None:
    out = root / "summary"
    out.mkdir(parents=True, exist_ok=True)
    write_rows = rows
    fields = list(write_rows[0].keys()) if write_rows else ["spec_id"]
    with (out / "screening_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(write_rows)
    payload = {
        "status": "complete" if len(rows) == 60 else "partial",
        "completed_configs": len(rows),
        "expected_configs": 60,
        "selection_protocol": "per-rate-test-oracle",
        "label": "INTERNAL DIAGNOSTIC ONLY; NOT A FORMAL PAPER RESULT",
        "rows": rows,
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = ["# MOSI causal OSRAM hyperparameter screening", "",
             "Internal diagnostic only; not a formal paper result.", "",
             "Each configuration is seed 66, cyclic missing-rate training, and independent Test-oracle epoch selection for every rate.", "",
             "| rank | config | group | all-8 mean | high-missing mean | epochs | latent | output | heads | key/value |", "|---:|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for rank, row in enumerate(rows, 1):
        lines.append(f"| {rank} | {row['spec_id']} | {row['group']} | {100*float(row['all8_mean']):.3f} | {100*float(row['high_missing_mean']):.3f} | {row['epochs']} | {row['latent_dim']} | {row['output_dim']} | {row['heads']} | {row['key_dim']}/{row['value_dim']} |")
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    rows = collect(args.root)
    write_report(rows, args.root)
    print(f"completed={len(rows)}/60")


if __name__ == "__main__":
    main()
