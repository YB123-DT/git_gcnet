"""Summarize the completed frozen L1/L2 MOSI diagnostic."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from gcnet_missing_m3.train_gcnet import _metrics

REMOTE = Path("/data2/yb/remote_experiments/osram_joint_frozen_reinjection_20260922")
BASE = REPO / "experiments/osram_no_aux_cfg84_cyclic_no0_20260920/results"
OUT = REPO / "experiments/osram_joint_frozen_reinjection_20260922/results"
SEEDS = (66, 67, 68)
RATES = ("0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7")
PATTERNS = {1: "V", 2: "T", 3: "TV", 4: "A", 5: "AV", 6: "AT", 7: "ATV"}


def load_metrics(variant: str, seed: int) -> dict:
    path = REMOTE / ("l1" if variant == "L1" else "") / f"seed_{seed}" / "metrics.json"
    return json.loads(path.read_text())


def load_predictions(variant: str, seed: int, rate: str):
    path = REMOTE / ("l1" if variant == "L1" else "") / f"seed_{seed}" / (
        f"predictions_miss_{rate.replace('.', 'p')}.npz"
    )
    with np.load(path) as archive:
        return archive["predictions"], archive["labels"], archive["availability"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    per_rate = []
    pattern_rows = []
    for seed in SEEDS:
        l1 = load_metrics("L1", seed)
        l2 = load_metrics("L2", seed)
        baseline = json.loads((BASE / f"seed_{seed}" / "metrics.json").read_text())
        for rate in RATES:
            l1_score = 100.0 * float(l1["test"][rate]["weighted_f1"])
            l2_score = 100.0 * float(l2["test"][rate]["weighted_f1"])
            no_jepa = 100.0 * float(baseline["test"][rate]["weighted_f1"])
            per_rate.append({
                "seed": seed,
                "rate": rate,
                "l1_frozen_no_reinjection_wf1": l1_score,
                "l2_frozen_reinjection_wf1": l2_score,
                "l2_minus_l1": l2_score - l1_score,
                "no_jepa_trainable_projector_wf1": no_jepa,
                "l1_selected_epoch": l1["selected_epoch_by_rate"][rate],
                "l2_selected_epoch": l2["selected_epoch_by_rate"][rate],
            })
            for variant in ("L1", "L2"):
                predictions, labels, availability = load_predictions(variant, seed, rate)
                observed_id = (
                    availability[:, 0].astype(np.int64) * 4
                    + availability[:, 1].astype(np.int64) * 2
                    + availability[:, 2].astype(np.int64)
                )
                for pattern_id, pattern in PATTERNS.items():
                    selected = observed_id == pattern_id
                    if not bool(selected.any()):
                        continue
                    metrics = _metrics(
                        "CMUMOSI", labels[selected], predictions[selected], "regression"
                    )
                    pattern_rows.append({
                        "variant": variant,
                        "seed": seed,
                        "rate": rate,
                        "pattern": pattern,
                        "count": int(selected.sum()),
                        "weighted_f1": 100.0 * metrics["weighted_f1"],
                        "mae": metrics["mae"],
                    })

    with (OUT / "per_seed_rate.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_rate[0]))
        writer.writeheader()
        writer.writerows(per_rate)
    with (OUT / "pattern_per_seed.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pattern_rows[0]))
        writer.writeheader()
        writer.writerows(pattern_rows)

    summary = []
    for rate in RATES:
        rows = [row for row in per_rate if row["rate"] == rate]
        l1 = np.asarray([row["l1_frozen_no_reinjection_wf1"] for row in rows])
        l2 = np.asarray([row["l2_frozen_reinjection_wf1"] for row in rows])
        base = np.asarray([row["no_jepa_trainable_projector_wf1"] for row in rows])
        summary.append({
            "rate": rate,
            "L1_mean": float(l1.mean()),
            "L1_std": float(l1.std(ddof=1)),
            "L2_mean": float(l2.mean()),
            "L2_std": float(l2.std(ddof=1)),
            "L2_minus_L1_mean": float((l2 - l1).mean()),
            "L2_positive_seeds": int((l2 > l1).sum()),
            "NoJEPA_mean": float(base.mean()),
        })
    with (OUT / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    all_l1 = np.asarray([row["l1_frozen_no_reinjection_wf1"] for row in per_rate])
    all_l2 = np.asarray([row["l2_frozen_reinjection_wf1"] for row in per_rate])
    high_l1 = np.asarray([
        row["l1_frozen_no_reinjection_wf1"] for row in per_rate if row["rate"] in {"0.5", "0.6", "0.7"}
    ])
    high_l2 = np.asarray([
        row["l2_frozen_reinjection_wf1"] for row in per_rate if row["rate"] in {"0.5", "0.6", "0.7"}
    ])
    lines = [
        "# Frozen joint-pretraining reinjection (MOSI, internal diagnostic)",
        "",
        "This run uses the three-dataset utterance-level pretraining checkpoint "
        "for frozen `P_A/P_T/P_V` and frozen SourceOnlyM3Predictor/MMoE. L1 "
        "and L2 use the same frozen representation and independent per-rate "
        "Test-oracle selection; L2 additionally feeds predicted missing slots "
        "through `CompletedReadFusion` before the single causal OSRAM path.",
        "",
        "> Internal diagnostic only; not a formal paper result. Test-oracle "
        "selection is used exactly as requested.",
        "",
        "## Paired overall summary",
        "",
        f"- L1 8-rate mean: **{all_l1.mean():.3f}%** (sample SD over 24 seed-rate cells {all_l1.std(ddof=1):.3f})",
        f"- L2 8-rate mean: **{all_l2.mean():.3f}%** (sample SD {all_l2.std(ddof=1):.3f})",
        f"- L2 − L1: **{(all_l2 - all_l1).mean():+.3f} pp**, positive in {int((all_l2 > all_l1).sum())}/24 paired cells",
        f"- L1 high-missing (0.5/0.6/0.7): **{high_l1.mean():.3f}%**",
        f"- L2 high-missing (0.5/0.6/0.7): **{high_l2.mean():.3f}%**",
        "",
        "## Per-rate means",
        "",
        "| rate | L1 frozen/no reinjection | L2 frozen/reinjection | Δ L2−L1 | positive seeds | trainable-projector No-JEPA reference |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['rate']} | {row['L1_mean']:.3f}±{row['L1_std']:.3f} | "
            f"{row['L2_mean']:.3f}±{row['L2_std']:.3f} | {row['L2_minus_L1_mean']:+.3f} | "
            f"{row['L2_positive_seeds']}/3 | {row['NoJEPA_mean']:.3f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The reinjection comparison is L2−L1, not L2 versus the trainable-projector "
        "No-JEPA reference. The latter is shown only to expose the cost of freezing "
        "the jointly pretrained projector space. Any future end-to-end adaptation "
        "must therefore retain L1 as its control.",
        "",
        "Pattern-level values are in `results/pattern_per_seed.csv`; the full paired "
        "table is in `results/per_seed_rate.csv`.",
    ])
    (OUT.parent / "RESULT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
