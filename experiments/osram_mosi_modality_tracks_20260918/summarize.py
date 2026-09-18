"""Summarize the MOSI modality-track no-JEPA diagnostic."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path("/data2/yb/remote_experiments/osram_mosi_modality_tracks_20260918")
BASE = Path("/data2/yb/remote_experiments/osram_causal_nojepa_20260910/mosi")
SEEDS = (66, 67, 68, 69, 70)
RATES = tuple(f"{value / 10:.1f}" for value in range(8))
PATTERNS = {1: "V", 2: "T", 3: "TV", 4: "A", 5: "AV", 6: "AT", 7: "ATV"}
PATTERN_ORDER = ("A", "T", "V", "AT", "AV", "TV", "ATV")
GROUP_PATTERNS = {"T-missing": {"A", "V", "AV"}, "T-present": {"T", "AT", "TV", "ATV"}}


def read_json(path: Path):
    return json.loads(path.read_text())


def mean_std(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1)) if values.size > 1 else 0.0


def load_predictions(root: Path, seed: int, rate: str):
    path = root / f"seed_{seed}" / f"predictions_miss_{rate.replace('.', 'p')}.npz"
    with np.load(path) as data:
        return {name: data[name].copy() for name in data.files}


def wf1(labels, predictions):
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    selected = np.isfinite(labels) & np.isfinite(predictions) & (labels != 0)
    if not selected.any():
        return None
    return float(f1_score(labels[selected] > 0, predictions[selected] > 0, average="weighted", zero_division=0))


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def rate_rows():
    rows = []
    for rate in RATES:
        for seed in SEEDS:
            base = read_json(BASE / f"seed_{seed}" / "metrics.json")["test"][rate]
            cur_metrics = read_json(ROOT / "mosi" / f"seed_{seed}" / "metrics.json")
            cur = cur_metrics["test"][rate]
            rows.append({
                "seed": seed,
                "rate": rate,
                "no_jepa_wf1": base["weighted_f1"],
                "tracks_wf1": cur["weighted_f1"],
                "no_jepa_accuracy": base["accuracy"],
                "tracks_accuracy": cur["accuracy"],
                "delta": cur["weighted_f1"] - base["weighted_f1"],
                "accuracy_delta": cur["accuracy"] - base["accuracy"],
                "selected_epoch": cur_metrics.get("selected_epoch_by_rate", {}).get(rate),
            })
    return rows


def pattern_rows():
    stores = {
        name: {pattern: {"labels": [], "predictions": []} for pattern in PATTERN_ORDER}
        for name in ("no-JEPA", "modality-tracks")
    }
    rows = []
    for variant, root in (("no-JEPA", BASE), ("modality-tracks", ROOT / "mosi")):
        for seed in SEEDS:
            for rate in RATES:
                data = load_predictions(root, seed, rate)
                availability = data["availability"]
                labels = data["labels"]
                predictions = data["predictions"]
                pattern_ids = availability[:, 0].astype(int) * 4 + availability[:, 1].astype(int) * 2 + availability[:, 2].astype(int)
                for pattern_id, pattern in PATTERNS.items():
                    selected = pattern_ids == pattern_id
                    kept = selected & (labels != 0)
                    if not kept.any():
                        continue
                    stores[variant][pattern]["labels"].extend(labels[kept].tolist())
                    stores[variant][pattern]["predictions"].extend(predictions[kept].tolist())
                    rows.append({
                        "variant": variant,
                        "seed": seed,
                        "rate": rate,
                        "pattern": pattern,
                        "count": int(kept.sum()),
                        "excluded_original_neutral": int((selected & (labels == 0)).sum()),
                        "weighted_f1": wf1(labels[kept], predictions[kept]),
                    })
    return rows, stores


def aggregate(store, names):
    labels = np.asarray([x for name in names for x in store[name]["labels"]])
    predictions = np.asarray([x for name in names for x in store[name]["predictions"]])
    scores = [wf1(store[name]["labels"], store[name]["predictions"]) for name in names if store[name]["labels"]]
    return {"pattern_macro": float(np.mean(scores)), "sample_pooled": wf1(labels, predictions), "sample_count": int(labels.size)}


def grouped(rows, rates):
    pairs = []
    for seed in SEEDS:
        current = [r["tracks_wf1"] for r in rows if r["seed"] == seed and r["rate"] in rates]
        baseline = [r["no_jepa_wf1"] for r in rows if r["seed"] == seed and r["rate"] in rates]
        pairs.append((np.mean(baseline), np.mean(current)))
    return {
        "no_jepa_mean": float(np.mean([x[0] for x in pairs])),
        "tracks_mean": float(np.mean([x[1] for x in pairs])),
        "delta": float(np.mean([x[1] - x[0] for x in pairs])),
        "positive_seeds": int(sum(x[1] > x[0] for x in pairs)),
    }


def main():
    rows = rate_rows()
    pattern_data, stores = pattern_rows()
    per_rate = []
    for rate in RATES:
        current = [r["tracks_wf1"] for r in rows if r["rate"] == rate]
        baseline = [r["no_jepa_wf1"] for r in rows if r["rate"] == rate]
        m0, s0 = mean_std(baseline)
        m1, s1 = mean_std(current)
        per_rate.append({
            "rate": rate,
            "no_jepa_mean": m0,
            "no_jepa_std": s0,
            "tracks_mean": m1,
            "tracks_std": s1,
            "delta": m1 - m0,
            "positive_seeds": int(sum(x > y for x, y in zip(current, baseline))),
        })
    summary = {
        "status": "complete",
        "selection_protocol": "per-rate-test-oracle",
        "configuration_delta": {"osram_readout_fusion": ["flat", "modality-tracks"]},
        "rate": per_rate,
        "overall": grouped(rows, RATES),
        "high_missing": grouped(rows, ("0.5", "0.6", "0.7")),
        "text_groups": {
            group: {
                variant: aggregate(stores[variant], names)
                for variant in ("no-JEPA", "modality-tracks")
            }
            for group, names in GROUP_PATTERNS.items()
        },
        "patterns": {
            pattern: {
                variant: aggregate(stores[variant], (pattern,))
                for variant in ("no-JEPA", "modality-tracks")
            }
            for pattern in PATTERN_ORDER
        },
        "pattern_rows": pattern_data,
    }
    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    write_csv(ROOT / "results" / "per_seed_rate.csv", rows)
    write_csv(ROOT / "results" / "pattern_per_seed.csv", pattern_data)
    (ROOT / "results" / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    lines = [
        "# MOSI modality-track no-JEPA diagnostic",
        "",
        "Only the emotion readout local path changes: OSRAM fused-node query/read/write and all JEPA/MMoE paths remain unchanged. Missing modality tracks are zeroed, then modality tracks, Base, missing-masked Gap contexts and availability are sent to one flat MLP.",
        "This is an internal Test-oracle diagnostic, not a formal paper result. Epoch selection is independent per seed and missing rate.",
        "",
        "## Per-rate weighted F1",
        "",
        "| rate | no-JEPA | modality-tracks | delta | positive seeds |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in per_rate:
        lines.append(f"| {row['rate']} | {row['no_jepa_mean']*100:.2f}±{row['no_jepa_std']*100:.2f} | {row['tracks_mean']*100:.2f}±{row['tracks_std']*100:.2f} | {row['delta']*100:+.2f} | {row['positive_seeds']}/5 |")
    lines += [
        "",
        f"8-rate mean: no-JEPA {summary['overall']['no_jepa_mean']*100:.2f}%, modality-tracks {summary['overall']['tracks_mean']*100:.2f}% ({summary['overall']['delta']*100:+.2f} pp; {summary['overall']['positive_seeds']}/5 seeds positive).",
        f"High-missing (.5/.6/.7): no-JEPA {summary['high_missing']['no_jepa_mean']*100:.2f}%, modality-tracks {summary['high_missing']['tracks_mean']*100:.2f}% ({summary['high_missing']['delta']*100:+.2f} pp; {summary['high_missing']['positive_seeds']}/5 seeds positive).",
        "",
        "## T-missing vs T-present",
        "",
        "| group | aggregation | no-JEPA | modality-tracks | delta |",
        "|---|---|---:|---:|---:|",
    ]
    for group, values in summary["text_groups"].items():
        for aggregation in ("pattern_macro", "sample_pooled"):
            left, right = values["no-JEPA"][aggregation], values["modality-tracks"][aggregation]
            lines.append(f"| {group} | {aggregation} | {left*100:.2f} | {right*100:.2f} | {(right-left)*100:+.2f} |")
    lines += ["", "## Seven-pattern sample-pooled W-F1", "", "| pattern | no-JEPA | modality-tracks | delta |", "|---|---:|---:|---:|"]
    for pattern in PATTERN_ORDER:
        left = summary["patterns"][pattern]["no-JEPA"]["sample_pooled"]
        right = summary["patterns"][pattern]["modality-tracks"]["sample_pooled"]
        lines.append(f"| {pattern} | {left*100:.2f} | {right*100:.2f} | {(right-left)*100:+.2f} |")
    (ROOT / "RESULT.md").write_text("\n".join(lines) + "\n")
    print((ROOT / "RESULT.md").read_text())


if __name__ == "__main__":
    main()
