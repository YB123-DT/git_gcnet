"""Validate and summarize the locked 40-pair RTDR full-rate experiment."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping

from experiments.second_graph_aggregation_iemocap6.summarize_gate import (
    _candidate_job_path,
    _original_job_path,
    _write_atomic,
    collect_job,
    summarize_candidate,
)


RATES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
SEEDS = (66, 67, 68, 69, 70)


def collect_full_grid(original_root: Path, phase_b_root: Path) -> Dict[str, list]:
    """Collect exactly 40 inherited-Original/RTDR pairs with locked masks."""
    rows = {"original": [], "rtdr": []}
    for rate in RATES:
        for seed in SEEDS:
            original = collect_job(
                _original_job_path(original_root, rate, seed),
                "original",
                rate,
                seed,
                historical_original=True,
            )
            candidate = collect_job(
                _candidate_job_path(phase_b_root, "rtdr", rate, seed),
                "rtdr",
                rate,
                seed,
            )
            if candidate["mask_sha256"] != original["mask_sha256"]:
                raise ValueError(
                    "paired mask SHA256 mismatch for rtdr at rate={}, seed={}".format(
                        rate, seed
                    )
                )
            rows["original"].append(original)
            rows["rtdr"].append(candidate)
    return rows


def summarize_full(rows: Mapping[str, list]) -> Dict[str, Any]:
    paired = summarize_candidate(
        "rtdr", rows["original"], rows["rtdr"], rates=RATES, seeds=SEEDS
    )
    positive_rates = sum(
        row["mean_delta"] > 0.0 for row in paired["rate_means"].values()
    )
    positive_seeds = sum(
        delta > 0.0 for delta in paired["seed_macro_deltas"].values()
    )
    all_finite = paired["gate"]["all_finite"]
    all_noncollapsed = paired["gate"]["all_noncollapsed"]
    stable_positive = (
        paired["macro_delta"] > 0.0
        and positive_rates >= 6
        and positive_seeds >= 3
        and all_finite
        and all_noncollapsed
    )
    return {
        "experiment": "RTDR full-rate vs inherited Original, locked IEMOCAPSix fold 5",
        "rates": list(RATES),
        "seeds": list(SEEDS),
        "tasks": paired["tasks"],
        "rate_means": paired["rate_means"],
        "seed_macro_deltas": paired["seed_macro_deltas"],
        "overall_macro_delta": paired["macro_delta"],
        "positive_rate_means": int(positive_rates),
        "positive_seed_macros": int(positive_seeds),
        "all_finite": bool(all_finite),
        "all_noncollapsed": bool(all_noncollapsed),
        "stable_positive": bool(stable_positive),
    }


def _render(summary: Mapping[str, Any], language: str) -> str:
    stable = summary["stable_positive"]
    if language == "zh":
        lines = [
            "# RTDR 完整 missing-rate 配对结果",
            "",
            "`stable_positive = {}`。该字段仅表示：总体宏差值为正、至少 6/8 个 rate 均值为正、至少 3/5 个 seed 宏差值为正、全部运行有限且六类不坍塌；它不是新的晋级 PASS。".format(str(stable).lower()),
        ]
    else:
        lines = [
            "# RTDR Full Missing-Rate Paired Results",
            "",
            "`stable_positive = {}` means only that the overall macro delta is positive, at least 6/8 rate means and 3/5 seed macros are positive, and every run is finite and non-collapsed. It is not a new advancement PASS.".format(str(stable).lower()),
        ]
    lines.extend(
        [
            "",
            "Overall macro delta: {:+.9f}; positive rates: {}/8; positive seeds: {}/5.".format(
                summary["overall_macro_delta"],
                summary["positive_rate_means"],
                summary["positive_seed_macros"],
            ),
            "",
            "## Missing-rate summary",
            "",
            "| rate | Original F1 | RTDR F1 | paired delta |",
            "|---:|---:|---:|---:|",
        ]
    )
    for rate in RATES:
        row = summary["rate_means"][str(float(rate))]
        lines.append(
            "| {:.1f} | {:.9f} | {:.9f} | {:+.9f} |".format(
                rate, row["original_mean"], row["candidate_mean"], row["mean_delta"]
            )
        )
    lines.extend(
        [
            "",
            "## Seed macro",
            "",
            "| seed | macro delta |",
            "|---:|---:|",
        ]
    )
    for seed in SEEDS:
        lines.append(
            "| {} | {:+.9f} |".format(seed, summary["seed_macro_deltas"][str(seed)])
        )
    lines.extend(
        [
            "",
            "## Task evidence",
            "",
            "| rate | seed | Original F1 | RTDR F1 | delta | mask SHA256 |",
            "|---:|---:|---:|---:|---:|---|",
        ]
    )
    for task in summary["tasks"]:
        lines.append(
            "| {rate:.1f} | {seed} | {original:.9f} | {rtdr:.9f} | {delta:+.9f} | `{mask}` |".format(
                rate=task["rate"],
                seed=task["seed"],
                original=task["original"]["weighted_f1"],
                rtdr=task["candidate"]["weighted_f1"],
                delta=task["delta_weighted_f1"],
                mask=task["candidate"]["mask_sha256"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_full_outputs(output_directory: Path, summary: Mapping[str, Any]) -> None:
    output_directory = Path(output_directory)
    _write_atomic(
        output_directory / "summary.json",
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    _write_atomic(output_directory / "RESULTS.md", _render(summary, "main"))
    _write_atomic(output_directory / "RESULTS.zh.md", _render(summary, "zh"))
    _write_atomic(output_directory / "RESULTS.en.md", _render(summary, "en"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and summarize the locked full-rate RTDR experiment"
    )
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--phase-b-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    rows = collect_full_grid(arguments.original_root, arguments.phase_b_root)
    write_full_outputs(arguments.output_dir, summarize_full(rows))


if __name__ == "__main__":
    main()
