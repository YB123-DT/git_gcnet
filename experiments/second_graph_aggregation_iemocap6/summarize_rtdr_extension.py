"""Validate and summarize the locked 15-pair RTDR extension experiment."""

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


RATES = (0.0, 0.5, 0.7)
SEEDS = (66, 67, 68, 69, 70)


def collect_extension_grid(original_root: Path, phase_b_root: Path) -> Dict[str, list]:
    """Collect exactly 15 inherited-Original/RTDR pairs with locked masks."""
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


def summarize_extension(rows: Mapping[str, list]) -> Dict[str, Any]:
    candidate = summarize_candidate(
        "rtdr", rows["original"], rows["rtdr"], rates=RATES, seeds=SEEDS
    )
    return {
        "experiment": "RTDR extension vs inherited Original, locked IEMOCAPSix fold 5",
        "rates": list(RATES),
        "seeds": list(SEEDS),
        "status": "PASS" if candidate["gate"]["passed"] else "FAIL",
        "candidate": candidate,
    }


def _render(summary: Mapping[str, Any], language: str) -> str:
    candidate = summary["candidate"]
    status = summary["status"]
    if language == "zh":
        lines = [
            "# RTDR 补充配对实验结果",
            "",
            "状态：**{}**。PASS 要求三个 missing rate 的均值差均为正、总体 seed 宏平均为正、至少 3/5 个 seed 宏平均为正，并且全部运行有限且六类不坍塌。".format(status),
            "",
            "## Missing-rate 汇总",
            "",
            "| missing rate | Original F1 | RTDR F1 | 配对差值 |",
            "|---:|---:|---:|---:|",
        ]
    else:
        lines = [
            "# RTDR Extension Paired Results",
            "",
            "Status: **{}**. PASS requires a positive mean delta at all three missing rates, a positive overall seed macro, at least 3/5 positive seed macros, finite runs, and six-class non-collapse throughout.".format(status),
            "",
            "## Missing-rate summary",
            "",
            "| missing rate | Original F1 | RTDR F1 | paired delta |",
            "|---:|---:|---:|---:|",
        ]
    for rate in RATES:
        row = candidate["rate_means"][str(float(rate))]
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
            "| {} | {:+.9f} |".format(
                seed, candidate["seed_macro_deltas"][str(seed)]
            )
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
    for task in candidate["tasks"]:
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


def write_extension_outputs(output_directory: Path, summary: Mapping[str, Any]) -> None:
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
        description="Validate and summarize the locked RTDR extension"
    )
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--phase-b-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    rows = collect_extension_grid(arguments.original_root, arguments.phase_b_root)
    write_extension_outputs(arguments.output_dir, summarize_extension(rows))


if __name__ == "__main__":
    main()
