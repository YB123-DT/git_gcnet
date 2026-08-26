"""Summarize four mechanisms under one locked 3-rate by 5-seed protocol."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping

from experiments.second_graph_aggregation_iemocap6.summarize_gate import (
    CANDIDATE_ARMS,
    _candidate_job_path,
    _formal_root,
    _original_job_path,
    _write_atomic,
    collect_job,
    summarize_candidate,
)


RATES = (0.0, 0.5, 0.7)
SEEDS = (66, 67, 68, 69, 70)


def collect_uniform_grid(
    original_root: Path, phase_a_root: Path, phase_b_root: Path
) -> Dict[str, list]:
    """Collect exactly 15 mask-paired jobs for each of the four candidates."""
    rows = {"original": []}
    rows.update({arm: [] for arm in CANDIDATE_ARMS})
    arm_roots = {
        "genagg": Path(phase_a_root),
        "soft_medoid": Path(phase_a_root),
        "ssma": Path(phase_b_root),
        "rtdr": Path(phase_b_root),
    }
    for arm in ("genagg", "soft_medoid"):
        if (_formal_root(phase_b_root) / arm).exists():
            raise ValueError("Phase A arm {} was found under Phase B root".format(arm))
    for arm in ("ssma", "rtdr"):
        if (_formal_root(phase_a_root) / arm).exists():
            raise ValueError("Phase B arm {} was found under Phase A root".format(arm))

    for rate in RATES:
        for seed in SEEDS:
            original = collect_job(
                _original_job_path(original_root, rate, seed),
                "original",
                rate,
                seed,
                historical_original=True,
            )
            rows["original"].append(original)
            for arm in CANDIDATE_ARMS:
                candidate = collect_job(
                    _candidate_job_path(arm_roots[arm], arm, rate, seed),
                    arm,
                    rate,
                    seed,
                )
                if candidate["mask_sha256"] != original["mask_sha256"]:
                    raise ValueError(
                        "paired mask SHA256 mismatch for {} at rate={}, seed={}".format(
                            arm, rate, seed
                        )
                    )
                rows[arm].append(candidate)
    return rows


def _uniform_candidate(arm: str, rows: Mapping[str, list]) -> Dict[str, Any]:
    paired = summarize_candidate(
        arm, rows["original"], rows[arm], rates=RATES, seeds=SEEDS
    )
    positive_rates = sum(
        item["mean_delta"] > 0.0 for item in paired["rate_means"].values()
    )
    positive_seeds = sum(
        delta > 0.0 for delta in paired["seed_macro_deltas"].values()
    )
    all_finite = paired["gate"]["all_finite"]
    all_noncollapsed = paired["gate"]["all_noncollapsed"]
    uniform_stable = (
        paired["macro_delta"] > 0.0
        and positive_rates == len(RATES)
        and positive_seeds >= 3
        and all_finite
        and all_noncollapsed
    )
    return {
        "arm": arm,
        "tasks": paired["tasks"],
        "rate_means": paired["rate_means"],
        "seed_macro_deltas": paired["seed_macro_deltas"],
        "overall_macro_delta": paired["macro_delta"],
        "positive_rate_means": int(positive_rates),
        "positive_seed_macros": int(positive_seeds),
        "all_finite": bool(all_finite),
        "all_noncollapsed": bool(all_noncollapsed),
        "uniform_stable": bool(uniform_stable),
    }


def summarize_uniform(rows: Mapping[str, list]) -> Dict[str, Any]:
    return {
        "experiment": (
            "Four second-graph mechanisms vs inherited Original, locked "
            "IEMOCAPSix fold 5 uniform 3-rate protocol"
        ),
        "rates": list(RATES),
        "seeds": list(SEEDS),
        "criterion": {
            "name": "uniform_stable",
            "overall_macro_delta_positive": True,
            "required_positive_rate_means": 3,
            "required_positive_seed_macros": 3,
            "requires_finite": True,
            "requires_noncollapsed": True,
            "initial_gate_is_unchanged": True,
        },
        "candidates": {
            arm: _uniform_candidate(arm, rows) for arm in CANDIDATE_ARMS
        },
    }


def _render(summary: Mapping[str, Any], language: str) -> str:
    if language == "zh":
        lines = [
            "# 四模块统一三档缺失率配对结果",
            "",
            "`uniform_stable` 要求总体宏差值为正、3/3 个 missing rate 均值为正、至少 3/5 个 seed 宏差值为正，并且全部运行有限且六类不坍塌。该字段不改写 initial gate。",
            "",
            "| 模块 | uniform_stable | 总体宏差值 | 正向 rate | 正向 seed |",
            "|---|---:|---:|---:|---:|",
        ]
    else:
        lines = [
            "# Uniform Three-Rate Paired Results for Four Mechanisms",
            "",
            "`uniform_stable` requires a positive overall macro delta, positive means at all 3/3 missing rates, at least 3/5 positive seed macros, and finite non-collapsed runs. It does not revise the initial gate.",
            "",
            "| arm | uniform_stable | overall macro delta | positive rates | positive seeds |",
            "|---|---:|---:|---:|---:|",
        ]
    for arm in CANDIDATE_ARMS:
        result = summary["candidates"][arm]
        lines.append(
            "| {} | {} | {:+.9f} | {}/3 | {}/5 |".format(
                arm,
                str(result["uniform_stable"]).lower(),
                result["overall_macro_delta"],
                result["positive_rate_means"],
                result["positive_seed_macros"],
            )
        )

    for arm in CANDIDATE_ARMS:
        result = summary["candidates"][arm]
        lines.extend(
            [
                "",
                "## {}".format(arm),
                "",
                "| rate | Original F1 | candidate F1 | paired delta |",
                "|---:|---:|---:|---:|",
            ]
        )
        for rate in RATES:
            item = result["rate_means"][str(float(rate))]
            lines.append(
                "| {:.1f} | {:.9f} | {:.9f} | {:+.9f} |".format(
                    rate,
                    item["original_mean"],
                    item["candidate_mean"],
                    item["mean_delta"],
                )
            )
        lines.extend(["", "| seed | macro delta |", "|---:|---:|"])
        for seed in SEEDS:
            lines.append(
                "| {} | {:+.9f} |".format(
                    seed, result["seed_macro_deltas"][str(seed)]
                )
            )
        lines.extend(
            [
                "",
                "| rate | seed | Original F1 | candidate F1 | delta | mask SHA256 |",
                "|---:|---:|---:|---:|---:|---|",
            ]
        )
        for task in result["tasks"]:
            lines.append(
                "| {rate:.1f} | {seed} | {original:.9f} | {candidate:.9f} | {delta:+.9f} | `{mask}` |".format(
                    rate=task["rate"],
                    seed=task["seed"],
                    original=task["original"]["weighted_f1"],
                    candidate=task["candidate"]["weighted_f1"],
                    delta=task["delta_weighted_f1"],
                    mask=task["candidate"]["mask_sha256"],
                )
            )
    lines.append("")
    return "\n".join(lines)


def write_uniform_outputs(output_directory: Path, summary: Mapping[str, Any]) -> None:
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
        description="Summarize the locked uniform 3-rate by 5-seed experiment"
    )
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--phase-a-root", type=Path, required=True)
    parser.add_argument("--phase-b-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    rows = collect_uniform_grid(
        arguments.original_root, arguments.phase_a_root, arguments.phase_b_root
    )
    write_uniform_outputs(arguments.output_dir, summarize_uniform(rows))


if __name__ == "__main__":
    main()
