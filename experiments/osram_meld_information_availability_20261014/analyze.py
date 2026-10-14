"""Aggregate frozen MELD probes without changing their paired cohorts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .probe import PROBE_PAIRS


METRICS = ("accuracy", "weighted_f1")


def _summary(values) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("summary requires finite values")
    return {
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "values": array.tolist(),
    }


def _variant_summary(rows: list[dict]) -> dict:
    seeds = sorted({int(row["seed"]) for row in rows})
    rates = sorted({float(row["rate"]) for row in rows})
    lookup = {(int(row["seed"]), float(row["rate"])): row for row in rows}
    if len(lookup) != len(rows) or any((seed, rate) not in lookup for seed in seeds for rate in rates):
        raise ValueError("variant grid is incomplete or duplicated")
    per_rate = {
        f"{rate:.1f}": {
            metric: _summary(lookup[(seed, rate)][metric] for seed in seeds) for metric in METRICS
        }
        for rate in rates
    }

    def across(selected_rates):
        return {
            metric: _summary(
                np.mean([lookup[(seed, rate)][metric] for rate in selected_rates]) for seed in seeds
            )
            for metric in METRICS
        }

    high = [rate for rate in rates if rate in (0.5, 0.6, 0.7)]
    return {
        "seeds": seeds,
        "rates": rates,
        "per_rate": per_rate,
        "eight_rate": across(rates),
        "high_missing": across(high) if high else None,
    }


def aggregate_records(records: list[dict]) -> dict:
    if not records:
        raise ValueError("no probe records")
    variants = {}
    for variant in sorted({row["variant"] for row in records}):
        variants[variant] = _variant_summary([row for row in records if row["variant"] == variant])
    paired = {}
    for real, donor in PROBE_PAIRS.items():
        if real not in variants or donor not in variants:
            continue
        real_rows = {(int(row["seed"]), float(row["rate"])): row for row in records if row["variant"] == real}
        donor_rows = {(int(row["seed"]), float(row["rate"])): row for row in records if row["variant"] == donor}
        if real_rows.keys() != donor_rows.keys():
            raise ValueError(f"paired grid differs for {real}/{donor}")
        delta_rows = []
        for key in sorted(real_rows):
            left, right = real_rows[key], donor_rows[key]
            if left["n"] != right["n"]:
                raise ValueError(f"paired coverage differs for {real}/{donor} at {key}")
            delta_rows.append({
                "seed": key[0], "rate": key[1], "variant": real,
                "n": int(left["n"]),
                **{metric: float(left[metric] - right[metric]) for metric in METRICS},
            })
        paired[real] = {
            **_variant_summary(delta_rows),
            "control": donor,
            "coverage": {
                "minimum_n": min(row["n"] for row in delta_rows),
                "maximum_n": max(row["n"] for row in delta_rows),
            },
        }
    return {"variants": variants, "paired_deltas": paired}


def classify_case(*, raw_history_signal: bool, memory_signal: bool, conditional_memory_signal: bool) -> str:
    if not raw_history_signal and not memory_signal:
        return "A"
    if raw_history_signal and not memory_signal:
        return "B"
    if memory_signal and conditional_memory_signal:
        return "C"
    return "mixed"


def _stable_positive(aggregate: dict, name: str) -> bool:
    values = aggregate["paired_deltas"][name]["eight_rate"]["weighted_f1"]["values"]
    return bool(values) and min(values) > 0


def load_records(root: Path) -> list[dict]:
    status = json.loads((root / "STATUS.json").read_text())
    if status.get("status") != "complete":
        raise ValueError("experiment status is not complete")
    records = []
    for rate_dir in sorted(root.glob("rate_*")):
        rate = float(rate_dir.name.removeprefix("rate_").replace("p", "."))
        probes = json.loads((rate_dir / "probes" / "PROBES.json").read_text())
        for run in probes["runs"]:
            records.append({
                "seed": int(run["seed"]), "rate": rate, "variant": run["variant"],
                "n": int(run["metrics"]["n"]),
                "accuracy": float(run["metrics"]["accuracy"]),
                "weighted_f1": float(run["metrics"]["weighted_f1"]),
            })
    return records


def render_markdown(aggregate: dict, decision: dict) -> str:
    lines = [
        "# MELD Information Availability Probe",
        "",
        "**INTERNAL DIAGNOSTIC ONLY.** The backbone and probe epochs are Test-oracle selected.",
        "The OSRAM backbone is frozen; only offline seven-class probes are trained.",
        "",
        "## Eight-rate and high-missing results",
        "",
        "| Probe | ACC mean±sd | W-F1 mean±sd | high ACC | high W-F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in aggregate["variants"].items():
        eight, high = values["eight_rate"], values["high_missing"]
        lines.append(
            f"| {name} | {100*eight['accuracy']['mean']:.3f}±{100*eight['accuracy']['sample_std']:.3f} "
            f"| {100*eight['weighted_f1']['mean']:.3f}±{100*eight['weighted_f1']['sample_std']:.3f} "
            f"| {100*high['accuracy']['mean']:.3f} | {100*high['weighted_f1']['mean']:.3f} |"
        )
    lines.extend(["", "## Paired real-minus-donor deltas", "", "| Real | Control | ΔACC | ΔW-F1 | min N |", "|---|---|---:|---:|---:|"])
    for name, values in aggregate["paired_deltas"].items():
        eight = values["eight_rate"]
        lines.append(
            f"| {name} | {values['control']} | {100*eight['accuracy']['mean']:+.3f} "
            f"| {100*eight['weighted_f1']['mean']:+.3f} | {values['coverage']['minimum_n']} |"
        )
    lines.extend([
        "", "## Diagnostic reading", "",
        f"- Rule-based outcome: **{decision['case']}**.",
        f"- Stable RawHistory signal across probe seeds: `{decision['raw_history_signal']}`.",
        f"- Stable Memory-only signal: `{decision['memory_signal']}`.",
        f"- Stable conditional Local+Memory signal: `{decision['conditional_memory_signal']}`.",
        "- Case C requires conditional Local+Memory value; Memory-only decodability is not enough.",
        "- P0/P1 behavioral interventions are reused as context and were not rerun here.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    args = parser.parse_args()
    aggregate = aggregate_records(load_records(args.input))
    raw = _stable_positive(aggregate, "RawHistory") and _stable_positive(aggregate, "LocalRawHistory")
    memory = _stable_positive(aggregate, "Memory")
    conditional = _stable_positive(aggregate, "LocalMemory")
    decision = {
        "raw_history_signal": raw,
        "memory_signal": memory,
        "conditional_memory_signal": conditional,
        "case": classify_case(
            raw_history_signal=raw, memory_signal=memory, conditional_memory_signal=conditional,
        ),
        "criterion": "strictly positive eight-rate W-F1 real-minus-donor delta for every probe seed",
    }
    payload = {"label": "INTERNAL DIAGNOSTIC ONLY", "aggregate": aggregate, "decision": decision}
    output_json = args.output_json or args.input / "SUMMARY.json"
    output_markdown = args.output_markdown or Path(__file__).with_name("RESULT.md")
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    output_markdown.write_text(render_markdown(aggregate, decision))


if __name__ == "__main__":
    main()
