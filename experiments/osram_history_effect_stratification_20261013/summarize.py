"""Pool folds within seed and report separate dataset/rate P0 diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments.osram_history_effect_stratification_20261013.evaluate import (
    COMBINATIONS, _metrics, _sha256, _write, per_utterance_effects, summarize_strata,
)


def aggregate(inputs: list[Path], *, allow_partial: bool = False) -> dict:
    runs = []
    seen = set()
    for directory in inputs:
        path = directory / "SUMMARY.json"
        summary = json.loads(path.read_text())
        if summary.get("status") != "complete" or summary.get("training") is not False or "delta_definition" not in summary or not summary.get("rates"):
            raise ValueError(f"not a complete frozen P0 summary: {path}")
        identity = (summary["dataset"], summary["seed"], summary["fold"])
        if identity in seen:
            raise ValueError(f"duplicate dataset/seed/fold: {identity}")
        seen.add(identity)
        runs.append((directory, summary))
    if not runs:
        raise ValueError("no P0 summaries found")
    datasets = {}
    for dataset in sorted({summary["dataset"] for _, summary in runs}):
        selected = [(directory, summary) for directory, summary in runs if summary["dataset"] == dataset]
        expected_folds = {1} if dataset == "MELD" else set(range(1, 6))
        seeds = sorted({summary["seed"] for _, summary in selected})
        rate_names = sorted({rate for _, summary in selected for rate in summary["rates"]}, key=float)
        output = {"seeds": seeds, "expected_folds": sorted(expected_folds), "rates": {}}
        for rate in rate_names:
            seed_summaries = {}
            pooled_seed_effects = []
            for seed in seeds:
                seed_runs = [(directory, summary) for directory, summary in selected if summary["seed"] == seed]
                folds = {summary["fold"] for _, summary in seed_runs if rate in summary["rates"]}
                if not folds <= expected_folds:
                    raise ValueError(f"unexpected folds for {dataset}: {sorted(folds - expected_folds)}")
                if not allow_partial and folds != expected_folds:
                    raise ValueError(f"{dataset} seed={seed} rate={rate} has folds {sorted(folds)}, expected {sorted(expected_folds)}")
                parts = []
                sources = []
                for directory, summary in sorted(seed_runs, key=lambda item: item[1]["fold"]):
                    if rate not in summary["rates"]:
                        continue
                    path = directory / summary["rates"][rate]["utterance_file"]
                    with np.load(path, allow_pickle=False) as loaded:
                        values = {name: loaded[name] for name in loaded.files}
                    effects = per_utterance_effects(values["labels"], {name: values[f"logits_{name}"] for name in COMBINATIONS})
                    for name, expected in effects.items():
                        np.testing.assert_array_equal(values[name], expected)
                    if parts and set(values) != set(parts[0]):
                        raise ValueError(f"artifact schema differs: {path}")
                    parts.append(values)
                    sources.append({"fold": summary["fold"], "path": str(path), "sha256": _sha256(path),
                                    "summary": str(directory / "SUMMARY.json"), "checkpoint_sha256": summary["rates"][rate]["checkpoint_sha256"]})
                if not parts:
                    continue
                pooled = {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}
                identities = list(zip(pooled["conversation_id"], pooled["position"]))
                if len(set(identities)) != len(identities):
                    raise ValueError(f"overlapping utterances across folds for {dataset} seed={seed} rate={rate}")
                result = summarize_strata(pooled)
                classes = pooled["logits_L"].shape[1]
                result["metrics"] = {name: _metrics(pooled["labels"], pooled[f"pred_{name}"], classes) for name in COMBINATIONS}
                result.update(folds=sorted(folds), sources=sources)
                seed_summaries[str(seed)] = result
                pooled_seed_effects.append(result["overall"])
            across_seeds = {}
            for effect in ("Base", "Gap", "Total"):
                across_seeds[effect] = {}
                for metric in ("mean_delta_ce", "correction_rate", "harm_rate", "mean_true_probability_change"):
                    values = np.asarray([row["effects"][effect][metric] for row in pooled_seed_effects])
                    across_seeds[effect][metric] = {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else None,
                                                  "seed_count": len(values), "seed_values": values.tolist()}
            across_seed_strata = {}
            for name in sorted({name for result in seed_summaries.values() for name in result["strata"]}):
                groups = sorted({group for result in seed_summaries.values() for group in result["strata"][name]})
                across_seed_strata[name] = {}
                for group in groups:
                    available = {seed: result["strata"][name][group] for seed, result in seed_summaries.items() if group in result["strata"][name]}
                    group_result = {"seeds": sorted(available), "counts_by_seed": {seed: row["count"] for seed, row in available.items()}, "effects": {}}
                    for effect in ("Base", "Gap", "Total"):
                        group_result["effects"][effect] = {}
                        for metric in ("mean_delta_ce", "correction_rate", "harm_rate", "mean_true_probability_change"):
                            values = np.asarray([row["effects"][effect][metric] for row in available.values()])
                            group_result["effects"][effect][metric] = {
                                "mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else None,
                                "seed_count": len(values), "seed_values": {seed: row["effects"][effect][metric] for seed, row in available.items()},
                            }
                    across_seed_strata[name][group] = group_result
            output["rates"][rate] = {"by_seed": seed_summaries, "across_seed_overall": across_seeds,
                                      "across_seed_strata": across_seed_strata}
        datasets[dataset] = output
    return {"status": "complete", "training": False, "allow_partial": allow_partial,
            "protocol": "INTERNAL DIAGNOSTIC ONLY; folds pooled within seed; datasets and rates kept separate",
            "uncertainty": "Across-seed sample std; utterance strata are descriptive and correlated within conversations. No significance claim.",
            "datasets": datasets}


def report_markdown(summary: dict) -> str:
    lines = ["# P0 frozen history effects", "", summary["protocol"], "", summary["uncertainty"], "",
             "Positive Delta means reduced CE. Base compares L→LB; Gap compares LB→LBG. Corrections and harms use all utterances as the denominator.", ""]
    for dataset, payload in summary["datasets"].items():
        lines.extend([f"## {dataset}", "", "| Rate | Seed | Folds | N | Delta Base | Delta Gap | Base correction / harm | Gap correction / harm |", "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"])
        for rate, row in payload["rates"].items():
            for seed, result in row["by_seed"].items():
                overall = result["overall"]
                base, gap = overall["effects"]["Base"], overall["effects"]["Gap"]
                lines.append(f"| {rate} | {seed} | {','.join(map(str, result['folds']))} | {overall['count']} | {base['mean_delta_ce']:.6f} | {gap['mean_delta_ce']:.6f} | {base['corrections']} / {base['harms']} | {gap['corrections']} / {gap['harms']} |")
        lines.extend(["", "Per-fold strata remain in original SUMMARY.json files. AGGREGATE.json contains each seed's pooled strata, probability changes, source hashes, and across-seed summaries.", ""])
    return "\n".join(lines)


def strata_rows(summary: dict) -> list[dict]:
    rows = []
    for dataset, payload in summary["datasets"].items():
        for rate, rate_result in payload["rates"].items():
            for seed, result in rate_result["by_seed"].items():
                for stratum, groups in result["strata"].items():
                    for group, row in groups.items():
                        for effect, values in row["effects"].items():
                            rows.append({"dataset": dataset, "seed": seed, "rate": rate, "folds": ",".join(map(str, result["folds"])),
                                         "stratum": stratum, "group": group, "count": row["count"], "effect": effect, **values})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new aggregate directory")
    parser.add_argument("--allow-partial", action="store_true", help="explicitly label missing folds; never silently omit them")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    inputs = [path.parent for path in sorted(args.input_root.rglob("SUMMARY.json"))]
    summary = aggregate(inputs, allow_partial=args.allow_partial)
    args.output.mkdir(parents=True)
    outputs = [(args.output, summary)] + [
        (args.output / dataset, dict(summary, datasets={dataset: payload}))
        for dataset, payload in summary["datasets"].items()
    ]
    for directory, payload in outputs:
        directory.mkdir(exist_ok=directory == args.output)
        _write(directory / "AGGREGATE.json", payload)
        with (directory / "REPORT.md").open("x") as stream:
            stream.write(report_markdown(payload))
        rows = strata_rows(payload)
        with (directory / "STRATA.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
