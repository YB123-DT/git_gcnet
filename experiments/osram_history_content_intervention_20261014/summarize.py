"""Pool P1 folds within seed, then summarize seeds separately at each rate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments.osram_history_effect_stratification_20261013.evaluate import _sha256, _write
from experiments.osram_history_content_intervention_20261014.intervention import (
    ARMS, MODALITIES, ROW_FIELDS, paired_effects, select_replacements, summarize_rows,
)


def load_checked(directory, rate):
    with np.load(directory/rate["intervention_file"], allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    if set(arrays) != set(ROW_FIELDS):
        raise ValueError("unexpected P1 artifact schema")
    count = len(arrays["label"])
    if any(len(value) != count for value in arrays.values()):
        raise ValueError("P1 artifact row lengths differ")
    if count:
        effects = paired_effects(arrays["label"], arrays["logits_original"], arrays["logits_replaced"])
        for name, expected in effects.items():
            np.testing.assert_array_equal(arrays[name], expected)
    rows = [{name: arrays[name][index] for name in arrays} for index in range(count)]
    skipped = json.loads((directory/rate["skipped_file"]).read_text())
    with np.load(directory/rate["original_file"], allow_pickle=False) as loaded:
        original = {name: loaded[name] for name in loaded.files}
    target_keys = list(zip(original["conversation_id"].astype(str), original["position"].tolist()))
    if len(set(target_keys)) != len(target_keys):
        raise ValueError("duplicate original target")
    originals = dict(zip(target_keys, range(len(target_keys))))
    conversations = {}
    for conversation in sorted(set(original["conversation_id"])):
        selected = np.flatnonzero(original["conversation_id"] == conversation)
        selected = selected[np.argsort(original["position"][selected])]
        np.testing.assert_array_equal(original["position"][selected], np.arange(len(selected)))
        conversations[str(conversation)] = (original["current_availability"][selected], original["speaker"][selected])
    expected_plans, expected_skips = {}, {}
    for (conversation, target) in target_keys:
        availability, speakers = conversations[conversation]
        selected, absent = select_replacements(availability, speakers, target)
        for plan in selected:
            expected_plans[conversation, target, plan["modality"], plan["arm"]] = plan
        for item in absent:
            expected_skips[conversation, target, item["modality"], item["arm"]] = item["reason"]
    seen = set()
    for row in rows:
        key = (str(row["conversation_id"]), int(row["position"]), str(row["modality"]), str(row["arm"]))
        if key in seen or key not in expected_plans:
            raise ValueError("duplicate or unexpected intervention row")
        seen.add(key)
        expected = expected_plans[key]
        for name in ("source_position", "donor_position", "source_lag", "donor_lag", "lag_mismatch",
                     "target_speaker", "source_speaker", "donor_speaker", "source_same_speaker", "donor_same_speaker"):
            if row[name] != expected[name]:
                raise ValueError(f"selection metadata mismatch: {name}")
        index = originals[key[:2]]
        np.testing.assert_array_equal(row["logits_original"], original["logits_LBG"][index])
        if row["label"] != original["labels"][index]:
            raise ValueError("original label mismatch")
        if not all(row[name] for name in ("current_local_within_tolerance", "current_availability_exact", "availability_trajectory_exact", "current_input_exact")):
            raise ValueError("failed P1 invariance check")
    if seen != set(expected_plans):
        raise ValueError("missing intervention rows")
    seen_skips = set()
    for row in skipped:
        key = (row["conversation_id"], row["position"], row["modality"], row["arm"])
        if key in seen_skips or key not in expected_skips or row["reason"] != expected_skips[key]:
            raise ValueError("invalid skip coverage")
        seen_skips.add(key)
    if seen_skips != set(expected_skips):
        raise ValueError("missing skipped targets")
    return rows, skipped, target_keys


def aggregate(inputs, *, allow_partial=False):
    runs, seen = [], set()
    for directory in inputs:
        summary = json.loads((directory/"SUMMARY.json").read_text())
        if summary.get("status") != "complete" or summary.get("training") is not False or summary.get("diagnostic") != "P1_history_content" or not summary.get("rates"):
            raise ValueError("not a complete frozen P1 summary")
        identity = (summary["dataset"], summary["seed"], summary["fold"])
        if identity in seen:
            raise ValueError("duplicate dataset/seed/fold")
        if identity[0] not in {"IEMOCAPFour", "IEMOCAPSix", "MELD"}:
            raise ValueError("unsupported dataset")
        seen.add(identity)
        runs.append((directory, summary))
    if not runs:
        raise ValueError("no P1 runs")
    datasets = {}
    for dataset in sorted({summary["dataset"] for _, summary in runs}):
        selected = [(directory, summary) for directory, summary in runs if summary["dataset"] == dataset]
        expected_folds = {1} if dataset == "MELD" else set(range(1, 6))
        seeds = sorted({summary["seed"] for _, summary in selected})
        payload = dict(seeds=seeds, expected_folds=sorted(expected_folds), rates={})
        for rate_name in sorted({name for _, summary in selected for name in summary["rates"]}, key=float):
            by_seed = {}
            for seed in seeds:
                seed_runs = [(directory, summary) for directory, summary in selected if summary["seed"] == seed and rate_name in summary["rates"]]
                folds = {summary["fold"] for _, summary in seed_runs}
                if not folds <= expected_folds or (not allow_partial and folds != expected_folds):
                    raise ValueError(f"{dataset} seed={seed} rate={rate_name}: folds={sorted(folds)}, expected={sorted(expected_folds)}")
                if not seed_runs:
                    continue
                rows, skipped, target_keys, sources = [], [], [], []
                for directory, summary in sorted(seed_runs, key=lambda item: item[1]["fold"]):
                    rate = summary["rates"][rate_name]
                    loaded, absent, targets = load_checked(directory, rate)
                    rows.extend(loaded)
                    skipped.extend(absent)
                    target_keys.extend(targets)
                    sources.append(dict(fold=summary["fold"], summary=str(directory/"SUMMARY.json"),
                        checkpoint_sha256=rate["checkpoint_sha256"],
                        artifact_sha256={name: _sha256(directory/rate[name]) for name in ("intervention_file", "original_file", "skipped_file")}))
                if len(set(target_keys)) != len(target_keys):
                    raise ValueError("overlapping conversations/targets across folds")
                by_seed[str(seed)] = dict(folds=sorted(folds), targets=len(target_keys), sources=sources, modalities=summarize_rows(rows, skipped))
            across = {}
            for modality in MODALITIES:
                across[modality] = {}
                for arm in ARMS:
                    item = {}
                    for metric in ("coverage", "mean_delta_ce", "mean_true_probability_delta", "mean_correction", "mean_harm", "mean_lag_mismatch", "mean_current_local_exact"):
                        values = {seed: row["modalities"][modality][arm][metric] for seed, row in by_seed.items() if row["modalities"][modality][arm][metric] is not None}
                        data = list(values.values())
                        item[metric] = dict(mean=float(np.mean(data)) if data else None,
                            std=float(np.std(data, ddof=1)) if len(data) > 1 else None, seed_count=len(data), seed_values=values)
                    across[modality][arm] = item
            payload["rates"][rate_name] = dict(by_seed=by_seed, across_seed=across)
        datasets[dataset] = payload
    return dict(status="complete", training=False, diagnostic="P1_history_content", allow_partial=allow_partial,
                protocol="INTERNAL DIAGNOSTIC ONLY; folds pooled within seed; datasets and missing rates kept separate",
                uncertainty="Across-seed sample std; correlated utterance diagnostics; no significance or generalization claim",
                delta_definition="CE(replaced)-CE(original); positive means harmful; probability replaced-original", datasets=datasets)


def report_markdown(summary):
    lines = ["# P1 matched history content", "", summary["protocol"], "", summary["delta_definition"], "",
             "| Dataset | Rate | Seed | Modality | Arm | Eligible / targets | Delta CE | Probability delta | Correction / harm |",
             "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for dataset, payload in summary["datasets"].items():
        for rate, rate_result in payload["rates"].items():
            for seed, seed_result in rate_result["by_seed"].items():
                for modality, arms in seed_result["modalities"].items():
                    for arm, row in arms.items():
                        ce = "NA" if row["mean_delta_ce"] is None else f"{row['mean_delta_ce']:.6f}"
                        probability = "NA" if row["mean_true_probability_delta"] is None else f"{row['mean_true_probability_delta']:.6f}"
                        lines.append(f"| {dataset} | {rate} | {seed} | {modality} | {arm} | {row['count']} / {row['targets']} | {ce} | {probability} | {row['corrections']} / {row['harms']} |")
    return "\n".join(lines)+"\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    summary = aggregate([path.parent for path in sorted(args.input_root.rglob("SUMMARY.json"))], allow_partial=args.allow_partial)
    args.output.mkdir(parents=True)
    for directory, payload in [(args.output, summary)]+[(args.output/dataset, dict(summary, datasets={dataset: result})) for dataset, result in summary["datasets"].items()]:
        directory.mkdir(exist_ok=directory == args.output)
        _write(directory/"AGGREGATE.json", payload)
        (directory/"REPORT.md").write_text(report_markdown(payload))
        rows = [dict(dataset=dataset, rate=rate, seed=seed, modality=modality, arm=arm,
                     count=item["count"], targets=item["targets"], coverage=item["coverage"],
                     delta_ce=item["mean_delta_ce"], probability_delta=item["mean_true_probability_delta"],
                     corrections=item["corrections"], harms=item["harms"])
                for dataset, result in payload["datasets"].items() for rate, rate_result in result["rates"].items()
                for seed, seed_result in rate_result["by_seed"].items() for modality, arms in seed_result["modalities"].items()
                for arm, item in arms.items()]
        with (directory/"SUMMARY.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
