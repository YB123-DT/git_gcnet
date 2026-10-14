"""Strict, prediction-grounded P2 aggregation; never launches training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from types import SimpleNamespace
from zipfile import BadZipFile

import numpy as np

from .run import CONTROLS, DATASETS, PROVENANCE_SOURCE_FILES, build_config, tasks

SEEDS = (66, 67, 68)
RATES = tuple(f"{index / 10:.1f}" for index in range(8))
HIGH_RATES = RATES[5:]
NOTICE = ("INTERNAL DIAGNOSTIC ONLY / Test-oracle: P2 controls use per-rate test weighted-F1; "
          "frozen anchors retain historical per-rate Test-oracle selection; no independent-test or significance claim.")
ANCHOR_CFG84_SAFETY = {
    **dict.fromkeys((
        "local_context_residual", "node_interaction_residual", "time_attention", "text_core",
        "osram_base_first_gap", "osram_history_recency", "osram_relation_block", "osram_relation_dual_readout",
        "osram_gap_increment_filter", "osram_decision_correction", "completion_write_to_memory",
        "osram_history_query_adapter", "osram_post_grn", "osram_local_skip_gate", "paired_history_views",
        "osram_memory_only_adapter", "osram_history_input_gate", "osram_local_evidence_gate",
        "osram_local_evidence_gate_gap_only", "osram_hierarchical_evidence_gate", "osram_hierarchical_feature_only",
        "osram_ced_block", "simple_regression_predictor"), False),
    "osram_adapter_hidden_dim": 0, "osram_counterfactual_readout": "none", "osram_readout_candidate": "none",
    "core20_method": "none", "osram_read_ridge": .001, "osram_write_ridge": .001,
    "osram_gap_read": "residual", "gap_residual_strength": 1.0, "beta_mode": "embedded",
    "osram_query_availability": True, "osram_predictor_mode": "structured", "latent_dim": 256,
    "readout_type": "shared", "target_private_rank": 0,
}


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def classification_metrics(labels, predictions, classes):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    require(labels.ndim == predictions.ndim == 1 and labels.shape == predictions.shape and labels.size,
            "labels/predictions must be matching nonempty vectors")
    for values in (labels, predictions):
        require(np.issubdtype(values.dtype, np.number) and np.isfinite(values).all()
                and np.equal(values, np.floor(values)).all()
                and ((values >= 0) & (values < classes)).all(), "invalid class labels/predictions")
    weighted = 0.0
    for label in range(classes):
        actual, predicted = labels == label, predictions == label
        support, selected = int(actual.sum()), int(predicted.sum())
        if support:
            weighted += support * 2 * int((actual & predicted).sum()) / (support + selected)
    return {"accuracy": float(np.mean(labels == predictions)), "weighted_f1": weighted / labels.size}


def check_metrics(actual, saved, context):
    for name, value in actual.items():
        observed = saved.get(name)
        require(isinstance(observed, (int, float)) and math.isfinite(observed)
                and math.isclose(value, observed, abs_tol=1e-7, rel_tol=1e-7),
                f"{context}: recomputed {name} mismatch ({value} vs {observed})")


def classes_for(dataset):
    return {"IEMOCAPFour": 4, "IEMOCAPSix": 6, "MELD": 7}[dataset]


def identity(task):
    return tuple(task[name] for name in ("dataset", "control", "seed", "fold"))


def digest_string(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_availability(values, count, rate):
    require(values.shape == (count, 3) and np.isin(values, [0, 1]).all()
            and (values.sum(axis=1) >= 1).all(), "invalid availability shape/binary/nonempty observed set")
    if rate == "0.0":
        require((values == 1).all(), "rate 0.0 availability must be fully observed")


def optional_identities(artifact, count):
    present = {name for name in ("conversation_id", "position") if name in artifact}
    if not present:
        return None
    require(len(present) == 2, "incomplete conversation_id/position metadata")
    conversations, positions = artifact["conversation_id"].copy(), artifact["position"].copy()
    require(conversations.shape == positions.shape == (count,)
            and np.issubdtype(positions.dtype, np.integer) and (positions >= 0).all(), "invalid utterance identities")
    keys = list(zip(conversations.tolist(), positions.tolist()))
    require(len(set(keys)) == count, "duplicate utterance identities")
    return keys


def expected_tasks(controls=CONTROLS):
    return {identity(task) for dataset in DATASETS for task in tasks(dataset, list(SEEDS), controls)}


def validate_config(config, task):
    expected = vars(build_config(**task, config_class=SimpleNamespace))
    for name, value in expected.items():
        require(config.get(name) == value, f"config protocol mismatch: {name}")
    require(config.get("train_missing_rates") == [index / 10 for index in range(8)],
            "config cyclic missing rates differ from 0.0--0.7")
    for name in ("initial_backbone_checkpoint", "pretrained_learning_rate", "b2_base_checkpoint",
                 "b2_pretrain_checkpoint", "joint_pretrain_checkpoint", "teacher_checkpoint",
                 "text_subspace_checkpoint"):
        require(config.get(name) is None, f"unexpected pretraining: {name}")
    require(config.get("teacher_mode") != "pretrained-frozen", "unexpected frozen teacher")


def load_control(directory):
    provenance = read_json(directory / "PROVENANCE.json")
    task = provenance["task"]
    require(identity(task) in expected_tasks(), f"unexpected task identity: {task}")
    require(provenance.get("status") == "complete", "task status is not complete")
    config = read_json(directory / "config.json")
    require(config == provenance.get("config"), "config.json differs from PROVENANCE config")
    validate_config(config, task)
    source_hashes = provenance.get("source_sha256", {})
    require(set(PROVENANCE_SOURCE_FILES) <= set(source_hashes)
            and all(digest_string(v) for v in source_hashes.values()), "missing/invalid source hashes")
    code_identity = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()
    require(code_identity == provenance.get("source_identity_sha256"), "source identity hash mismatch")
    require(provenance.get("commit") not in (None, "", "unknown"), "source commit is unknown")
    require(digest_string(provenance.get("dataset_sha256")), "missing dataset hash")
    metrics = read_json(directory / "metrics.json")
    for name in ("parameter_count", "trainable_parameter_count", "task_connected_parameter_count"):
        if name in metrics:
            require(type(metrics[name]) is int and metrics[name] >= 0, f"invalid {name}")
    for name, value in (("selection_split", "test"), ("selection_protocol", "per-rate-test-oracle"),
                        ("selection_metric", "weighted_f1"),
                        ("missing_mask_protocol", config["missing_mask_protocol"])):
        require(metrics.get(name) == value, f"metrics protocol mismatch: {name}")
    require(provenance.get("selection_metric") == "weighted_f1", "provenance selection metric mismatch")
    for name in ("test", "selected_epoch_by_rate", "selected_score_by_rate",
                 "selected_weighted_f1_by_rate", "selected_accuracy_by_rate", "mask_sha256"):
        require(set(metrics.get(name, {})) == set(RATES), f"missing/unexpected rates: {name}")
    history = read_json(directory / "history.json")
    require(isinstance(history, list) and len(history) == config["epochs"]
            and [row.get("epoch") for row in history] == list(range(1, config["epochs"] + 1)),
            "incomplete/duplicate training history epochs")
    arrays, hashes, availabilities, utterance_ids = {}, {}, {}, {}
    labels_reference = None
    identities_reference = None
    for rate in RATES:
        suffix = rate.replace(".", "p")
        checkpoint = directory / f"best_miss_{suffix}.pt"
        require(checkpoint.is_file() and checkpoint.stat().st_size > 0, f"missing/empty weights: {rate}")
        path = directory / f"predictions_miss_{suffix}.npz"
        with np.load(path, allow_pickle=False) as artifact:
            labels, predictions = artifact["labels"].copy(), artifact["predictions"].copy()
            availability = artifact["availability"].copy()
            ids = optional_identities(artifact, len(labels))
        validate_availability(availability, len(labels), rate)
        mask_hash = metrics["mask_sha256"][rate]
        require(digest_string(mask_hash) and metrics["test"][rate].get("mask_sha256") == mask_hash,
                f"rate {rate}: mask_sha256 metadata mismatch")
        actual = classification_metrics(labels, predictions, classes_for(task["dataset"]))
        check_metrics(actual, metrics["test"][rate], f"rate {rate}")
        if labels_reference is not None:
            require(np.array_equal(labels, labels_reference), "test label/order changes across rates")
            require(ids == identities_reference, "test identities/order change across rates")
        labels_reference = labels
        identities_reference = ids
        scores = []
        for row in history:
            require(row.get("selection_metric") == "weighted_f1"
                    and set(row.get("test_oracle", {})) == set(RATES), "history protocol/rate mismatch")
            score = row["test_oracle"][rate]["weighted_f1"]
            require(isinstance(score, (int, float)) and math.isfinite(score) and 0 <= score <= 1,
                    "invalid history weighted_f1")
            scores.append(score)
        epoch = metrics["selected_epoch_by_rate"][rate]
        require(type(epoch) is int and epoch == scores.index(max(scores)) + 1,
                f"rate {rate}: selected epoch does not match first weighted-F1 maximum")
        check_metrics(actual, {"accuracy": metrics["selected_accuracy_by_rate"][rate],
                              "weighted_f1": metrics["selected_weighted_f1_by_rate"][rate]}, rate)
        require(math.isclose(actual["weighted_f1"], metrics["selected_score_by_rate"][rate], abs_tol=1e-7)
                and math.isclose(actual["weighted_f1"], max(scores), abs_tol=1e-7),
                f"rate {rate}: oracle selection score mismatch")
        arrays[rate] = (labels, predictions)
        availabilities[rate], utterance_ids[rate] = availability, ids
        hashes[rate] = {"predictions": sha256(path), "weights": sha256(checkpoint),
                        "saved_batch_major_availability_bytes": hashlib.sha256(np.ascontiguousarray(availability).tobytes()).hexdigest(),
                        "reported_time_major_mask": mask_hash}
    metadata_hashes = {name: sha256(directory / name)
                       for name in ("PROVENANCE.json", "config.json", "metrics.json", "history.json")}
    return {"task": task, "config": config, "provenance": provenance, "metrics": metrics,
            "metadata_sha256": metadata_hashes,
            "availability": availabilities, "utterance_ids": utterance_ids,
            "arrays": arrays, "path": str(directory), "artifact_sha256": hashes}


def load_anchor(directory):
    summary = read_json(directory / "SUMMARY.json")
    task = {name: summary[name] for name in ("dataset", "seed", "fold")}
    task["control"] = "full-chm"
    require((task["dataset"] in DATASETS and task["seed"] in SEEDS
             and task["fold"] in (range(1, 6) if task["dataset"] != "MELD" else (1,))),
            "unexpected anchor dataset/seed/fold")
    require(summary.get("status") == "complete" and summary.get("training") is False
            and "frozen source per-rate Test-oracle checkpoints" in summary.get("protocol", ""),
            "anchor is not a complete frozen Test-oracle P0 source")
    require(set(summary.get("rates", {})) == set(RATES), "anchor missing/unexpected rates")
    require(digest_string(summary.get("source_config_sha256")), "anchor missing source config hash")
    config = read_json(directory / summary.get("runtime_config_file", "runtime_config.json"))
    for name in ("dataset", "seed", "fold"):
        require(config.get(name) == task[name], f"anchor config identity mismatch: {name}")
    full_expected = vars(build_config(task["dataset"], "local-only", task["seed"], task["fold"], config_class=SimpleNamespace))
    full_expected.update(osram_ablation="full", osram_causal_gru_history=False)
    for name, value in {**full_expected, **ANCHOR_CFG84_SAFETY}.items():
        # Historical source selection metadata is deliberately not upgraded;
        # MELD's constructor-only mask adaptation is checked separately below.
        if name in {"checkpoint_selection_metric", "missing_mask_protocol"}:
            continue
        if name == "osram_causal_gru_history" and name not in config:
            continue  # This switch was introduced after the retained P0 source.
        require(name in config and config[name] == value, f"anchor cfg84 config mismatch/missing: {name}")
    for name in ("initial_backbone_checkpoint", "pretrained_learning_rate", "b2_base_checkpoint",
                 "b2_pretrain_checkpoint", "joint_pretrain_checkpoint", "teacher_checkpoint", "text_subspace_checkpoint"):
        require(config.get(name) is None, f"anchor cfg84 unexpected pretraining: {name}")
    require(config.get("teacher_mode") != "pretrained-frozen", "anchor cfg84 unexpected frozen teacher")
    mask = "gcnet-legacy-fixed-v1" if task["dataset"] == "MELD" else "conversation-mask-v2"
    require(summary.get("source_missing_mask_protocol") == mask, "anchor mask protocol mismatch")
    require(config.get("missing_mask_protocol") in ({mask, "conversation-mask-v2"} if task["dataset"] == "MELD" else {mask}),
            "anchor runtime mask adaptation mismatch")
    arrays, metadata, hashes, availabilities, utterance_ids = {}, {}, {}, {}, {}
    baseline = None
    for rate in RATES:
        row = summary["rates"][rate]
        require(digest_string(row.get("checkpoint_sha256")), "anchor checkpoint hash missing")
        require(type(row.get("epoch")) is int and 1 <= row["epoch"] <= config["epochs"],
                "anchor retained epoch missing/invalid")
        checks = row.get("checks", {})
        require(checks.get("LBG_uses_original_logits") is True
                and checks.get("full_replay_class_mismatches") == 0,
                "anchor LBG original replay check failed")
        if task["dataset"] != "MELD":
            require(checks.get("source_prediction_exact") is True, "anchor source prediction check missing")
        path = directory / row["utterance_file"]
        require(path.resolve().parent == directory.resolve(), "anchor artifact escapes source directory")
        with np.load(path, allow_pickle=False) as artifact:
            labels, predictions = artifact["labels"].copy(), artifact["pred_LBG"].copy()
            logits = artifact["logits_LBG"]
            require(logits.shape == (len(labels), classes_for(task["dataset"]))
                    and np.isfinite(logits).all() and np.array_equal(logits.argmax(1), predictions),
                    "anchor LBG logits/predictions mismatch")
            conversations, positions = artifact["conversation_id"].copy(), artifact["position"].copy()
            availability = artifact["current_availability"].copy()
            ids = optional_identities(artifact, len(labels))
        validate_availability(availability, len(labels), rate)
        require(conversations.shape == positions.shape == labels.shape
                and len(set(zip(conversations.tolist(), positions.tolist()))) == len(labels),
                "anchor duplicate/invalid utterance identities")
        require(np.issubdtype(positions.dtype, np.integer) and (positions >= 0).all(), "anchor invalid positions")
        current = (labels, conversations, positions)
        if baseline is not None:
            require(all(np.array_equal(a, b) for a, b in zip(current, baseline)),
                    "anchor test identities/labels change across rates")
        baseline = current
        check_metrics(classification_metrics(labels, predictions, classes_for(task["dataset"])),
                      row["metrics"]["LBG"], f"anchor rate {rate}")
        arrays[rate], metadata[rate] = (labels, predictions), (conversations, positions)
        availabilities[rate], utterance_ids[rate] = availability, ids
        hashes[rate] = {"utterances": sha256(path), "source_weights": row["checkpoint_sha256"]}
    return {"task": task, "config": config, "arrays": arrays, "metadata": metadata,
            "metadata_sha256": {"SUMMARY.json": sha256(directory / "SUMMARY.json"),
                                "runtime_config.json": sha256(directory / summary.get("runtime_config_file", "runtime_config.json"))},
            "availability": availabilities, "utterance_ids": utterance_ids,
            "path": str(directory), "artifact_sha256": hashes, "anchor_summary": summary}


def stats(values):
    return {"mean": statistics.mean(values), "std": statistics.stdev(values) if len(values) > 1 else None,
            "seed_count": len(values), "seed_values": values}


def aggregate(input_root, *, full_anchor_root=None, allow_partial=False):
    runs, issues, seen = [], [], set()
    sources = [(Path(input_root), "PROVENANCE.json", load_control)]
    if full_anchor_root is not None:
        sources.append((Path(full_anchor_root), "SUMMARY.json", load_anchor))
    for root, filename, loader in sources:
        for path in sorted(root.rglob(filename)):
            try:
                run = loader(path.parent)
                key = identity(run["task"])
                require(key not in seen, f"duplicate task: {key}")
                seen.add(key)
                runs.append(run)
            except (ValueError, KeyError, OSError, TypeError, BadZipFile, EOFError) as error:
                issues.append(f"{path}: {error}")
    expected = expected_tasks()
    if full_anchor_root is not None:
        expected |= {(d, "full-chm", s, f) for d, _, s, f in expected_tasks(("local-only",))}
    missing = sorted(expected - seen)
    if missing:
        issues.append(f"missing valid tasks ({len(missing)}): {missing}")
    # All controls must share one immutable training source and one effective
    # config, apart from documented dataset/control/seed/fold differences.
    controls = [run for run in runs if run["task"]["control"] in CONTROLS]
    if controls:
        reference = controls[0]
        varying = {"dataset", "seed", "fold", "epochs", "missing_mask_protocol", "osram_ablation", "osram_causal_gru_history"}
        normalized = lambda run: {k: v for k, v in run["config"].items() if k not in varying}
        for run in controls:
            a, b = run["provenance"], reference["provenance"]
            if any(a.get(k) != b.get(k) for k in ("commit", "source_sha256", "code_diff_sha256")):
                issues.append(f"{run['path']}: inconsistent training commit/source hashes/diff")
            if normalized(run) != normalized(reference):
                issues.append(f"{run['path']}: inconsistent effective configuration")
        for dataset in DATASETS:
            selected = [run for run in controls if run["task"]["dataset"] == dataset]
            if selected and any((run["provenance"]["dataset_sha256"], run["provenance"].get("feature_manifest"))
                                != (selected[0]["provenance"]["dataset_sha256"], selected[0]["provenance"].get("feature_manifest"))
                                for run in selected):
                issues.append(f"{dataset}: inconsistent dataset/feature provenance")
    datasets = {}
    all_controls = (*CONTROLS, "full-chm") if full_anchor_root is not None else CONTROLS
    for dataset in DATASETS:
        dataset_output = {}
        for control in all_controls:
            by_seed = {}
            for seed in SEEDS:
                selected = sorted([run for run in runs if identity(run["task"])[:3] == (dataset, control, seed)],
                                  key=lambda run: run["task"]["fold"])
                folds = list(range(1, 6)) if dataset != "MELD" else [1]
                if [run["task"]["fold"] for run in selected] != folds:
                    continue  # Partial fold coverage never becomes a seed estimate.
                rates = {}
                overlapping = False
                for rate in RATES:
                    if control == "full-chm":
                        conversation_sets = [set(run["metadata"][rate][0].tolist()) for run in selected]
                        combined = set()
                        for conversations in conversation_sets:
                            if combined & conversations:
                                issues.append(f"{dataset} seed={seed} rate={rate}: overlapping anchor conversations")
                                overlapping = True
                            combined |= conversations
                    labels = np.concatenate([run["arrays"][rate][0] for run in selected])
                    predictions = np.concatenate([run["arrays"][rate][1] for run in selected])
                    rates[rate] = {**classification_metrics(labels, predictions, classes_for(dataset)), "count": len(labels)}
                if overlapping:
                    continue
                by_seed[str(seed)] = {"folds": folds, "rates": rates,
                                     "eight_rate_mean": {k: statistics.mean(rates[r][k] for r in RATES) for k in ("accuracy", "weighted_f1")},
                                     "high_missing_mean": {k: statistics.mean(rates[r][k] for r in HIGH_RATES) for k in ("accuracy", "weighted_f1")}}
            if not by_seed:
                continue
            output = {"by_seed": by_seed, "rates": {rate: {k: stats([row["rates"][rate][k] for row in by_seed.values()])
                                                                      for k in ("accuracy", "weighted_f1")} for rate in RATES}}
            for group in ("eight_rate_mean", "high_missing_mean"):
                output[group] = {k: stats([row[group][k] for row in by_seed.values()]) for k in ("accuracy", "weighted_f1")}
            selected = [run for run in runs if run["task"]["dataset"] == dataset and run["task"]["control"] == control]
            output["parameters"] = {name: sorted({run.get("metrics", {}).get(name) for run in selected
                                                  if run.get("metrics", {}).get(name) is not None})
                                    for name in ("parameter_count", "trainable_parameter_count", "task_connected_parameter_count")}
            for name, counts in output["parameters"].items():
                if len(counts) > 1:
                    issues.append(f"{dataset}/{control}: inconsistent {name}")
            dataset_output[control] = output
        datasets[dataset] = dataset_output
    # Metadata identities alone do not establish matching test samples.
    for dataset, _, seed, fold in expected_tasks(("local-only",)):
        comparable = [run for run in runs if (run["task"]["dataset"], run["task"]["seed"], run["task"]["fold"])
                      == (dataset, seed, fold)]
        if comparable:
            for run in comparable[1:]:
                if any(not np.array_equal(run["arrays"][rate][0], comparable[0]["arrays"][rate][0]) for rate in RATES):
                    issues.append(f"{dataset} seed={seed} fold={fold}: control/anchor test labels/order mismatch")
            control_runs = [run for run in comparable if run["task"]["control"] in CONTROLS]
            for run in control_runs[1:]:
                if any(not np.array_equal(run["availability"][rate], control_runs[0]["availability"][rate]) for rate in RATES):
                    issues.append(f"{dataset} seed={seed} fold={fold}: control pair availability mismatch")
                if run["metrics"]["mask_sha256"] != control_runs[0]["metrics"]["mask_sha256"]:
                    issues.append(f"{dataset} seed={seed} fold={fold}: control pair reported mask hashes mismatch")
            anchors = [run for run in comparable if run["task"]["control"] == "full-chm"]
            for run in control_runs:
                if anchors:
                    anchor = anchors[0]
                    for rate in RATES:
                        ids = run["utterance_ids"][rate]
                        if ids is None:
                            continue  # Existing trainer NPZs do not save utterance identities.
                        anchor_ids = anchor["utterance_ids"][rate]
                        lookup = {key: index for index, key in enumerate(anchor_ids)}
                        if set(ids) != set(anchor_ids):
                            issues.append(f"{dataset} seed={seed} fold={fold} rate={rate}: control/anchor utterance identities mismatch")
                            continue
                        aligned = np.asarray([lookup[key] for key in ids])
                        if (not np.array_equal(run["availability"][rate], anchor["availability"][rate][aligned])
                                or not np.array_equal(run["arrays"][rate][0], anchor["arrays"][rate][0][aligned])):
                            issues.append(f"{dataset} seed={seed} fold={fold} rate={rate}: control/anchor aligned labels/availability mismatch")
    if issues and not allow_partial:
        raise ValueError("strict P2 aggregation failed:\n" + "\n".join(issues))
    return {"status": "PARTIAL DEBUG ONLY" if issues or allow_partial else "complete", "notice": NOTICE,
            "allow_partial": allow_partial, "report_scope": "three-model comparison" if full_anchor_root else "controls-only; full-chm anchor absent",
            "uncertainty": "Across-seed sample std (ddof=1); folds pooled before metrics; rate means computed within seed before seed statistics.",
            "anchor_selection_metadata": ("IEMOCAP: historical weighted-F1 provenance; legacy configs may omit the selection-metric field. "
                                          "MELD: retained per-rate Test-oracle source weights; selection-metric metadata unavailable. "
                                          "Full CHM frozen replay preserves historical selection, rather than establishing identical new CLI metadata."
                                          if full_anchor_root is not None else None),
            "mask_verification": ("Saved availability is batch-major, [N,3], binary, with at least one observed modality; "
                                  "rate 0.0 is fully observed. Control pairs must match labels and availability exactly. "
                                  "Reported metrics.mask_sha256 hashes time-major rows per batch; the prediction NPZ lacks "
                                  "conversation IDs, lengths and batch boundaries, so this hash cannot be recomputed from "
                                  "its batch-major availability. Reported per-rate hashes are checked for valid format and "
                                  "consistent test/top-level metadata only. P0 availability is compared by conversation_id/position "
                                  "only when control artifacts explicitly contain those IDs; absent IDs are never guessed."),
            "expected_control_tasks": 66, "expected_anchor_sources": 33 if full_anchor_root else 0,
            "valid_task_count": len(runs), "issues": issues, "datasets": datasets,
            "sources": [{"task": run["task"], "path": run["path"], "artifact_sha256": run["artifact_sha256"],
                         "metadata_sha256": run["metadata_sha256"],
                         "utterance_identity_available": all(run["utterance_ids"][rate] is not None for rate in RATES),
                         **({"commit": run["provenance"]["commit"], "source_sha256": run["provenance"]["source_sha256"]}
                            if "provenance" in run else {"source_config_sha256": run["anchor_summary"]["source_config_sha256"]})}
                        for run in runs]}


def report_markdown(summary):
    lines = ["# P2 history training controls", "", summary["notice"], "", f"Status: {summary['status']}",
             f"Scope: {summary['report_scope']}", "", summary["uncertainty"], "",
             "ACC and weighted-F1 are fractions. Parameter counts describe saved artifacts; trainable counts do not establish task-connected counts. Missing task-connected counts are unavailable, never inferred.", ""]
    if summary["allow_partial"] or summary["issues"]:
        lines += ["**PARTIAL DEBUG ONLY — NOT A FINAL EXPERIMENT REPORT**", "", *[f"- {issue}" for issue in summary["issues"]], ""]
    if summary["anchor_selection_metadata"]:
        lines += [summary["anchor_selection_metadata"], ""]
    lines += [summary["mask_verification"], ""]
    def display(value):
        std = f"{value['std']:.6f}" if value["std"] is not None else "NA"
        return f"{value['mean']:.6f} ± {std} (n={value['seed_count']})"
    for dataset, controls in summary["datasets"].items():
        lines += [f"## {dataset}", "", "| Control | Rate / summary | ACC mean ± std | weighted-F1 mean ± std |",
                  "| --- | --- | --- | --- |"]
        for control, row in controls.items():
            for rate, result in [*row["rates"].items(), ("8-rate mean", row["eight_rate_mean"]),
                                 ("high missing 0.5/0.6/0.7 mean", row["high_missing_mean"])]:
                lines.append(f"| {control} | {rate} | {display(result['accuracy'])} | {display(result['weighted_f1'])} |")
            lines += ["", f"{control} saved parameter counts: `{json.dumps(row['parameters'], sort_keys=True)}`", ""]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--full-anchor-root", type=Path, help="33 frozen P0 sources; required for final three-model comparison")
    parser.add_argument("--output", type=Path, required=True, help="new directory; existing paths never overwritten")
    parser.add_argument("--allow-partial", action="store_true", help="DEBUG ONLY: list failures and omit incomplete seed groups")
    args = parser.parse_args()
    require(not args.output.exists(), f"output already exists: {args.output}")
    summary = aggregate(args.input_root, full_anchor_root=args.full_anchor_root, allow_partial=args.allow_partial)
    args.output.mkdir(parents=True)
    (args.output / "AGGREGATE.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (args.output / "REPORT.md").write_text(report_markdown(summary))


if __name__ == "__main__":
    main()
