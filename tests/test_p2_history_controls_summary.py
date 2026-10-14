"""Synthetic artifact tests: no model imports or training required."""

import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.osram_history_training_controls_20261014 import summarize as s


def write_json(path, value):
    path.write_text(json.dumps(value))


def examples(fold):
    # The tiny first fold is wrong; the other folds contain ten correct rows.
    # Pooled ACC=40/41, very different from the mean fold ACC=4/5.
    return (np.zeros(1, dtype=int), np.ones(1, dtype=int)) if fold == 1 else (
        np.zeros(10, dtype=int), np.zeros(10, dtype=int))


def make_control(root, dataset="IEMOCAPFour", control="local-only", seed=66, fold=1):
    task = dict(dataset=dataset, control=control, seed=seed, fold=fold)
    directory = root / dataset / control / f"seed_{seed}" / f"fold_{fold}"
    directory.mkdir(parents=True)
    config = vars(s.build_config(**task, config_class=SimpleNamespace))
    config["train_missing_rates"] = [index / 10 for index in range(8)]
    source_hashes = {name: "a" * 64 for name in s.PROVENANCE_SOURCE_FILES}
    provenance = dict(status="complete", task=task, config=config, commit="a" * 40,
                      source_sha256=source_hashes, dataset_sha256="b" * 64,
                      source_identity_sha256=hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest(),
                      selection_metric="weighted_f1")
    labels, predictions = examples(fold)
    actual = s.classification_metrics(labels, predictions, s.classes_for(dataset))
    metrics = dict(selection_split="test", selection_protocol="per-rate-test-oracle", selection_metric="weighted_f1",
                   missing_mask_protocol=config["missing_mask_protocol"], test={rate: actual for rate in s.RATES},
                   parameter_count=100, trainable_parameter_count=100,
                   selected_epoch_by_rate={rate: 1 for rate in s.RATES},
                   selected_score_by_rate={rate: actual["weighted_f1"] for rate in s.RATES},
                   selected_weighted_f1_by_rate={rate: actual["weighted_f1"] for rate in s.RATES},
                   selected_accuracy_by_rate={rate: actual["accuracy"] for rate in s.RATES})
    metrics["mask_sha256"] = {rate: "e" * 64 for rate in s.RATES}
    metrics["test"] = {rate: dict(actual, mask_sha256="e" * 64) for rate in s.RATES}
    write_json(directory / "PROVENANCE.json", provenance)
    write_json(directory / "config.json", config)
    write_json(directory / "metrics.json", metrics)
    write_json(directory / "history.json", [dict(epoch=epoch, selection_metric="weighted_f1",
               test_oracle={rate: actual for rate in s.RATES}) for epoch in range(1, config["epochs"] + 1)])
    for rate in s.RATES:
        suffix = rate.replace(".", "p")
        np.savez(directory / f"predictions_miss_{suffix}.npz", labels=labels, predictions=predictions,
                 availability=np.ones((len(labels), 3), dtype=np.uint8))
        (directory / f"best_miss_{suffix}.pt").write_bytes(b"synthetic checkpoint; aggregator does not deserialize")
    return directory


def make_anchor(root, dataset="IEMOCAPFour", seed=66, fold=1):
    directory = root / dataset / f"seed_{seed}" / f"fold_{fold}"
    directory.mkdir(parents=True)
    config = vars(s.build_config(dataset, "local-only", seed, fold, config_class=SimpleNamespace))
    config["osram_ablation"] = "full"
    config.update({name: False for name in (
        "local_context_residual", "node_interaction_residual", "time_attention", "text_core",
        "osram_base_first_gap", "osram_history_recency", "osram_relation_block", "osram_relation_dual_readout",
        "osram_gap_increment_filter", "osram_decision_correction", "completion_write_to_memory",
        "osram_history_query_adapter", "osram_post_grn", "osram_local_skip_gate", "paired_history_views",
        "osram_memory_only_adapter", "osram_history_input_gate", "osram_local_evidence_gate",
        "osram_local_evidence_gate_gap_only", "osram_hierarchical_evidence_gate", "osram_hierarchical_feature_only",
        "osram_ced_block", "simple_regression_predictor")})
    config.update(osram_adapter_hidden_dim=0, osram_counterfactual_readout="none", osram_readout_candidate="none",
                  core20_method="none", osram_read_ridge=.001, osram_write_ridge=.001,
                  osram_gap_read="residual", gap_residual_strength=1.0, beta_mode="embedded",
                  osram_query_availability=True, osram_predictor_mode="structured", latent_dim=256,
                  readout_type="shared", target_private_rank=0)
    summary = dict(dataset=dataset, seed=seed, fold=fold, status="complete", training=False,
                   protocol="INTERNAL DIAGNOSTIC ONLY; frozen source per-rate Test-oracle checkpoints; descriptive strata",
                   source_config_sha256="c" * 64, source_missing_mask_protocol=config["missing_mask_protocol"], rates={})
    labels, predictions = examples(fold)
    logits = np.zeros((len(labels), s.classes_for(dataset)))
    logits[np.arange(len(labels)), predictions] = 1
    for rate in s.RATES:
        filename = f"utterances_miss_{rate.replace('.', 'p')}.npz"
        np.savez(directory / filename, labels=labels, pred_LBG=predictions, logits_LBG=logits,
                 conversation_id=np.full(len(labels), f"conversation_fold_{fold}"), position=np.arange(len(labels)),
                 current_availability=np.ones((len(labels), 3), dtype=np.uint8))
        summary["rates"][rate] = dict(utterance_file=filename, checkpoint_sha256="d" * 64, epoch=1,
            metrics={"LBG": s.classification_metrics(labels, predictions, s.classes_for(dataset))},
            checks=dict(LBG_uses_original_logits=True, full_replay_class_mismatches=0, source_prediction_exact=True))
    write_json(directory / "SUMMARY.json", summary)
    write_json(directory / "runtime_config.json", config)
    return directory


def change_json(path, mutate):
    value = s.read_json(path)
    mutate(value)
    write_json(path, value)


def test_complete_66_tasks_and_33_anchors(tmp_path):
    controls, anchors = tmp_path / "controls", tmp_path / "anchors"
    for dataset in s.DATASETS:
        for seed in s.SEEDS:
            for fold in (range(1, 6) if dataset != "MELD" else (1,)):
                for control in s.CONTROLS:
                    make_control(controls, dataset, control, seed, fold)
                make_anchor(anchors, dataset, seed, fold)
    result = s.aggregate(controls, full_anchor_root=anchors)
    assert result["status"] == "complete"
    assert result["valid_task_count"] == 99
    row = result["datasets"]["IEMOCAPFour"]["local-only"]
    assert row["rates"]["0.0"]["accuracy"]["mean"] == pytest.approx(40 / 41)
    assert row["rates"]["0.0"]["weighted_f1"]["mean"] == pytest.approx(80 / 81)
    assert row["rates"]["0.0"]["accuracy"]["mean"] != pytest.approx(.8)
    assert row["rates"]["0.0"]["accuracy"]["std"] == 0
    assert row["high_missing_mean"]["accuracy"]["seed_count"] == 3
    assert result["datasets"]["MELD"]["full-chm"]["rates"]["0.0"]["accuracy"]["mean"] == 0
    report = s.report_markdown(result)
    assert "INTERNAL DIAGNOSTIC ONLY / Test-oracle" in report
    assert "MELD: retained per-rate Test-oracle source weights" in report
    # Default rejects one missing control, one missing rate, and one missing anchor.
    victim = controls / "MELD" / "local-only" / "seed_68" / "fold_1"
    (victim / "best_miss_0p7.pt").unlink()
    with pytest.raises(ValueError, match="missing/empty weights"):
        s.aggregate(controls, full_anchor_root=anchors)
    (victim / "best_miss_0p7.pt").write_bytes(b"restored synthetic")
    change_json(victim / "metrics.json", lambda m: m["test"].pop("0.7"))
    with pytest.raises(ValueError, match="missing/unexpected rates"):
        s.aggregate(controls, full_anchor_root=anchors)
    shutil.rmtree(victim)
    with pytest.raises(ValueError, match="missing valid tasks"):
        s.aggregate(controls, full_anchor_root=anchors)
    make_control(controls, "MELD", "local-only", 68, 1)
    shutil.rmtree(anchors / "MELD" / "seed_68" / "fold_1")
    with pytest.raises(ValueError, match="missing valid tasks"):
        s.aggregate(controls, full_anchor_root=anchors)


def test_partial_does_not_average_incomplete_folds(tmp_path):
    make_control(tmp_path, fold=1)
    result = s.aggregate(tmp_path, allow_partial=True)
    assert result["status"] == "PARTIAL DEBUG ONLY"
    assert result["datasets"]["IEMOCAPFour"] == {}
    assert "PARTIAL DEBUG ONLY" in s.report_markdown(result)
    assert result["issues"]


def test_duplicate_task_is_rejected(tmp_path):
    directory = make_control(tmp_path)
    shutil.copytree(directory, tmp_path / "duplicate")
    with pytest.raises(ValueError, match="duplicate task"):
        s.aggregate(tmp_path)


@pytest.mark.parametrize("mutate,pattern", [
    (lambda p: p.update(status="running"), "status is not complete"),
    (lambda p: p["config"].update(checkpoint_selection_metric="accuracy"), "differs from PROVENANCE"),
    (lambda p: p.update(commit="unknown"), "commit is unknown"),
    (lambda p: p["source_sha256"].update({s.PROVENANCE_SOURCE_FILES[0]: "bad"}), "source hashes"),
])
def test_invalid_provenance(tmp_path, mutate, pattern):
    directory = make_control(tmp_path)
    change_json(directory / "PROVENANCE.json", mutate)
    with pytest.raises(ValueError, match=pattern):
        s.aggregate(tmp_path)


def test_effective_protocol_cannot_be_overridden(tmp_path):
    directory = make_control(tmp_path)
    change_json(directory / "config.json", lambda c: c.update(checkpoint_selection_metric="accuracy"))
    change_json(directory / "PROVENANCE.json", lambda p: p["config"].update(checkpoint_selection_metric="accuracy"))
    with pytest.raises(ValueError, match="config protocol mismatch"):
        s.load_control(directory)


def test_metrics_are_recomputed_from_predictions(tmp_path):
    directory = make_control(tmp_path)
    change_json(directory / "metrics.json", lambda m: m["test"]["0.0"].update(weighted_f1=1.0))
    with pytest.raises(ValueError, match="recomputed weighted_f1 mismatch"):
        s.load_control(directory)
    with pytest.raises(ValueError, match="invalid class"):
        s.classification_metrics([0, 1], [0, np.nan], 4)


def test_oracle_epoch_first_maximum_is_checked(tmp_path):
    directory = make_control(tmp_path)
    change_json(directory / "metrics.json", lambda m: m["selected_epoch_by_rate"].update({"0.0": 2}))
    with pytest.raises(ValueError, match="first weighted-F1 maximum"):
        s.load_control(directory)


def test_source_commit_and_effective_config_consistency(tmp_path):
    make_control(tmp_path, fold=1)
    directory = make_control(tmp_path, fold=2)
    change_json(directory / "PROVENANCE.json", lambda p: p.update(commit="b" * 40))
    change_json(directory / "config.json", lambda c: c.update(projector_dropout=.7))
    change_json(directory / "PROVENANCE.json", lambda p: p["config"].update(projector_dropout=.7))
    with pytest.raises(ValueError, match="inconsistent training commit"):
        s.aggregate(tmp_path)
    result = s.aggregate(tmp_path, allow_partial=True)
    assert any("inconsistent effective configuration" in issue for issue in result["issues"])


def test_anchor_identity_and_frozen_protocol(tmp_path):
    directory = make_anchor(tmp_path)
    change_json(directory / "runtime_config.json", lambda c: c.update(seed=67))
    with pytest.raises(ValueError, match="anchor config identity mismatch"):
        s.load_anchor(directory)
    change_json(directory / "runtime_config.json", lambda c: c.update(seed=66))
    change_json(directory / "SUMMARY.json", lambda c: c.update(training=True))
    with pytest.raises(ValueError, match="complete frozen"):
        s.load_anchor(directory)


def test_anchor_conversation_overlap_rejected(tmp_path):
    anchors = tmp_path / "anchors"
    for fold in range(1, 6):
        directory = make_anchor(anchors, fold=fold)
        for rate in s.RATES:
            path = directory / f"utterances_miss_{rate.replace('.', 'p')}.npz"
            with np.load(path) as loaded:
                values = {name: loaded[name] for name in loaded.files}
            values["conversation_id"] = np.full(len(values["labels"]), "overlapping")
            np.savez(path, **values)
    with pytest.raises(ValueError, match="overlapping anchor conversations"):
        s.aggregate(tmp_path / "controls", full_anchor_root=anchors)


def test_seed_statistics_use_sample_std():
    assert s.stats([.1, .2, .3])["std"] == pytest.approx(.1)
    assert s.stats([.2])["std"] is None


def test_rate_means_are_computed_within_seed(tmp_path):
    directory = make_control(tmp_path, "MELD")
    metrics = s.read_json(directory / "metrics.json")
    history = s.read_json(directory / "history.json")
    for rate in s.HIGH_RATES:
        actual = dict(accuracy=1.0, weighted_f1=1.0, mask_sha256="e" * 64)
        metrics["test"][rate] = actual
        for field in ("selected_score_by_rate", "selected_weighted_f1_by_rate", "selected_accuracy_by_rate"):
            metrics[field][rate] = 1.0
        for row in history:
            row["test_oracle"][rate] = actual
        np.savez(directory / f"predictions_miss_{rate.replace('.', 'p')}.npz",
                 labels=np.zeros(1, dtype=int), predictions=np.zeros(1, dtype=int), availability=np.ones((1, 3), dtype=np.uint8))
    write_json(directory / "metrics.json", metrics)
    write_json(directory / "history.json", history)
    result = s.aggregate(tmp_path, allow_partial=True)["datasets"]["MELD"]["local-only"]
    assert result["eight_rate_mean"]["accuracy"]["mean"] == pytest.approx(3 / 8)
    assert result["high_missing_mean"]["accuracy"]["mean"] == 1.0
    assert result["high_missing_mean"]["accuracy"]["std"] is None


def test_valid_but_different_source_hashes_are_rejected(tmp_path):
    make_control(tmp_path)
    directory = make_control(tmp_path, fold=2)
    provenance = s.read_json(directory / "PROVENANCE.json")
    provenance["source_sha256"][s.PROVENANCE_SOURCE_FILES[0]] = "b" * 64
    provenance["source_identity_sha256"] = hashlib.sha256(
        json.dumps(provenance["source_sha256"], sort_keys=True).encode()).hexdigest()
    write_json(directory / "PROVENANCE.json", provenance)
    with pytest.raises(ValueError, match="inconsistent training commit/source hashes"):
        s.aggregate(tmp_path)


@pytest.mark.parametrize("field,value", [
    ("osram_output_dim", 800), ("osram_num_heads", 4), ("osram_key_dim", 32),
    ("osram_value_dim", 32), ("osram_write_step", .3), ("osram_forward_slot_reuse", True),
    ("osram_emotion_ablation", "local-only"), ("osram_relation_block", True),
    ("osram_meaningful_block", "nested_gnn_rooted_evidence"), ("osram_local_evidence_gate", True),
    ("osram_history_input_gate", True), ("completion_write_to_memory", True),
    ("paired_history_views", True), ("osram_decision_correction", True),
    ("disable_unused_aux_modules", False), ("text_core", True), ("core20_method", "other"),
])
def test_anchor_rejects_changed_cfg84_semantics(tmp_path, field, value):
    directory = make_anchor(tmp_path)
    change_json(directory / "runtime_config.json", lambda c: c.update({field: value}))
    with pytest.raises(ValueError, match="anchor.*config|anchor.*cfg84"):
        s.load_anchor(directory)


def test_anchor_rejects_missing_safety_metadata(tmp_path):
    directory = make_anchor(tmp_path)
    change_json(directory / "runtime_config.json", lambda c: c.pop("osram_relation_block"))
    with pytest.raises(ValueError, match="anchor.*config|anchor.*cfg84"):
        s.load_anchor(directory)


@pytest.mark.parametrize("availability", [np.zeros((1, 3)), np.ones((1, 2)), np.array([[1, 0, 2]])])
def test_control_rejects_invalid_availability(tmp_path, availability):
    directory = make_control(tmp_path)
    np.savez(directory / "predictions_miss_0p5.npz", labels=[0], predictions=[1], availability=availability)
    with pytest.raises(ValueError, match="availability"):
        s.load_control(directory)


def test_control_rejects_mask_hash_metadata_mismatch(tmp_path):
    directory = make_control(tmp_path)
    change_json(directory / "metrics.json", lambda m: m["mask_sha256"].update({"0.5": "f" * 64}))
    with pytest.raises(ValueError, match="mask.*hash|mask_sha256"):
        s.load_control(directory)


def test_control_pair_rejects_changed_valid_mask(tmp_path):
    make_control(tmp_path)
    directory = make_control(tmp_path, control="causal-gru")
    np.savez(directory / "predictions_miss_0p5.npz", labels=[0], predictions=[1], availability=[[1, 0, 0]])
    with pytest.raises(ValueError, match="control.*availability.*mismatch"):
        s.aggregate(tmp_path)


def test_notice_does_not_assign_weighted_f1_selection_to_all_anchors():
    assert "P2 controls use per-rate test weighted-F1" in s.NOTICE
    assert "frozen anchors retain historical per-rate Test-oracle selection" in s.NOTICE


def test_anchor_compatibility_is_limited_to_known_historical_fields(tmp_path):
    directory = make_anchor(tmp_path)
    change_json(directory / "runtime_config.json", lambda c: c.pop("checkpoint_selection_metric"))
    change_json(directory / "runtime_config.json", lambda c: c.pop("osram_causal_gru_history"))
    assert s.load_anchor(directory)["task"]["control"] == "full-chm"
    change_json(directory / "runtime_config.json", lambda c: c.pop("osram_emotion_ablation"))
    with pytest.raises(ValueError, match="anchor cfg84"):
        s.load_anchor(directory)


def test_control_pair_rejects_consistently_tampered_hash_metadata(tmp_path):
    make_control(tmp_path)
    directory = make_control(tmp_path, control="causal-gru")
    metrics = s.read_json(directory / "metrics.json")
    metrics["mask_sha256"]["0.5"] = "f" * 64
    metrics["test"]["0.5"]["mask_sha256"] = "f" * 64
    write_json(directory / "metrics.json", metrics)
    with pytest.raises(ValueError, match="control pair reported mask hashes mismatch"):
        s.aggregate(tmp_path)


def test_anchor_mask_comparison_requires_real_control_ids(tmp_path):
    controls, anchors = tmp_path / "controls", tmp_path / "anchors"
    directory = make_control(controls)
    make_anchor(anchors)
    np.savez(directory / "predictions_miss_0p5.npz", labels=[0], predictions=[1], availability=[[1, 0, 0]])
    # No IDs: no claim that repeated class labels establish sample identity.
    summary = s.aggregate(controls, full_anchor_root=anchors, allow_partial=True)
    assert not any("aligned labels/availability mismatch" in issue for issue in summary["issues"])
    for rate in s.RATES:
        path = directory / f"predictions_miss_{rate.replace('.', 'p')}.npz"
        with np.load(path) as artifact:
            values = {name: artifact[name] for name in artifact.files}
        np.savez(path, **values, conversation_id=["conversation_fold_1"], position=[0])
    with pytest.raises(ValueError, match="control/anchor aligned labels/availability mismatch"):
        s.aggregate(controls, full_anchor_root=anchors)


def test_saved_batch_major_hash_is_not_misrepresented_as_metrics_hash(tmp_path):
    directory = make_control(tmp_path)
    run = s.load_control(directory)
    assert run["artifact_sha256"]["0.0"]["reported_time_major_mask"] == "e" * 64
    assert run["artifact_sha256"]["0.0"]["saved_batch_major_availability_bytes"] != "e" * 64
    summary = s.aggregate(tmp_path, allow_partial=True)
    assert "cannot be recomputed" in summary["mask_verification"]
