from __future__ import annotations

from dataclasses import replace

import pytest
import torch

import gcnet_modality_jepa.model as gcnet_graph_model
from gcnet_missing_m3.train_gcnet import TrainConfig
import scripts.run_mosi_latent_oracle as runner
from scripts.run_mosi_latent_oracle import (
    _aggregate_summary,
    find_history_record,
    parse_sample_keys,
    prepare_output_directory,
    sample_order_sha256,
    temporal_relation_order,
    validate_run_config,
)


def _formal_config():
    return TrainConfig(
        dataset="CMUMOSI",
        fold=1,
        fusion_type="slot",
        local_context_residual=False,
        mosi_task_mode="regression",
        evaluation_protocol="official",
        mmoe_variant="dual-gate",
        classification_completion=True,
    )


def test_validate_run_config_accepts_only_the_locked_mosi_contract():
    validate_run_config(_formal_config())

    invalid = {
        "dataset": "IEMOCAPSix",
        "fold": 2,
        "fusion_type": "mean",
        "local_context_residual": True,
        "mosi_task_mode": "binary",
        "evaluation_protocol": "loso",
        "mmoe_variant": "paper-shared",
        "classification_completion": False,
    }
    for field, value in invalid.items():
        with pytest.raises(ValueError, match=field):
            validate_run_config(replace(_formal_config(), **{field: value}))


def test_prepare_output_directory_refuses_existing_artifacts(tmp_path):
    output = tmp_path / "oracle"
    prepare_output_directory(output)
    assert output.is_dir()

    (output / "partial.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="non-empty"):
        prepare_output_directory(output)


def test_find_history_record_checks_epoch_uniqueness_and_saved_score():
    history = [
        {"epoch": 1, "validation_mean_weighted_f1": 0.7},
        {"epoch": 2, "validation_mean_weighted_f1": 0.8},
    ]
    assert find_history_record(history, epoch=2, saved_score=0.8) is history[1]

    with pytest.raises(ValueError, match="saved score"):
        find_history_record(history, epoch=2, saved_score=0.81)
    with pytest.raises(ValueError, match="exactly one"):
        find_history_record(history, epoch=3, saved_score=0.8)
    with pytest.raises(ValueError, match="exactly one"):
        find_history_record(history + [dict(history[1])], epoch=2, saved_score=0.8)


def test_parse_sample_keys_uses_the_final_colon_and_order_hash_is_stable():
    keys = ("video:with:colon:0", "video:with:colon:3", "plain:1")
    conversations, indices = parse_sample_keys(keys)

    assert conversations == ("video:with:colon", "video:with:colon", "plain")
    assert indices == (0, 3, 1)
    assert sample_order_sha256(keys) == sample_order_sha256(tuple(keys))
    assert sample_order_sha256(keys) != sample_order_sha256(tuple(reversed(keys)))

    with pytest.raises(ValueError, match="sample key"):
        parse_sample_keys(("missing-index",))
    with pytest.raises(ValueError, match="utterance index"):
        parse_sample_keys(("video:not-an-int",))


def test_summary_keeps_an_empty_high_missing_group_json_safe():
    metrics = {
        "weighted_f1": 0.7,
        "macro_f1": 0.69,
        "accuracy": 0.71,
        "mae": 0.8,
        "correlation": 0.6,
    }
    record = {
        "seed": 66,
        "rate": 0.4,
        "metrics": {
            "graph_only": metrics,
            "predicted": metrics,
            "real_teacher": metrics,
            "shuffled_teacher_mean": metrics,
        },
        "deltas": {
            "real_minus_predicted": {name: 0.0 for name in metrics},
            "real_minus_shuffled_mean": {name: 0.0 for name in metrics},
        },
    }

    summary = _aggregate_summary([record], [66], [0.4], "commit")

    assert summary["high_missing_rates"] == {
        "rates": [],
        "per_seed": {
            "real_minus_predicted": [],
            "real_minus_shuffled": [],
        },
        "across_seed": {},
    }


def test_temporal_relation_order_remaps_ids_by_name_and_restores_global(monkeypatch):
    original_reference = gcnet_graph_model.batch_graphify

    def randomized_graphify(*_args, **_kwargs):
        return (
            torch.zeros(3, 2),
            torch.tensor([[0, 1, 2], [1, 2, 0]]),
            torch.tensor([0, 1, 2]),
            {"future": 0, "past": 1, "now": 2},
        )

    monkeypatch.setattr(gcnet_graph_model, "batch_graphify", randomized_graphify)
    patched_reference = gcnet_graph_model.batch_graphify

    with temporal_relation_order(("past", "now", "future")):
        _, _, edge_type, mapping = gcnet_graph_model.batch_graphify(
            None, None, None, 1, 2, 2, "temporal", True
        )
        assert torch.equal(edge_type, torch.tensor([2, 0, 1]))
        assert mapping == {"past": 0, "now": 1, "future": 2}

    assert gcnet_graph_model.batch_graphify is patched_reference
    assert gcnet_graph_model.batch_graphify is not original_reference


def test_temporal_relation_order_leaves_speaker_mapping_untouched(monkeypatch):
    def speaker_graphify(*_args, **_kwargs):
        return (
            torch.zeros(1, 2),
            torch.tensor([[0], [0]]),
            torch.tensor([0]),
            {"00": 0},
        )

    monkeypatch.setattr(gcnet_graph_model, "batch_graphify", speaker_graphify)
    with temporal_relation_order(("past", "now", "future")):
        _, _, edge_type, mapping = gcnet_graph_model.batch_graphify(
            None, None, None, 1, 2, 2, "speaker", True
        )
    assert torch.equal(edge_type, torch.tensor([0]))
    assert mapping == {"00": 0}


def test_relation_order_recovery_selects_the_unique_history_match(monkeypatch):
    expected_order = ("now", "future", "past")
    expected_metrics = {
        "weighted_f1": 0.7,
        "macro_f1": 0.69,
        "accuracy": 0.71,
        "mae": 0.8,
        "correlation": 0.6,
    }

    def base_graphify(*_args, **_kwargs):
        return (
            torch.zeros(1, 2),
            torch.tensor([[0], [0]]),
            torch.tensor([0]),
            {"past": 0, "now": 1, "future": 2},
        )

    def fake_quick_metrics(*_args, **_kwargs):
        _, _, _, mapping = gcnet_graph_model.batch_graphify(
            None, None, None, 1, 2, 2, "temporal", True
        )
        order = tuple(sorted(mapping, key=mapping.get))
        result = dict(expected_metrics)
        if order != expected_order:
            result["accuracy"] += 0.1
        return result

    monkeypatch.setattr(gcnet_graph_model, "batch_graphify", base_graphify)
    monkeypatch.setattr(runner, "_quick_validation_metrics", fake_quick_metrics)
    history = {"validation": {"0.4": expected_metrics}}

    selected, candidates = runner._discover_temporal_relation_order(
        model=None,
        loader=None,
        schedules={0.4: object()},
        dimensions=(1, 1, 1),
        device=torch.device("cpu"),
        history_record=history,
        requested_rates=[0.4],
    )

    assert selected == expected_order
    assert len(candidates) == 6
    assert sum(item["max_abs_error"] < 1e-6 for item in candidates) == 1
