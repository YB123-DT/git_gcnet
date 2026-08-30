from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

import gcnet_missing_m3.train_gcnet as base_train
from gcnet_missing_m3.mixed_rate import MISSING_RATES
from gcnet_missing_m3_sdt_backbone.model import (
    MissingM3SDTModel,
    SDTStyleConversationBackbone,
)
from gcnet_modality_jepa.protocol import SeedBundle


MODULE_NAME = "gcnet_missing_m3_sdt_backbone.train_gcnet"


def _trainer():
    spec = importlib.util.find_spec(MODULE_NAME)
    assert spec is not None, "independent SDT training entry is not implemented"
    return importlib.import_module(MODULE_NAME)


def _locked_defaults():
    return {
        "dataset": "CMUMOSI",
        "fold": 1,
        "seed": 66,
        "base_model": "LSTM",
        "window_past": 1,
        "window_future": 1,
        "hidden": 100,
        "dropout": 0.5,
        "batch_size": 32,
        "epochs": 100,
        "learning_rate": 5e-4,
        "weight_decay": 1e-5,
        "latent_dim": 256,
        "num_experts": 4,
        "top_k": 2,
        "projector_dropout": 0.1,
        "predictor_dropout": 0.1,
        "fusion_type": "slot",
        "local_context_residual": False,
        "local_fusion_hidden_dim": 256,
        "local_fusion_dropout": 0.2,
        "jepa_weight": 0.1,
        "temperature": 0.03,
        "ema_tau": 0.996,
        "gradient_clip_norm": 1.0,
        "time_attention": False,
        "evaluation_protocol": "official",
        "validation_fraction": 0.1,
        "device": "cuda",
        "train_rate_mode": "all",
        "mosi_task_mode": "regression",
        "graph_branch_mode": "both",
        "mmoe_variant": "dual-gate",
        "classification_completion": False,
        "representation_type": "slot",
        "node_interaction_residual": False,
        "readout_type": "shared",
        "readout_rank": 8,
        "evaluate_test": True,
        "jepa_regression_aggregation": "target",
        "recurrent_padding_mode": "legacy",
        "task_regression_loss": "mse",
        "task_smooth_l1_beta": 1.0,
        "postgraph_sequence_mode": "independent",
        "jepa_rate_weighting": "uniform",
        "graph_message_calibration": "none",
        "transformer_dim": 384,
        "transformer_heads": 8,
        "transformer_layers": 5,
        "transformer_ff_dim": 704,
        "transformer_max_len": 512,
    }


def test_sdt_config_inherits_control_and_locks_the_complete_treatment():
    module = _trainer()

    config = module.SDTTrainConfig()

    assert issubclass(module.SDTTrainConfig, base_train.TrainConfig)
    assert asdict(config) == _locked_defaults()


def test_sdt_config_allows_only_run_identity_and_lifecycle_changes():
    module = _trainer()

    config = module.SDTTrainConfig(
        seed=70,
        device="cpu",
        epochs=2,
        evaluate_test=False,
    )

    expected = _locked_defaults()
    expected.update(seed=70, device="cpu", epochs=2, evaluate_test=False)
    assert asdict(config) == expected


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("dataset", "IEMOCAPSix"),
        ("fold", 2),
        ("base_model", "GRU"),
        ("window_past", 2),
        ("window_future", 2),
        ("hidden", 200),
        ("dropout", 0.4),
        ("batch_size", 16),
        ("learning_rate", 1e-3),
        ("weight_decay", 0.0),
        ("latent_dim", 128),
        ("num_experts", 3),
        ("top_k", 1),
        ("projector_dropout", 0.0),
        ("predictor_dropout", 0.0),
        ("fusion_type", "mean"),
        ("local_context_residual", True),
        ("local_fusion_hidden_dim", 128),
        ("local_fusion_dropout", 0.0),
        ("jepa_weight", 0.2),
        ("temperature", 0.1),
        ("ema_tau", 0.99),
        ("gradient_clip_norm", 0.5),
        ("time_attention", True),
        ("evaluation_protocol", "strict"),
        ("validation_fraction", 0.2),
        ("train_rate_mode", "cyclic"),
        ("mosi_task_mode", "binary"),
        ("graph_branch_mode", "temporal-only"),
        ("mmoe_variant", "paper-faithful"),
        ("classification_completion", True),
        ("representation_type", "track"),
        ("node_interaction_residual", True),
        ("readout_type", "availability-affine"),
        ("readout_rank", 4),
        ("jepa_regression_aggregation", "utterance"),
        ("recurrent_padding_mode", "packed"),
        ("task_regression_loss", "smooth-l1"),
        ("task_smooth_l1_beta", 0.5),
        ("postgraph_sequence_mode", "shared-bilstm"),
        ("jepa_rate_weighting", "sparsity-budget"),
        ("graph_message_calibration", "branch-layernorm-residual"),
        ("transformer_dim", 256),
        ("transformer_heads", 4),
        ("transformer_layers", 4),
        ("transformer_ff_dim", 768),
        ("transformer_max_len", 256),
    ],
)
def test_sdt_config_rejects_every_locked_field_drift(field, changed):
    module = _trainer()

    with pytest.raises(ValueError):
        module.SDTTrainConfig(**{field: changed})


def _required_cli():
    return [
        "--audio-feature",
        "audio",
        "--text-feature",
        "text",
        "--video-feature",
        "video",
        "--output-dir",
        "output",
    ]


def test_parser_exposes_only_paths_lifecycle_and_explicit_fixed_protocol():
    module = _trainer()
    parser = module.build_parser()

    options = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert options == {
        "-h",
        "--help",
        "--dataset",
        "--train-rate-mode",
        "--fusion-type",
        "--lr",
        "--audio-feature",
        "--text-feature",
        "--video-feature",
        "--feature-root",
        "--output-dir",
        "--seed",
        "--epochs",
        "--device",
        "--skip-test-evaluation",
    }

    args = parser.parse_args(_required_cli())
    assert vars(args) == {
        "dataset": "CMUMOSI",
        "train_rate_mode": "all",
        "fusion_type": "slot",
        "lr": pytest.approx(5e-4),
        "audio_feature": "audio",
        "text_feature": "text",
        "video_feature": "video",
        "feature_root": None,
        "output_dir": "output",
        "seed": 66,
        "epochs": 100,
        "device": "cuda",
        "skip_test_evaluation": False,
    }


@pytest.mark.parametrize(
    "changed",
    [
        ["--dataset", "IEMOCAPSix"],
        ["--train-rate-mode", "cyclic"],
        ["--fusion-type", "mean"],
        ["--lr", "0.001"],
    ],
)
def test_parser_rejects_drift_from_each_explicit_fixed_protocol_field(changed):
    module = _trainer()

    with pytest.raises(SystemExit):
        module.build_parser().parse_args([*_required_cli(), *changed])


def test_build_model_constructs_only_the_locked_sdt_candidate_on_device():
    module = _trainer()
    config = module.SDTTrainConfig(device="cpu", epochs=1)

    model = module.build_model(
        config,
        adim=2,
        tdim=3,
        vdim=4,
        device=torch.device("cpu"),
    )

    assert type(model) is MissingM3SDTModel
    assert model.dimensions == (2, 3, 4)
    assert model.base_model == "LSTM"
    assert model.no_cuda is True
    assert model.n_speakers == 1
    assert (model.window_past, model.window_future) == (1, 1)
    assert model.time_attn is False
    assert model.latent_dim == 256
    assert model.observed_set.fusion_type == "slot"
    assert model.representation_type == "slot"
    assert model.graph_branch_mode == "both"
    assert model.missing_predictor.mmoe.variant == "dual-gate"
    assert model.missing_predictor.mmoe.num_experts == 4
    assert model.missing_predictor.mmoe.top_k == 2
    assert model.classification_completion is False
    assert model.local_context_residual is False
    assert model.node_interaction_residual is False
    assert model.readout_type == "shared"
    assert model.recurrent_padding_mode == "legacy"
    assert model.postgraph_sequence_mode == "independent"
    assert model.graph_message_calibration == "none"
    assert model.smax_fc.in_features == 250
    assert model.smax_fc.out_features == 1
    assert next(model.parameters()).device == torch.device("cpu")

    backbone = model.conversation_backbone
    assert type(backbone) is SDTStyleConversationBackbone
    assert backbone.input_dim == 256
    assert backbone.output_dim == 250
    assert backbone.n_speakers == 1
    assert backbone.validate_inputs is False
    assert backbone.input_projection.out_features == 384
    assert len(backbone.layers) == 5
    assert all(layer.self_attn.num_heads == 8 for layer in backbone.layers)
    assert all(layer.linear1.out_features == 704 for layer in backbone.layers)
    assert backbone.max_len == 512
    assert backbone.input_dropout.p == pytest.approx(0.5)

    registered_backbone = sum(
        parameter.numel() for parameter in backbone.parameters()
    )
    active_backbone = (
        registered_backbone - backbone.speaker_embedding.embedding_dim
    )
    assert registered_backbone == 5_869_754
    assert active_backbone == 5_869_370


class _EpochSampler:
    def __init__(self):
        self.epochs = []

    def set_epoch(self, epoch):
        self.epochs.append(epoch)


class _Loader:
    def __init__(self):
        self.sampler = _EpochSampler()


def _patch_primitive(monkeypatch, module, name, replacement):
    monkeypatch.setattr(module, name, replacement, raising=False)
    monkeypatch.setattr(base_train, name, replacement, raising=False)


def test_run_experiment_selects_earliest_tied_eight_rate_mean_and_records_sdt(
    monkeypatch,
    tmp_path,
):
    module = _trainer()
    config = module.SDTTrainConfig(
        seed=73,
        device="cpu",
        epochs=2,
        evaluate_test=True,
    )
    train_loader = _Loader()
    validation_loader = _Loader()
    test_loader = _Loader()
    loader_calls = []
    train_calls = []
    evaluation_calls = []
    checkpoint_calls = []
    seed_calls = []
    build_calls = []
    current_epoch = {"value": None}

    def get_loaders(**kwargs):
        loader_calls.append(kwargs)
        return (
            [train_loader],
            [validation_loader],
            [test_loader],
            2,
            3,
            4,
        )

    def schedules(_config, split):
        return {
            rate: {"split": split, "rate": rate}
            for rate in MISSING_RATES
        }

    def train_epoch(
        model,
        loader,
        optimizer,
        config_value,
        rate_schedules,
        epoch,
        dimensions,
        device,
    ):
        assert type(model) is MissingM3SDTModel
        assert loader is train_loader
        assert isinstance(optimizer, torch.optim.Adam)
        assert config_value is config
        assert tuple(rate_schedules) == MISSING_RATES
        assert dimensions == (2, 3, 4)
        assert device == torch.device("cpu")
        current_epoch["value"] = epoch
        train_calls.append(epoch)
        return {
            "weighted_f1": 0.5,
            "classification_loss": 0.25,
            "jepa_loss": 0.05,
        }

    def evaluate_rate(
        model,
        loader,
        schedule,
        dataset,
        dimensions,
        device,
        collect,
        **kwargs,
    ):
        assert type(model) is MissingM3SDTModel
        assert dataset == "CMUMOSI"
        assert dimensions == (2, 3, 4)
        assert device == torch.device("cpu")
        assert kwargs == {
            "mosi_task_mode": "regression",
            "task_regression_loss": "mse",
            "task_smooth_l1_beta": 1.0,
        }
        split = schedule["split"]
        rate = schedule["rate"]
        if split == "validation":
            assert loader is validation_loader
            assert collect is False
            epoch = current_epoch["value"]
            score = 0.8 - rate / 10.0
            artifacts = None
        else:
            assert split == "test"
            assert loader is test_loader
            assert collect is True
            epoch = None
            score = 0.7 - rate / 10.0
            artifacts = {
                "predictions": np.array([rate], dtype=np.float32),
                "labels": np.array([0.0], dtype=np.float32),
                "availability": np.ones((1, 3), dtype=np.float32),
            }
        evaluation_calls.append((split, epoch, rate))
        return {
            "weighted_f1": score,
            "loss": 1.0 - score,
            "mask_sha256": "mask-{:0.1f}".format(rate),
        }, artifacts

    real_seed = base_train.set_random_seed

    def set_random_seed(seed):
        seed_calls.append(seed)
        real_seed(seed)

    real_build_model = module.build_model

    def build_model(config_value, adim, tdim, vdim, device):
        build_calls.append((config_value, adim, tdim, vdim, device))
        return real_build_model(config_value, adim, tdim, vdim, device)

    def state_to_cpu(model):
        return model.state_dict()

    def save_best_checkpoint(
        path,
        model_state,
        config_value,
        epoch,
        validation_mean_weighted_f1,
    ):
        checkpoint_calls.append(
            (
                Path(path),
                model_state,
                config_value,
                epoch,
                validation_mean_weighted_f1,
            )
        )

    _patch_primitive(monkeypatch, module, "get_loaders", get_loaders)
    _patch_primitive(monkeypatch, module, "_schedules", schedules)
    _patch_primitive(monkeypatch, module, "train_epoch", train_epoch)
    _patch_primitive(monkeypatch, module, "evaluate_rate", evaluate_rate)
    _patch_primitive(monkeypatch, module, "set_random_seed", set_random_seed)
    _patch_primitive(monkeypatch, module, "_state_to_cpu", state_to_cpu)
    _patch_primitive(
        monkeypatch,
        module,
        "_save_best_checkpoint",
        save_best_checkpoint,
    )
    monkeypatch.setattr(module, "build_model", build_model)

    result = module.run_experiment(
        config,
        "audio-root",
        "text-root",
        "visual-root",
        tmp_path,
    )

    expected_seed = SeedBundle(73).derive("missing_m3_model_init:fold:5")
    assert seed_calls == [73, expected_seed]
    assert len(build_calls) == 1
    assert build_calls[0][0] is config
    assert build_calls[0][1:4] == (2, 3, 4)
    assert build_calls[0][4] == torch.device("cpu")
    assert len(loader_calls) == 1
    assert loader_calls[0] == {
        "audio_root": "audio-root",
        "text_root": "text-root",
        "video_root": "visual-root",
        "num_folder": 1,
        "dataset": "CMUMOSI",
        "batch_size": 32,
        "num_workers": 0,
        "seed": 73,
        "validation_fraction": 0.1,
        "evaluation_protocol": "official",
    }
    assert train_calls == [0, 1]
    assert train_loader.sampler.epochs == [0, 1]
    assert evaluation_calls == [
        ("validation", epoch, rate)
        for epoch in (0, 1)
        for rate in MISSING_RATES
    ] + [
        ("test", None, rate)
        for rate in MISSING_RATES
    ]
    assert len(checkpoint_calls) == 1
    assert checkpoint_calls[0][0] == tmp_path / "best.pt"
    assert checkpoint_calls[0][2] is config
    assert checkpoint_calls[0][3] == 1
    assert result["best_epoch"] == 1
    assert result["best_validation_mean_weighted_f1"] == pytest.approx(
        sum(0.8 - rate / 10.0 for rate in MISSING_RATES)
        / len(MISSING_RATES)
    )
    assert tuple(result["test"]) == tuple(
        format(rate, ".1f") for rate in MISSING_RATES
    )
    assert result["evaluation_stage"] == "train-validation-test"

    model = build_calls[0] and real_build_model(
        module.SDTTrainConfig(device="cpu", epochs=1),
        2,
        3,
        4,
        torch.device("cpu"),
    )
    assert result["registered_parameters"] == sum(
        parameter.numel() for parameter in model.parameters()
    )
    assert result["trainable_parameters"] == sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    assert result["backbone"] == "sdt-style-full-context"
    assert result["transformer"] == {
        "d_model": 384,
        "heads": 8,
        "layers": 5,
        "ff_dim": 704,
    }
    assert result["registered_backbone_parameters"] == 5_869_754
    assert result["active_backbone_parameters"] == 5_869_370
    assert result["control_active_backbone_parameters"] == 5_864_700

    assert json.loads((tmp_path / "config.json").read_text(encoding="utf-8")) == asdict(
        config
    )
    history = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
    assert [record["epoch"] for record in history] == [1, 2]
    assert json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8")) == result
    assert len(list(tmp_path.glob("predictions_miss_*.npz"))) == len(MISSING_RATES)
    assert not list(tmp_path.rglob("*.tmp"))


def test_main_resolves_feature_paths_and_passes_only_open_config_fields(
    monkeypatch,
    tmp_path,
):
    module = _trainer()
    captured = {}

    def run_experiment(config_value, audio_root, text_root, visual_root, output_dir):
        captured.update(
            config=config_value,
            roots=(audio_root, text_root, visual_root),
            output_dir=output_dir,
        )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_gcnet.py",
            *_required_cli(),
            "--feature-root",
            str(tmp_path),
            "--seed",
            "70",
            "--epochs",
            "2",
            "--device",
            "cpu",
            "--skip-test-evaluation",
        ],
    )
    monkeypatch.setattr(module.os.path, "exists", lambda _path: True)
    monkeypatch.setattr(module, "run_experiment", run_experiment)

    module.main()

    assert asdict(captured["config"]) == {
        **_locked_defaults(),
        "seed": 70,
        "epochs": 2,
        "device": "cpu",
        "evaluate_test": False,
    }
    assert captured["roots"] == tuple(
        str(tmp_path / name) for name in ("audio", "text", "video")
    )
    assert captured["output_dir"] == "output"
