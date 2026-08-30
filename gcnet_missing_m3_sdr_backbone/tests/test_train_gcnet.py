import inspect
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from gcnet_missing_m3 import train_gcnet as base_train
from gcnet_missing_m3.mixed_rate import MISSING_RATES
from gcnet_missing_m3_sdr_backbone.model import MissingM3SDRModel
from gcnet_modality_jepa.protocol import SeedBundle


def test_run_experiment_exposes_keyword_only_reuse_hooks():
    from gcnet_missing_m3_sdr_backbone import train_gcnet

    parameters = inspect.signature(train_gcnet.run_experiment).parameters

    for name in ("model_builder", "result_identity"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters[name].default is None


def test_sdr_config_inherits_control_and_locks_registered_treatment():
    from gcnet_missing_m3_sdr_backbone.train_gcnet import SDRTrainConfig

    config = SDRTrainConfig()

    assert issubclass(SDRTrainConfig, base_train.TrainConfig)
    assert asdict(config) == {
        "dataset": "CMUMOSI",
        "fold": 1,
        "seed": 66,
        "base_model": "LSTM",
        "window_past": 2,
        "window_future": 2,
        "hidden": 200,
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
        "fixed_missing_rate": None,
        "graph_hidden": 100,
        "sdr_variant": "sdr-public",
    }


def test_sdr_config_allows_only_lifecycle_and_registered_variant():
    from gcnet_missing_m3_sdr_backbone.train_gcnet import SDRTrainConfig

    config = SDRTrainConfig(
        seed=70,
        device="cpu",
        epochs=2,
        evaluate_test=False,
        sdr_variant="sdr-paper",
    )
    assert config.sdr_variant == "sdr-paper"

    with pytest.raises(ValueError):
        SDRTrainConfig(hidden=199)
    with pytest.raises(ValueError):
        SDRTrainConfig(sdr_variant="public")


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("dataset", "IEMOCAPSix"),
        ("fold", 2),
        ("base_model", "GRU"),
        ("window_past", 1),
        ("window_future", 1),
        ("hidden", 100),
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
        ("fixed_missing_rate", 0.3),
        ("graph_hidden", 64),
    ],
)
def test_sdr_config_rejects_every_locked_field_drift(field, changed):
    from gcnet_missing_m3_sdr_backbone.train_gcnet import SDRTrainConfig

    with pytest.raises(ValueError):
        SDRTrainConfig(**{field: changed})


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


def test_parser_exposes_only_paths_lifecycle_variant_and_fixed_markers():
    from gcnet_missing_m3_sdr_backbone import train_gcnet

    parser = train_gcnet.build_parser()
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
        "--sdr-variant",
        "--audio-feature",
        "--text-feature",
        "--video-feature",
        "--feature-root",
        "--output-dir",
        "--seed",
        "--epochs",
        "--device",
        "--skip-test",
    }
    args = parser.parse_args(_required_cli())
    assert vars(args) == {
        "dataset": "CMUMOSI",
        "train_rate_mode": "all",
        "fusion_type": "slot",
        "lr": pytest.approx(5e-4),
        "sdr_variant": "sdr-public",
        "audio_feature": "audio",
        "text_feature": "text",
        "video_feature": "video",
        "feature_root": None,
        "output_dir": "output",
        "seed": 66,
        "epochs": 100,
        "device": "cuda",
        "skip_test": False,
    }


@pytest.mark.parametrize(
    "changed",
    [
        ["--dataset", "IEMOCAPSix"],
        ["--train-rate-mode", "cyclic"],
        ["--fusion-type", "mean"],
        ["--lr", "0.001"],
        ["--sdr-variant", "public"],
    ],
)
def test_parser_rejects_protocol_drift(changed):
    from gcnet_missing_m3_sdr_backbone import train_gcnet

    with pytest.raises(SystemExit):
        train_gcnet.build_parser().parse_args([*_required_cli(), *changed])


@pytest.mark.parametrize("variant", ["sdr-public", "sdr-paper"])
def test_build_model_constructs_registered_variant_with_500_wide_output(variant):
    from gcnet_missing_m3_sdr_backbone import train_gcnet

    config = train_gcnet.SDRTrainConfig(
        device="cpu",
        epochs=1,
        sdr_variant=variant,
    )
    model = train_gcnet.build_model(
        config,
        adim=2,
        tdim=3,
        vdim=4,
        device=torch.device("cpu"),
    )

    assert type(model) is MissingM3SDRModel
    assert model.sdr_variant == variant
    assert model.dimensions == (2, 3, 4)
    assert model.base_model == "LSTM"
    assert model.no_cuda is True
    assert model.n_speakers == 1
    assert (model.window_past, model.window_future) == (2, 2)
    assert model.latent_dim == 256
    assert model.conversation_backbone.input_dim == 256
    assert model.conversation_backbone.recurrent_hidden == 200
    assert model.conversation_backbone.graph_hidden == 100
    assert model.conversation_backbone.output_dim == 500
    assert model.smax_fc.in_features == 500
    assert model.smax_fc.out_features == 1
    assert model.observed_set.fusion_type == "slot"
    assert model.missing_predictor.mmoe.variant == "dual-gate"
    assert model.recurrent_padding_mode == "legacy"
    assert model.postgraph_sequence_mode == "independent"
    assert model.graph_message_calibration == "none"
    assert next(model.parameters()).device == torch.device("cpu")
    forbidden = (
        "lstm.",
        "gru.",
        "graph_net_temporal.",
        "graph_net_speaker.",
    )
    assert not any(
        name.startswith(forbidden) for name, _ in model.named_parameters()
    )


@pytest.mark.parametrize(
    ("variant", "full", "backbone"),
    [
        ("sdr-public", 11_170_118, 9_444_901),
        ("sdr-paper", 19_763_519, 18_038_302),
    ],
)
def test_verified_formal_parameter_counts_are_locked(variant, full, backbone):
    from gcnet_missing_m3_sdr_backbone import train_gcnet

    model = train_gcnet.build_model(
        train_gcnet.SDRTrainConfig(
            device="cpu",
            epochs=1,
            sdr_variant=variant,
        ),
        adim=2,
        tdim=3,
        vdim=4,
        device=torch.device("cpu"),
    )

    assert sum(parameter.numel() for parameter in model.parameters()) == full
    assert (
        sum(parameter.numel() for parameter in model.conversation_backbone.parameters())
        == backbone
    )
    assert sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    ) == full - 202_002


class _EpochSampler:
    def __init__(self):
        self.epochs = []

    def set_epoch(self, epoch):
        self.epochs.append(epoch)


class _Loader:
    def __init__(self):
        self.sampler = _EpochSampler()


def test_run_experiment_selects_only_eight_rate_validation_and_records_provenance(
    monkeypatch,
    tmp_path,
):
    from gcnet_missing_m3_sdr_backbone import train_gcnet

    config = train_gcnet.SDRTrainConfig(
        seed=73,
        device="cpu",
        epochs=2,
        evaluate_test=True,
        sdr_variant="sdr-paper",
    )
    train_loader = _Loader()
    validation_loader = _Loader()
    test_loader = _Loader()
    current_epoch = {"value": None}
    evaluations = []
    checkpoints = []
    seeds = []

    def get_loaders(**kwargs):
        assert kwargs == {
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
        return [train_loader], [validation_loader], [test_loader], 2, 3, 4

    def schedules(_config, split):
        return {rate: (split, rate) for rate in MISSING_RATES}

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
        assert type(model) is MissingM3SDRModel
        assert loader is train_loader
        assert isinstance(optimizer, torch.optim.Adam)
        assert config_value is config
        assert tuple(rate_schedules) == MISSING_RATES
        assert dimensions == (2, 3, 4)
        assert device == torch.device("cpu")
        current_epoch["value"] = epoch
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
        assert type(model) is MissingM3SDRModel
        assert dataset == "CMUMOSI"
        assert dimensions == (2, 3, 4)
        assert device == torch.device("cpu")
        assert kwargs == {
            "mosi_task_mode": "regression",
            "task_regression_loss": "mse",
            "task_smooth_l1_beta": 1.0,
        }
        split, rate = schedule
        if split == "validation":
            assert loader is validation_loader
            assert collect is False
            epoch = current_epoch["value"]
            # Epoch 2 ties epoch 1, so selection must retain the earliest.
            score = 0.80 - rate / 10.0
            artifacts = None
        else:
            assert split == "test"
            assert loader is test_loader
            assert collect is True
            epoch = None
            score = 0.70 - rate / 10.0
            artifacts = {
                "predictions": np.array([-1.0, 1.0], dtype=np.float32),
                "labels": np.array([-1.0, 1.0], dtype=np.float32),
                "availability": np.ones((2, 3), dtype=np.float32),
            }
        evaluations.append((split, epoch, rate))
        metrics = {
            "weighted_f1": score,
            "loss": 1.0 - score,
        }
        if collect:
            metrics.update(
                prediction_std=1.0,
                predicted_sign_count=2,
                mask_sha256=format(int(rate * 10) + 1, "064x"),
            )
        return metrics, artifacts

    real_seed = base_train.set_random_seed

    def set_random_seed(seed):
        seeds.append(seed)
        real_seed(seed)

    monkeypatch.setattr(train_gcnet, "get_loaders", get_loaders)
    monkeypatch.setattr(train_gcnet, "_schedules", schedules)
    monkeypatch.setattr(train_gcnet, "train_epoch", train_epoch)
    monkeypatch.setattr(train_gcnet, "evaluate_rate", evaluate_rate)
    monkeypatch.setattr(train_gcnet, "set_random_seed", set_random_seed)
    monkeypatch.setattr(
        train_gcnet,
        "_save_best_checkpoint",
        lambda *args, **kwargs: checkpoints.append((args, kwargs)),
    )

    result = train_gcnet.run_experiment(
        config,
        "audio-root",
        "text-root",
        "visual-root",
        tmp_path,
    )

    assert seeds == [
        73,
        SeedBundle(73).derive("missing_m3_model_init:fold:5"),
    ]
    assert train_loader.sampler.epochs == [0, 1]
    assert evaluations == [
        ("validation", epoch, rate)
        for epoch in (0, 1)
        for rate in MISSING_RATES
    ] + [("test", None, rate) for rate in MISSING_RATES]
    assert len(checkpoints) == 1
    assert result["best_epoch"] == 1
    expected_validation = {
        format(rate, ".1f"): {
            "weighted_f1": 0.80 - rate / 10.0,
            "loss": 0.20 + rate / 10.0,
        }
        for rate in MISSING_RATES
    }
    assert set(result["best_validation"]) == set(expected_validation)
    for rate, expected in expected_validation.items():
        assert result["best_validation"][rate] == pytest.approx(expected)
    assert result["best_validation_mean_weighted_f1"] == pytest.approx(
        sum(value["weighted_f1"] for value in expected_validation.values()) / 8
    )
    assert tuple(result["test"]) == tuple(
        format(rate, ".1f") for rate in MISSING_RATES
    )
    assert result["selection_missing_rates"] == list(MISSING_RATES)
    assert result["variant"] == "sdr-paper"
    assert result["sdr_variant"] == "sdr-paper"
    assert "sdr_input_type" not in result
    assert result["backbone"] == "sdr-gnn-whole-backbone"
    assert result["evaluation_stage"] == "train-validation-test"
    assert result["registered_parameters"] > 0
    assert result["trainable_parameters"] > 0
    assert result["registered_backbone_parameters"] > 0
    assert result["trainable_backbone_parameters"] > 0
    assert result["ema_steps"] == 0
    assert result["wall_time_seconds"] >= 0.0
    assert result["peak_memory_bytes"] == 0
    assert len(result["mask_sha256"]) == 8
    artifact_hash = train_gcnet._sha256_tensor(
        torch.ones(2, 3, dtype=torch.float32)
    )
    assert result["prediction_availability_sha256"] == {
        format(rate, ".1f"): artifact_hash for rate in MISSING_RATES
    }
    for rate in MISSING_RATES:
        rate_key = format(rate, ".1f")
        assert result["test"][rate_key][
            "prediction_availability_sha256"
        ] == artifact_hash
        assert result["test"][rate_key]["mask_sha256"] != artifact_hash
    assert json.loads((tmp_path / "config.json").read_text()) == asdict(config)
    assert len(json.loads((tmp_path / "history.json").read_text())) == 2
    assert json.loads((tmp_path / "metrics.json").read_text()) == result
    assert len(list(tmp_path.glob("predictions_miss_*.npz"))) == 8
    assert not list(tmp_path.rglob("*.tmp"))


@pytest.mark.parametrize(
    "result_identity",
    [
        {"best_epoch": "forbidden"},
        {"registered_parameters": "forbidden"},
        {"sdr_input_type": 1},
        [],
    ],
)
def test_run_experiment_rejects_uncontrolled_result_identity(
    result_identity,
    tmp_path,
):
    from gcnet_missing_m3_sdr_backbone import train_gcnet

    with pytest.raises((TypeError, ValueError), match="result_identity"):
        train_gcnet.run_experiment(
            train_gcnet.SDRTrainConfig(
                device="cpu",
                epochs=1,
                evaluate_test=False,
            ),
            "audio",
            "text",
            "visual",
            tmp_path,
            result_identity=result_identity,
        )


def test_run_experiment_can_skip_test_without_calling_test_evaluation(
    monkeypatch,
    tmp_path,
):
    from gcnet_missing_m3_sdr_backbone import train_gcnet

    config = train_gcnet.SDRTrainConfig(
        device="cpu",
        epochs=1,
        evaluate_test=False,
    )
    loader = _Loader()
    monkeypatch.setattr(
        train_gcnet,
        "get_loaders",
        lambda **_kwargs: ([loader], [loader], [loader], 2, 3, 4),
    )
    monkeypatch.setattr(
        train_gcnet,
        "_schedules",
        lambda _config, split: {rate: (split, rate) for rate in MISSING_RATES},
    )
    monkeypatch.setattr(
        train_gcnet,
        "train_epoch",
        lambda *args, **kwargs: {
            "weighted_f1": 0.5,
            "classification_loss": 0.2,
            "jepa_loss": 0.1,
        },
    )

    def evaluate_rate(*args, **kwargs):
        split, rate = args[2]
        assert split == "validation"
        assert kwargs["collect"] is False
        return {"weighted_f1": 0.5 + rate / 100.0, "loss": 0.5}, None

    monkeypatch.setattr(train_gcnet, "evaluate_rate", evaluate_rate)
    monkeypatch.setattr(
        train_gcnet,
        "_save_best_checkpoint",
        lambda *args, **kwargs: None,
    )
    result = train_gcnet.run_experiment(
        config,
        "audio",
        "text",
        "visual",
        tmp_path,
    )

    assert result["evaluation_stage"] == "train-validation-only"
    assert result["test"] == {}
    assert result["mask_sha256"] == {}
    assert not list(tmp_path.glob("predictions_miss_*.npz"))


def test_main_resolves_features_and_passes_only_open_fields(monkeypatch, tmp_path):
    from gcnet_missing_m3_sdr_backbone import train_gcnet

    captured = {}

    def run_experiment(config_value, *roots, output_dir):
        captured.update(config=config_value, roots=roots, output_dir=output_dir)

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
            "--sdr-variant",
            "sdr-paper",
            "--skip-test",
        ],
    )
    monkeypatch.setattr(train_gcnet.os.path, "exists", lambda _path: True)
    monkeypatch.setattr(train_gcnet, "run_experiment", run_experiment)

    train_gcnet.main()

    assert captured["config"].seed == 70
    assert captured["config"].epochs == 2
    assert captured["config"].device == "cpu"
    assert captured["config"].evaluate_test is False
    assert captured["config"].sdr_variant == "sdr-paper"
    assert captured["roots"] == tuple(
        str(tmp_path / name) for name in ("audio", "text", "video")
    )
    assert captured["output_dir"] == "output"
