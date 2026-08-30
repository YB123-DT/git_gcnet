import json
import subprocess
import sys
from dataclasses import asdict

import pytest
import torch

from gcnet_missing_m3.mixed_rate import MISSING_RATES
from gcnet_missing_m3_sdr_backbone import train_gcnet as shared_train
from gcnet_missing_m3_sdr_backbone.train_gcnet import SDRTrainConfig
from gcnet_modality_jepa.protocol import SeedBundle

import gcnet_missing_m3_raw_sdr as raw_sdr_package
from gcnet_missing_m3_raw_sdr.model import MissingM3RawSDRModel
from gcnet_missing_m3_raw_sdr import train_gcnet


def test_package_exports_only_the_formal_model_and_training_surface():
    assert set(raw_sdr_package.__all__) == {
        "MissingM3RawSDRModel",
        "RawSDRTrainConfig",
        "build_model",
        "run_experiment",
    }
    assert raw_sdr_package.MissingM3RawSDRModel is MissingM3RawSDRModel
    assert raw_sdr_package.RawSDRTrainConfig is train_gcnet.RawSDRTrainConfig
    assert raw_sdr_package.build_model is train_gcnet.build_model
    assert raw_sdr_package.run_experiment is train_gcnet.run_experiment


def test_raw_config_inherits_sdr_protocol_and_opens_only_lifecycle_fields():
    config = train_gcnet.RawSDRTrainConfig()

    assert issubclass(train_gcnet.RawSDRTrainConfig, SDRTrainConfig)
    assert config.fusion_type == "raw-residual"
    assert config.sdr_variant == "sdr-public"
    assert config.sdr_input_type == "raw-residual"
    assert config.learning_rate == pytest.approx(5e-4)
    assert config.train_rate_mode == "all"
    assert config.epochs == 100
    assert config._OPEN_FIELDS == {
        "seed",
        "device",
        "epochs",
        "evaluate_test",
    }
    assert asdict(config)["sdr_input_type"] == "raw-residual"

    changed = train_gcnet.RawSDRTrainConfig(
        seed=71,
        device="cpu",
        epochs=2,
        evaluate_test=False,
    )
    assert (changed.seed, changed.device, changed.epochs) == (71, "cpu", 2)
    assert changed.evaluate_test is False


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("base_model", "GRU"),
        ("hidden", 199),
        ("learning_rate", 1e-3),
        ("train_rate_mode", "cyclic"),
        ("fusion_type", "slot"),
        ("representation_type", "track"),
        ("sdr_variant", "sdr-paper"),
        ("sdr_input_type", "slot"),
    ],
)
def test_raw_config_rejects_every_structural_drift(name, value):
    with pytest.raises(ValueError, match=name):
        train_gcnet.RawSDRTrainConfig(**{name: value})


def test_build_model_constructs_only_formal_raw_public_sdr_model():
    model = train_gcnet.build_model(
        train_gcnet.RawSDRTrainConfig(device="cpu", epochs=1),
        adim=512,
        tdim=1024,
        vdim=1024,
        device=torch.device("cpu"),
    )

    assert type(model) is MissingM3RawSDRModel
    assert model.dimensions == (512, 1024, 1024)
    assert model.conversation_backbone.input_dim == 2560
    assert model.conversation_backbone.variant == "sdr-public"
    assert not hasattr(model.conversation_backbone, "speaker_branch")
    assert next(model.parameters()).device == torch.device("cpu")


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


def test_parser_exposes_paths_lifecycle_and_fixed_protocol_markers():
    parser = train_gcnet.build_parser()
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert options == {
        "-h",
        "--help",
        "--audio-feature",
        "--text-feature",
        "--video-feature",
        "--feature-root",
        "--output-dir",
        "--train-rate-mode",
        "--lr",
        "--seed",
        "--epochs",
        "--device",
        "--skip-test",
    }
    assert vars(parser.parse_args(_required_cli())) == {
        "audio_feature": "audio",
        "text_feature": "text",
        "video_feature": "video",
        "feature_root": None,
        "output_dir": "output",
        "train_rate_mode": "all",
        "lr": pytest.approx(5e-4),
        "seed": 66,
        "epochs": 100,
        "device": "cuda",
        "skip_test": False,
    }


def test_parser_accepts_explicit_formal_protocol_markers():
    args = train_gcnet.build_parser().parse_args(
        [
            *_required_cli(),
            "--train-rate-mode",
            "all",
            "--lr",
            "0.0005",
        ]
    )

    assert args.train_rate_mode == "all"
    assert args.lr == pytest.approx(5e-4)


@pytest.mark.parametrize(
    "changed",
    [
        ["--fusion-type", "slot"],
        ["--sdr-variant", "sdr-paper"],
        ["--train-rate-mode", "cyclic"],
        ["--lr", "0.001"],
    ],
)
def test_parser_does_not_accept_structure_search(changed):
    with pytest.raises(SystemExit):
        train_gcnet.build_parser().parse_args([*_required_cli(), *changed])


def test_module_entrypoint_does_not_eagerly_import_itself():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "gcnet_missing_m3_raw_sdr.train_gcnet",
            "--help",
        ],
        cwd=str(__file__.rsplit("/gcnet_missing_m3_raw_sdr/", 1)[0]),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "RuntimeWarning" not in completed.stderr
    assert "--train-rate-mode" in completed.stdout
    assert "--lr" in completed.stdout


class _EpochSampler:
    def __init__(self):
        self.epochs = []

    def set_epoch(self, epoch):
        self.epochs.append(epoch)


class _Loader:
    def __init__(self):
        self.sampler = _EpochSampler()


def test_run_experiment_reuses_shared_lifecycle_and_records_raw_identity(
    monkeypatch,
    tmp_path,
):
    config = train_gcnet.RawSDRTrainConfig(
        seed=79,
        device="cpu",
        epochs=1,
        evaluate_test=False,
    )
    loader = _Loader()
    seeds = []

    monkeypatch.setattr(
        shared_train,
        "get_loaders",
        lambda **_kwargs: ([loader], [loader], [loader], 2, 3, 4),
    )
    monkeypatch.setattr(
        shared_train,
        "_schedules",
        lambda _config, split: {
            rate: (split, rate) for rate in MISSING_RATES
        },
    )

    def train_epoch(model, *_args, **_kwargs):
        assert type(model) is MissingM3RawSDRModel
        return {
            "weighted_f1": 0.5,
            "classification_loss": 0.2,
            "jepa_loss": 0.1,
        }

    def evaluate_rate(model, _loader, schedule, *_args, **kwargs):
        assert type(model) is MissingM3RawSDRModel
        split, rate = schedule
        assert split == "validation"
        assert kwargs["collect"] is False
        return {"weighted_f1": 0.6 + rate / 100.0, "loss": 0.4}, None

    real_seed = shared_train.set_random_seed

    def set_random_seed(seed):
        seeds.append(seed)
        real_seed(seed)

    monkeypatch.setattr(shared_train, "train_epoch", train_epoch)
    monkeypatch.setattr(shared_train, "evaluate_rate", evaluate_rate)
    monkeypatch.setattr(shared_train, "set_random_seed", set_random_seed)
    monkeypatch.setattr(
        shared_train,
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

    assert seeds == [
        79,
        SeedBundle(79).derive("missing_m3_model_init:fold:5"),
    ]
    assert loader.sampler.epochs == [0]
    assert result["variant"] == "raw-residual-sdr-public"
    assert result["sdr_variant"] == "sdr-public"
    assert result["sdr_input_type"] == "raw-residual"
    assert result["backbone"] == "raw-residual-sdr-public"
    assert result["registered_parameters"] > 0
    assert result["trainable_parameters"] > 0
    assert result["registered_backbone_parameters"] > 0
    assert result["trainable_backbone_parameters"] > 0
    assert json.loads((tmp_path / "config.json").read_text()) == asdict(config)
    assert json.loads((tmp_path / "metrics.json").read_text()) == result


def test_main_resolves_paths_and_constructs_only_open_config(monkeypatch, tmp_path):
    captured = {}
    config_class = train_gcnet.RawSDRTrainConfig

    def build_config(**kwargs):
        captured["config_kwargs"] = kwargs
        return config_class(**kwargs)

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
            "72",
            "--epochs",
            "2",
            "--device",
            "cpu",
            "--skip-test",
            "--train-rate-mode",
            "all",
            "--lr",
            "0.0005",
        ],
    )
    monkeypatch.setattr(train_gcnet.os.path, "exists", lambda _path: True)
    monkeypatch.setattr(train_gcnet, "RawSDRTrainConfig", build_config)
    monkeypatch.setattr(train_gcnet, "run_experiment", run_experiment)

    train_gcnet.main()

    assert type(captured["config"]) is config_class
    assert captured["config_kwargs"] == {
        "seed": 72,
        "epochs": 2,
        "device": "cpu",
        "evaluate_test": False,
        "train_rate_mode": "all",
        "learning_rate": pytest.approx(5e-4),
    }
    assert captured["config"].seed == 72
    assert captured["config"].epochs == 2
    assert captured["config"].device == "cpu"
    assert captured["config"].evaluate_test is False
    assert captured["roots"] == tuple(
        str(tmp_path / name) for name in ("audio", "text", "video")
    )
    assert captured["output_dir"] == "output"
