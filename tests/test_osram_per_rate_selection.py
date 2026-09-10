"""Deterministic checkpoint lifecycle tests; no real training or datasets."""
import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gcnet_missing_m3 import train_gcnet as trainer
from gcnet_missing_m3.mixed_rate import MISSING_RATES


@pytest.mark.parametrize("fusion", ["flat", "local-gated"])
def test_per_rate_oracle_restores_independent_earliest_maxima(monkeypatch, tmp_path, fusion):
    calls = []

    class Model(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))
            self.ema_step = 0
            self.readout_type = "shared"
            self.readout_rank = 8
            self.osram = SimpleNamespace(last_diagnostics={})
            assert kwargs["osram_readout_fusion"] == fusion

    monkeypatch.setattr(trainer, "MissingM3GraphModel", Model)
    monkeypatch.setattr(trainer, "get_loaders", lambda **kw: (
        [["train"]], [["validation"]], [["test"]], 1, 1, 1))
    monkeypatch.setattr(trainer, "_schedules", lambda cfg, split: {
        rate: (split, rate) for rate in MISSING_RATES})

    def epoch(model, loader, optimizer, cfg, schedules, epoch, *args):
        with torch.no_grad():
            model.weight.fill_(epoch + 1)
        return dict(weighted_f1=0.5, classification_loss=1., jepa_loss=0.)

    def evaluate(model, loader, schedule, *args, collect=False, **kwargs):
        split, rate = schedule
        selected_epoch = int(model.weight.item())
        calls.append((split, rate, selected_epoch, collect))
        model.osram.last_diagnostics = {"test_rate": rate, "epoch": selected_epoch}
        # First rate peaks at epoch 1 (ties at 2); others peak at epoch 2.
        scores = (0.9, 0.9, 0.1) if rate == MISSING_RATES[0] else (0.2, 0.8, 0.7)
        metrics = dict(weighted_f1=scores[selected_epoch - 1], mask_sha256=str(rate))
        return metrics, {"predictions": np.array([selected_epoch])} if collect else None

    monkeypatch.setattr(trainer, "train_epoch", epoch)
    monkeypatch.setattr(trainer, "evaluate_rate", evaluate)
    result = trainer.run_experiment(trainer.TrainConfig(
        dataset="CMUMOSI", fold=1, epochs=3, device="cpu",
        backbone_type="osram", osram_readout_fusion=fusion,
        checkpoint_selection="test-oracle-per-rate"), "a", "t", "v", tmp_path)
    expected = {format(rate, ".1f"): (1 if rate == MISSING_RATES[0] else 2)
                for rate in MISSING_RATES}
    assert result["selected_epoch_by_rate"] == expected
    assert result["best_epoch"] is None
    assert result["best_selection_mean_weighted_f1"] is None
    assert result["selection_protocol"] == "per-rate-test-oracle"
    assert result["selection_split"] == "test"
    assert not (tmp_path / "best.pt").exists()
    assert len(calls) == 4 * len(MISSING_RATES)
    assert all(split == "test" for split, *_ in calls)
    for rate in MISSING_RATES:
        key = format(rate, ".1f")
        suffix = key.replace(".", "p")
        checkpoint = torch.load(tmp_path / f"best_miss_{suffix}.pt", weights_only=False)
        assert checkpoint["epoch"] == expected[key]
        assert checkpoint["selection_rate"] == rate
        assert checkpoint["selection_split"] == "test"
        assert checkpoint["selection_protocol"] == "per-rate-test-oracle"
        assert checkpoint["selection_weighted_f1"] == result["test"][key]["weighted_f1"]
        assert "selection_mean_weighted_f1" not in checkpoint
        saved = np.load(tmp_path / f"predictions_miss_{suffix}.npz")
        assert saved["predictions"].item() == expected[key]
    assert json.loads((tmp_path / "metrics.json").read_text())["best_epoch"] is None
    diagnostics = json.loads((tmp_path / "diagnostics.json").read_text())
    if fusion == "local-gated":
        assert diagnostics["per_rate_scope"] == "last evaluation batch, not a dataset aggregate"
        for key, epoch in expected.items():
            record = diagnostics["selected_checkpoint_by_rate"][key]
            assert record["selected_epoch"] == epoch
            assert record["last_batch"] == {"test_rate": float(key), "epoch": epoch}
    else:
        assert "selected_checkpoint_by_rate" not in diagnostics


def test_local_gated_config_requires_per_rate_protocol_and_is_backward_compatible(tmp_path):
    with pytest.raises(ValueError, match="test-oracle-per-rate"):
        trainer.run_experiment(trainer.TrainConfig(
            backbone_type="osram", osram_readout_fusion="local-gated"),
            "a", "t", "v", tmp_path)
    cfg = trainer.TrainConfig(backbone_type="osram", osram_readout_fusion="local-gated",
                              checkpoint_selection="test-oracle-per-rate")
    assert trainer.TrainConfig(**asdict(cfg)).osram_readout_fusion == "local-gated"
    legacy = asdict(trainer.TrainConfig())
    legacy.pop("osram_readout_fusion")
    assert trainer.TrainConfig(**legacy).osram_readout_fusion == "flat"
    assert trainer.build_parser().get_default("osram_readout_fusion") == "flat"


@pytest.mark.parametrize("options", [
    {"train_rate_mode": "fixed", "fixed_missing_rate": 0.5},
    {"evaluate_test": False},
])
def test_per_rate_selection_cannot_silently_reduce_evaluation(options):
    with pytest.raises(ValueError, match="all eight rates and test evaluation"):
        trainer.TrainConfig(checkpoint_selection="test-oracle-per-rate", **options)


def test_cli_forwards_fusion_and_selection_and_persists_config(monkeypatch, tmp_path):
    captured = []
    monkeypatch.setattr(trainer, "run_experiment", lambda cfg, *args, **kwargs: captured.append(cfg))
    for name in ("a", "t", "v"):
        (tmp_path / name).mkdir()
    trainer.main([
        "--feature-root", str(tmp_path), "--audio-feature", "a",
        "--text-feature", "t", "--video-feature", "v", "--output-dir", str(tmp_path),
        "--backbone-type", "osram", "--osram-readout-fusion", "local-gated",
        "--checkpoint-selection", "test-oracle-per-rate",
    ])
    cfg = captured[0]
    trainer._write_run_config(tmp_path / "config.json", cfg)
    restored = trainer.TrainConfig(**json.loads((tmp_path / "config.json").read_text()))
    assert restored.osram_readout_fusion == "local-gated"
    assert restored.checkpoint_selection == "test-oracle-per-rate"
