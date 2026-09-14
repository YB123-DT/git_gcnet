"""Write-state supervision uses the one causal forward, not old JEPA."""

import pytest
import torch

from gcnet_missing_m3 import train_gcnet
from gcnet_missing_m3.train_gcnet import TrainConfig
from gcnet_missing_m3.mixed_rate import MISSING_RATES
from test_missing_m3 import (
    _LifecycleModel, _CountingOptimizer, _install_train_epoch_lifecycle_fakes,
)


@pytest.mark.parametrize("rate,count", [(0.0, 0), (0.5, 5)])
def test_write_state_one_forward_loss_and_ema_lifecycle(monkeypatch, rate, count):
    _install_train_epoch_lifecycle_fakes(monkeypatch)

    class WriteModel(_LifecycleModel):
        def forward(self, *args, **kwargs):
            output = super().forward(*args, **kwargs)
            self.cached_predictions = output[0] + 2.0
            return output

        def write_state_loss(self, complete, availability, qmask, umask):
            assert complete.shape == (3, 1, 1)
            selected = availability.eq(0) & umask.T.bool().unsqueeze(-1)
            assert int(selected.sum()) == count
            return self.cached_predictions.sum() * (2.0 if count else 0.0), count

        def encode_teacher_targets(self, *_):
            raise AssertionError("legacy modality target path must not run")

        def update_teacher(self, tau):
            assert len(optimizer.step_gradients) == self.ema_calls + 1
            super().update_teacher(tau)

    def forbidden(*args, **kwargs):
        raise AssertionError("legacy M3 loss must not run")

    monkeypatch.setattr(train_gcnet, "missing_m3_loss", forbidden)
    model = WriteModel()
    optimizer = _CountingOptimizer(model.weight)
    cfg = TrainConfig(training_objective="write-state", backbone_type="osram",
                      osram_bidirectional=False, osram_write_step=.6,
                      train_rate_mode="fixed", fixed_missing_rate=rate)
    result = train_gcnet.train_epoch(model, [["batch"]], optimizer, cfg,
                                    {r: r for r in MISSING_RATES}, 0, (1, 1, 1),
                                    torch.device("cpu"))
    assert model.predict_missing_flags == [False]
    assert model.ema_calls == 1
    assert result["model_forward_count"] == 1
    assert result["write_state_target_count"] == count
    assert result["write_state_loss"] == pytest.approx(result["jepa_loss"])
    assert result["loss"] == pytest.approx(result["classification_loss"] + .1 * result["write_state_loss"])
    assert result["rate_jepa_target_counts"][str(rate)] == count
    multiplier = MISSING_RATES.index(rate) + 1
    assert optimizer.step_gradients[0].item() == pytest.approx(multiplier * (1.2 if count else 1.0))


@pytest.mark.parametrize("override", [
    {"osram_bidirectional": True}, {"osram_write_step": 1.0},
    {"classification_completion": True}, {"completion_path": "pre_osram_b2"},
    {"osram_readout_fusion": "local-gated"},
])
def test_write_state_configuration_rejects_other_interventions(override):
    config = dict(training_objective="write-state", backbone_type="osram",
                  osram_bidirectional=False, osram_write_step=.6)
    config.update(override)
    with pytest.raises(ValueError, match="write-state"):
        TrainConfig(**config)


def test_write_state_cli_and_config_roundtrip():
    from dataclasses import asdict

    args = train_gcnet.build_parser().parse_args([
        "--audio-feature", "a", "--text-feature", "t", "--video-feature", "v",
        "--output-dir", "out", "--training-objective", "write-state",
    ])
    assert args.training_objective == "write-state"
    cfg = TrainConfig(training_objective="write-state", backbone_type="osram",
                      osram_bidirectional=False, osram_write_step=.6)
    assert TrainConfig(**asdict(cfg)) == cfg
    assert TrainConfig().training_objective == "joint"


def test_write_evaluation_does_not_call_target_supervision(monkeypatch):
    _install_train_epoch_lifecycle_fakes(monkeypatch)

    class WriteEvaluationModel(_LifecycleModel):
        def write_state_loss(self, *args):
            raise AssertionError("evaluation must not generate teacher supervision")

        def encode_teacher_targets(self, *args):
            raise AssertionError("evaluation must not call teacher")

    model = WriteEvaluationModel()
    metrics, _ = train_gcnet.evaluate_rate(
        model, [["batch"]], .5, "IEMOCAPSix", (1, 1, 1), torch.device("cpu"), False,
    )
    assert model.predict_missing_flags == [False]
    assert model.teacher_calls == model.ema_calls == 0
    assert "loss" in metrics
