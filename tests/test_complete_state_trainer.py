"""Complete-view state objective replaces (not stacks on) directional JEPA."""

import pytest
import torch

from gcnet_missing_m3 import train_gcnet
from gcnet_missing_m3.train_gcnet import TrainConfig
from gcnet_missing_m3.mixed_rate import MISSING_RATES
from test_missing_m3 import (
    _LifecycleModel, _CountingOptimizer, _install_train_epoch_lifecycle_fakes,
)


@pytest.mark.parametrize("rate,count", [(0.0, 0), (0.5, 2)])
def test_state_training_uses_single_forward_and_updates_ema_after_optimizer(monkeypatch, rate, count):
    _install_train_epoch_lifecycle_fakes(monkeypatch)

    class StateModel(_LifecycleModel):
        def forward(self, *args, **kwargs):
            logits, _, latent, predictions = super().forward(*args, **kwargs)
            self.returned_hidden = logits + 2.0
            return logits, self.returned_hidden, latent, predictions

        def complete_state_loss(self, hidden, complete, availability, umask):
            assert hidden is self.returned_hidden
            assert complete.shape == (3, 1, 1)
            selected = availability.eq(0).any(-1) & umask.T.bool()
            assert int(selected.sum()) == count
            return hidden.sum() * (2.0 if count else 0.0), count

        def encode_teacher_targets(self, *_):
            raise AssertionError("legacy modality teacher must not run")

        def update_teacher(self, tau):
            assert len(optimizer.step_gradients) == self.ema_calls + 1
            super().update_teacher(tau)

    def forbidden(*args, **kwargs):
        raise AssertionError("legacy M3 loss must not run")

    monkeypatch.setattr(train_gcnet, "missing_m3_loss", forbidden)
    model = StateModel()
    optimizer = _CountingOptimizer(model.weight)
    cfg = TrainConfig(training_objective="complete-state", backbone_type="osram",
                      osram_bidirectional=False, osram_write_step=0.6,
                      train_rate_mode="fixed", fixed_missing_rate=rate)
    result = train_gcnet.train_epoch(model, [["batch"]], optimizer, cfg,
                                    {r: r for r in MISSING_RATES}, 0, (1, 1, 1),
                                    torch.device("cpu"))
    assert model.predict_missing_flags == [False]
    assert model.ema_calls == 1
    assert result["model_forward_count"] == 1
    assert result["state_target_count"] == count
    assert result["state_loss"] == pytest.approx(result["jepa_loss"])
    assert result["loss"] == pytest.approx(result["classification_loss"] + .1 * result["state_loss"])
    assert result["rate_jepa_target_counts"][str(rate)] == count
    multiplier = MISSING_RATES.index(rate) + 1
    assert optimizer.step_gradients[0].item() == pytest.approx(multiplier * (1.2 if count else 1.0))


def test_state_objective_rejects_conflicting_completions():
    with pytest.raises(ValueError, match="complete-state"):
        TrainConfig(training_objective="complete-state", backbone_type="osram",
                    osram_bidirectional=False, classification_completion=True)


def test_legacy_config_keeps_original_objective():
    assert TrainConfig().training_objective == "joint"
