"""Future-state training supervises adjacent transitions, including complete views."""

from dataclasses import asdict

import pytest
import torch

from gcnet_missing_m3 import train_gcnet
from gcnet_missing_m3.train_gcnet import TrainConfig
from gcnet_missing_m3.mixed_rate import MISSING_RATES
from test_missing_m3 import (
    _LifecycleModel, _CountingOptimizer, _install_train_epoch_lifecycle_fakes,
    _LifecycleLoader,
)


def future_config(**kwargs):
    values = dict(training_objective="future-state", backbone_type="osram",
                  osram_bidirectional=False, osram_write_step=.6)
    values.update(kwargs)
    return TrainConfig(**values)


@pytest.mark.parametrize("rate", [0.0, 0.7])
def test_future_one_forward_transition_loss_and_post_optimizer_ema(monkeypatch, rate):
    _install_train_epoch_lifecycle_fakes(monkeypatch)

    class FutureModel(_LifecycleModel):
        def forward(self, *args, **kwargs):
            output = super().forward(*args, **kwargs)
            self.cached = output[0]
            return output

        def future_state_loss(self, complete, umask):
            assert complete.shape == (3, 1, 1)
            transitions = umask[:, :-1].bool() & umask[:, 1:].bool()
            return self.cached.sum() * 2, int(transitions.sum())

        def encode_teacher_targets(self, *_):
            raise AssertionError("old MMoE supervision is forbidden")

        def update_teacher(self, tau):
            assert len(optimizer.step_gradients) == self.ema_calls + 1
            super().update_teacher(tau)

    def forbidden(*args, **kwargs):
        raise AssertionError("old MMoE loss is forbidden")

    monkeypatch.setattr(train_gcnet, "missing_m3_loss", forbidden)
    model = FutureModel()
    optimizer = _CountingOptimizer(model.weight)
    cfg = future_config(train_rate_mode="fixed", fixed_missing_rate=rate)
    result = train_gcnet.train_epoch(model, [["batch"]], optimizer, cfg,
                                    {r: r for r in MISSING_RATES}, 0, (1, 1, 1),
                                    torch.device("cpu"))
    assert model.predict_missing_flags == [False]
    assert model.ema_calls == result["model_forward_count"] == 1
    assert result["future_state_transition_count"] == 2
    assert result["rate_future_transition_counts"][str(rate)] == 2
    assert result["jepa_target_count"] == 2
    assert result["future_state_loss"] == result["jepa_loss"]
    assert result["loss"] == pytest.approx(result["classification_loss"] + .1 * result["future_state_loss"])
    assert optimizer.step_gradients[0].item() == pytest.approx(1.2 * (MISSING_RATES.index(rate) + 1))


@pytest.mark.parametrize("override", [
    {"backbone_type": "gcnet"}, {"osram_bidirectional": True},
    {"osram_write_step": 1.0}, {"osram_forward_slot_reuse": True},
    {"classification_completion": True}, {"completion_path": "pre_osram_b2"},
    {"osram_readout_fusion": "local-gated"}, {"osram_ablation": "local-only"},
    {"osram_emotion_ablation": "local-only"}, {"fusion_type": "slot"},
    {"jepa_rate_weighting": "missing"}, {"initial_backbone_checkpoint": "old.pt"},
    {"osram_query_availability": False}, {"local_context_residual": True},
    {"node_interaction_residual": True}, {"readout_type": "pattern"},
    {"jepa_weight": 0.2},
])
def test_future_configuration_rejects_other_interventions(override):
    with pytest.raises(ValueError, match="future-state"):
        future_config(**override)


def test_future_cli_and_config_roundtrip():
    args = train_gcnet.build_parser().parse_args([
        "--audio-feature", "a", "--text-feature", "t", "--video-feature", "v",
        "--output-dir", "out", "--training-objective", "future-state",
    ])
    assert args.training_objective == "future-state"
    cfg = future_config(checkpoint_selection="test-oracle-per-rate")
    assert TrainConfig(**asdict(cfg)) == cfg
    assert TrainConfig().training_objective == "joint"


def test_future_full_runner_rejects_average_epoch_selection_before_loading_data():
    with pytest.raises(ValueError, match="test-oracle-per-rate"):
        train_gcnet.run_experiment(future_config(checkpoint_selection="test-oracle"),
                                  "a", "t", "v", "unused")


def test_future_evaluation_has_no_auxiliary_teacher_call(monkeypatch):
    _install_train_epoch_lifecycle_fakes(monkeypatch)

    class EvaluationModel(_LifecycleModel):
        def future_state_loss(self, *args):
            raise AssertionError("evaluation must not compute future loss")

        def encode_teacher_targets(self, *args):
            raise AssertionError("evaluation must not call teacher")

    model = EvaluationModel()
    train_gcnet.evaluate_rate(model, [["batch"]], .7, "IEMOCAPSix",
                             (1, 1, 1), torch.device("cpu"), False)
    assert model.predict_missing_flags == [False]
    assert model.teacher_calls == model.ema_calls == 0


def test_future_stratified_counts_do_not_cross_padding_or_conversation(monkeypatch):
    lengths = [1, 2, 4]
    prepared, _, _ = _install_train_epoch_lifecycle_fakes(monkeypatch, lengths)

    class FutureModel(_LifecycleModel):
        def future_state_loss(self, complete, umask):
            count = int((umask[:, :-1].bool() & umask[:, 1:].bool()).sum())
            return self.weight * 2, count

    model = FutureModel()
    result = train_gcnet.train_epoch(
        model, _LifecycleLoader(["a", "b", "c"]), _CountingOptimizer(model.weight),
        future_config(train_rate_mode="stratified"), {r: r for r in MISSING_RATES},
        0, (1, 1, 1), torch.device("cpu"),
    )
    expected = {str(r): 0 for r in MISSING_RATES}
    for rate, length in zip(prepared, lengths):
        expected[str(rate)] += length - 1
    assert result["future_state_transition_count"] == 4
    assert result["rate_future_transition_counts"] == expected


def test_future_full_runner_passes_exclusive_model_flag(monkeypatch, tmp_path):
    class ConstructionCaptured(Exception):
        pass

    def capture(*args, **kwargs):
        assert kwargs["future_state_jepa"] is True
        assert kwargs["complete_state_jepa"] is False
        assert kwargs["write_state_completion"] is False
        assert kwargs["classification_completion"] is False
        assert kwargs["completion_path"] == "none"
        raise ConstructionCaptured

    monkeypatch.setattr(train_gcnet, "MissingM3GraphModel", capture)
    monkeypatch.setattr(train_gcnet, "get_loaders", lambda **kwargs: ([[]], [[]], [[]], 1, 1, 1))
    with pytest.raises(ConstructionCaptured):
        train_gcnet.run_experiment(
            future_config(dataset="CMUMOSI", fold=1, device="cpu",
                          checkpoint_selection="test-oracle-per-rate"),
            "a", "t", "v", tmp_path,
        )
