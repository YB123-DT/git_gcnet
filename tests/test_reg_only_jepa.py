"""Regression-only JEPA candidate: one-variable auxiliary-loss ablation."""

import pytest
import torch

from gcnet_missing_m3.loss import missing_m3_loss
from gcnet_missing_m3.model import ContextualM3Predictor, MissingM3Predictions
from gcnet_missing_m3.train_gcnet import TrainConfig


def _predictions():
    target_mask = torch.zeros(2, 1, 3, dtype=torch.bool)
    target_mask[..., 0] = True
    reg = torch.randn(2, 1, 3, 2, requires_grad=True)
    cl = torch.randn(2, 1, 3, 2, requires_grad=True)
    return MissingM3Predictions(
        reg_predictions=reg,
        cl_predictions=cl,
        target_mask=target_mask,
        source_counts=torch.ones(2, 1, dtype=torch.long),
    )


def _teacher():
    return {
        "audio": torch.randn(2, 1, 2),
        "text": torch.zeros(2, 1, 2),
        "visual": torch.zeros(2, 1, 2),
    }


def test_reg_only_keeps_half_regression_and_drops_contrastive():
    predictions = _predictions()
    teacher = _teacher()

    joint = missing_m3_loss(predictions, teacher, temperature=0.1)
    reg_only = missing_m3_loss(
        predictions,
        teacher,
        temperature=0.1,
        include_contrastive=False,
    )

    assert reg_only.contrastive.item() == 0.0
    assert torch.allclose(reg_only.total, 0.5 * reg_only.regression)
    assert torch.allclose(reg_only.regression, joint.regression)
    assert torch.allclose(joint.total, 0.5 * joint.regression + 0.5 * joint.contrastive)

    reg_only.total.backward()
    assert predictions.reg_predictions.grad is not None
    assert predictions.reg_predictions.grad.abs().sum() > 0
    assert predictions.cl_predictions.grad is None


def _frozen_teacher_osram_config(**overrides):
    values = dict(
        training_objective="joint-reg-only",
        teacher_mode="pretrained-frozen",
        teacher_checkpoint="teacher_projectors.pt",
        backbone_type="osram",
        osram_bidirectional=False,
        osram_write_step=0.6,
        osram_forward_slot_reuse=False,
        fusion_type="mean",
        osram_readout_fusion="flat",
        osram_predictor_mode="structured",
        osram_ablation="full",
        osram_emotion_ablation="full",
        local_context_residual=False,
        node_interaction_residual=False,
        readout_type="shared",
        classification_completion=False,
        completion_path="none",
        initial_backbone_checkpoint=None,
    )
    values.update(overrides)
    return values


def test_joint_reg_only_requires_frozen_teacher_and_locked_jepa_weight():
    config = TrainConfig(**_frozen_teacher_osram_config())
    assert config.training_objective == "joint-reg-only"
    assert config.jepa_weight == 0.1

    with pytest.raises(ValueError, match="joint-reg-only requires"):
        TrainConfig(**_frozen_teacher_osram_config(teacher_mode="ema",
                                                  teacher_checkpoint=None))
    with pytest.raises(ValueError, match="joint-reg-only requires"):
        TrainConfig(**_frozen_teacher_osram_config(jepa_weight=0.2))


def test_reg_only_regression_gradient_reaches_shared_hidden():
    torch.manual_seed(0)
    predictor = ContextualM3Predictor(4, 5, num_experts=2, top_k=1, dropout=0.0)
    latents = {
        name: torch.randn(3, 2, 4)
        for name in ("audio", "text", "visual")
    }
    hidden = torch.randn(3, 2, 5, requires_grad=True)
    umask = torch.ones(2, 3)
    availability = torch.ones(3, 2, 3)
    availability[..., 2] = 0.0
    teacher = {name: torch.randn(3, 2, 4) for name in latents}

    predictions = predictor(latents, hidden, availability, umask)
    result = missing_m3_loss(
        predictions,
        teacher,
        temperature=0.1,
        include_contrastive=False,
    )
    result.total.backward()

    assert result.target_count > 0
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()
    assert hidden.grad.abs().sum() > 0


def test_fixed_teacher_no_jepa_control_is_configurable():
    values = _frozen_teacher_osram_config(training_objective="emotion-only")
    config = TrainConfig(**values)
    assert config.training_objective == "emotion-only"
    assert config.teacher_mode == "pretrained-frozen"


def test_supervised_teacher_runner_exposes_three_stage2_objectives():
    from experiments.osram_supervised_teacher_20260914.run import student_config

    base = TrainConfig(
        dataset="CMUMOSI",
        backbone_type="osram",
        osram_bidirectional=False,
        osram_write_step=0.6,
        checkpoint_selection="test-oracle-per-rate",
    )
    for objective in ("joint", "joint-reg-only", "emotion-only"):
        config = student_config(base, "projectors.pt", objective)
        assert config.training_objective == objective
        assert config.teacher_mode == "pretrained-frozen"
        assert config.teacher_checkpoint == "projectors.pt"
        assert config.train_rate_mode == "cyclic"
