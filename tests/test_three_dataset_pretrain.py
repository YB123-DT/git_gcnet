from __future__ import annotations

import importlib

import pytest
import torch


MODULE = "experiments.m3_pretrain_three_datasets_20260921.run"


def components():
    return importlib.import_module(MODULE)


def test_source_patterns_cover_single_and_double_observed_sets():
    module = components()
    assert module.SOURCE_PATTERNS == (
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
        (1, 0, 1),
        (0, 1, 1),
    )


def test_random_availability_masks_padding_and_never_all_missing():
    module = components()
    umask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.float32)
    availability = module.sample_source_availability(
        umask, generator=torch.Generator().manual_seed(7)
    )
    assert availability.shape == (4, 2, 3)
    assert torch.equal(availability[3, 0], torch.zeros(3))
    assert torch.equal(availability[2, 1], torch.zeros(3))
    assert torch.all(availability.sum(dim=-1)[umask.T.bool()] >= 1)
    observed = {tuple(row.tolist()) for row in availability[umask.T.bool()]}
    assert observed <= set(module.SOURCE_PATTERNS)


def test_joint_model_shared_projectors_teacher_and_loss_have_gradients():
    module = components()
    torch.manual_seed(3)
    model = module.ThreeDatasetJEPA(
        dimensions=(4, 5, 6), latent_dim=8, num_experts=2, top_k=1, dropout=0.0
    )
    features = torch.randn(5, 2, 15)
    umask = torch.ones(2, 5)
    availability = module.sample_source_availability(
        umask, generator=torch.Generator().manual_seed(11)
    )
    predictions, targets = model(features, availability, umask)
    loss = module.missing_m3_loss(
        predictions, targets, temperature=0.2, include_contrastive=True
    )
    assert loss.target_count > 0
    loss.total.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.projectors.parameters()
    )
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.predictor.parameters()
    )
    assert all(parameter.grad is None for parameter in model.teacher.parameters())


def test_source_latents_ignore_raw_values_in_missing_slots():
    module = components()
    torch.manual_seed(13)
    model = module.ThreeDatasetJEPA(
        dimensions=(3, 3, 3), latent_dim=6, num_experts=2, top_k=1, dropout=0.0
    )
    features = torch.randn(2, 1, 9)
    umask = torch.ones(1, 2)
    availability = torch.tensor(
        [[[1.0, 0.0, 1.0]], [[0.0, 1.0, 0.0]]]
    )
    changed = features.clone()
    changed[0, 0, 3:6] = 1000.0
    changed[1, 0, 0:3] = -1000.0
    changed[1, 0, 6:9] = 1000.0
    original = model.source_latents(features, availability, umask)
    perturbed = model.source_latents(changed, availability, umask)
    for name in module.MODALITIES:
        assert torch.equal(original[name], perturbed[name])


def test_route_metrics_report_train_target_mean_gap():
    module = components()
    prediction = torch.tensor([[1.0, 0.0], [3.0, 0.0]])
    target = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    metrics = module.latent_metrics(prediction, target, target_mean=torch.tensor([1.0, 0.0]))
    assert set(metrics) == {
        "count",
        "centered_cosine",
        "prediction_std",
        "target_std",
        "prediction_mse",
        "train_target_mean_baseline_gap",
    }
    assert metrics["count"] == 2
    assert metrics["prediction_mse"] == pytest.approx(0.5)
    assert metrics["train_target_mean_baseline_gap"] == pytest.approx(0.0)


def test_dataset_schedule_is_uniform_and_validation_is_separate():
    module = components()
    schedule = module.uniform_dataset_schedule(
        ["CMUMOSI", "CMUMOSEI", "IEMOCAPSix"],
        steps=300,
        generator=torch.Generator().manual_seed(9),
    )
    assert len(schedule) == 300
    counts = {name: schedule.count(name) for name in set(schedule)}
    assert set(counts) == {"CMUMOSI", "CMUMOSEI", "IEMOCAPSix"}
    assert max(counts.values()) - min(counts.values()) < 45
