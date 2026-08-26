import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from gcnet_plci_jepa.modules import (
    EMATeacherBank,
    ModalityProjector,
    StudentAdapterBank,
    bounded_residual,
    normalize_latent,
)


MODALITIES = ("audio", "text", "visual")


def test_normalize_latent_matches_layer_norm_then_l2_normalization():
    value = torch.tensor(
        [
            [[1.0, 2.0, 4.0], [3.0, 3.0, 3.0]],
            [[-2.0, 0.0, 8.0], [1.0, -1.0, 0.0]],
        ]
    )

    result = normalize_latent(value)
    expected = F.normalize(
        F.layer_norm(value, (value.shape[-1],), eps=1e-6), dim=-1, eps=1e-6
    )

    assert result.shape == value.shape
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, expected)


def test_bounded_residual_matches_formula_and_respects_radius():
    value = torch.tensor([[[3.0, 4.0], [0.0, 0.0]], [[-6.0, 8.0], [1.0, -2.0]]])
    kappa = 0.7
    norm = torch.norm(value, dim=-1, keepdim=True)

    result = bounded_residual(value, kappa)
    expected = kappa * value / (norm + 1e-6) * torch.tanh(norm)

    assert result.shape == value.shape
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, expected)
    assert torch.all(torch.norm(result, dim=-1) <= kappa)


@pytest.mark.parametrize(
    "kappa", [0.0, -0.1, float("nan"), float("inf"), -float("inf"), "bad", 1j]
)
def test_bounded_residual_rejects_invalid_radius(kappa):
    with pytest.raises(ValueError, match="kappa"):
        bounded_residual(torch.ones(2, 3), kappa)


def test_modality_projector_has_the_required_dropout_free_structure():
    projector = ModalityProjector(input_dim=5, latent_dim=3)

    assert list(projector.children()) == [
        projector.input_norm,
        projector.input_projection,
        projector.activation,
        projector.latent_projection,
        projector.output_norm,
    ]
    assert isinstance(projector.input_norm, nn.LayerNorm)
    assert isinstance(projector.input_projection, nn.Linear)
    assert isinstance(projector.activation, nn.GELU)
    assert isinstance(projector.latent_projection, nn.Linear)
    assert isinstance(projector.output_norm, nn.LayerNorm)
    assert not any(
        isinstance(module, (nn.Dropout, nn.modules.batchnorm._BatchNorm))
        for module in projector.modules()
    )
    assert projector(torch.randn(4, 2, 5)).shape == (4, 2, 3)


def test_student_bank_zero_initializes_all_adapters():
    bank = StudentAdapterBank((2, 3, 4), latent_dim=5)

    assert tuple(bank.projectors.keys()) == MODALITIES
    assert tuple(bank.adapters.keys()) == MODALITIES
    for name, output_dim in zip(MODALITIES, (2, 3, 4)):
        adapter = bank.adapters[name]
        assert isinstance(adapter, nn.Linear)
        assert adapter.in_features == 5
        assert adapter.out_features == output_dim
        assert torch.count_nonzero(adapter.weight) == 0
        assert torch.count_nonzero(adapter.bias) == 0


def test_student_bank_routes_only_observed_incomplete_rows_and_bypasses_atv():
    dimensions = (2, 1, 3)
    bank = StudentAdapterBank(dimensions, latent_dim=4)
    masked_features = torch.arange(
        6 * sum(dimensions), dtype=torch.float32
    ).reshape(2, 3, -1) + 1.0
    availability = torch.tensor(
        [
            [[1, 0, 0], [1, 1, 0], [1, 1, 1]],
            [[0, 0, 1], [0, 1, 1], [0, 0, 0]],
        ],
        dtype=torch.float32,
    )
    slices = {
        "audio": slice(0, 2),
        "text": slice(2, 3),
        "visual": slice(3, 6),
    }
    seen = {}

    def capture(name):
        def hook(_module, inputs):
            seen[name] = inputs[0].detach().clone()

        return hook

    handles = [
        bank.projectors[name].register_forward_pre_hook(capture(name))
        for name in MODALITIES
    ]
    try:
        adapted, latents = bank(masked_features, availability)
    finally:
        for handle in handles:
            handle.remove()

    incomplete = (availability.sum(dim=-1) == 1) | (availability.sum(dim=-1) == 2)
    full = availability.sum(dim=-1) == 3
    padding = availability.sum(dim=-1) == 0
    for index, name in enumerate(MODALITIES):
        block = masked_features[..., slices[name]]
        selected = incomplete & availability[..., index].bool()
        torch.testing.assert_close(seen[name], block[selected])
        torch.testing.assert_close(
            latents[name][selected], bank.projectors[name](block[selected])
        )
        assert torch.count_nonzero(latents[name][~selected]) == 0
        torch.testing.assert_close(adapted[..., slices[name]][selected], block[selected])
        missing = incomplete & ~availability[..., index].bool()
        assert torch.count_nonzero(adapted[..., slices[name]][missing]) == 0

    assert torch.equal(adapted[full], masked_features[full])
    assert torch.count_nonzero(adapted[padding]) == 0


def test_student_bank_adds_adapter_residual_only_to_observed_incomplete_blocks():
    bank = StudentAdapterBank((2, 1, 3), latent_dim=4)
    for adapter in bank.adapters.values():
        nn.init.constant_(adapter.weight, 0.25)
        nn.init.constant_(adapter.bias, 0.5)
    features = torch.randn(1, 3, 6)
    availability = torch.tensor(
        [[[1, 0, 1], [1, 1, 1], [0, 1, 0]]], dtype=torch.float32
    )

    adapted, latents = bank(features, availability)

    starts = (0, 2, 3)
    for index, (name, width, start) in enumerate(
        zip(MODALITIES, (2, 1, 3), starts)
    ):
        block = features[..., start : start + width]
        selected = (availability.sum(dim=-1) < 3) & availability[..., index].bool()
        expected = block[selected] + bank.adapters[name](latents[name][selected])
        torch.testing.assert_close(
            adapted[..., start : start + width][selected], expected
        )
    assert torch.equal(adapted[0, 1], features[0, 1])


@pytest.mark.parametrize(
    "features,availability,message",
    [
        (torch.ones(2, 6), torch.ones(2, 1, 3), r"\[L, B, sumD\]"),
        (torch.ones(2, 1, 5), torch.ones(2, 1, 3), "feature dimension"),
        (torch.ones(2, 1, 6), torch.ones(2, 3), r"\[L, B, 3\]"),
        (torch.ones(2, 1, 6), torch.ones(3, 1, 3), "leading dimensions"),
    ],
)
def test_student_bank_rejects_invalid_forward_shapes(features, availability, message):
    bank = StudentAdapterBank((2, 1, 3), latent_dim=4)
    with pytest.raises(ValueError, match=message):
        bank(features, availability)


def test_ema_teacher_starts_identical_frozen_and_permanently_in_eval_mode():
    students = StudentAdapterBank((2, 3, 4), latent_dim=5).projectors
    teacher = EMATeacherBank(students)

    assert teacher.projectors is not students
    assert tuple(teacher.projectors.keys()) == MODALITIES
    for name in MODALITIES:
        assert teacher.projectors[name] is not students[name]
    for teacher_value, student_value in zip(
        teacher.state_dict().values(), students.state_dict().values()
    ):
        assert torch.equal(teacher_value, student_value)
    assert not teacher.training
    assert all(not module.training for module in teacher.modules())
    assert all(not parameter.requires_grad for parameter in teacher.parameters())

    returned = teacher.train(True)

    assert returned is teacher
    assert not teacher.training
    assert all(not module.training for module in teacher.modules())


def test_ema_teacher_splits_full_features_and_returns_pre_normalized_latents():
    students = StudentAdapterBank((2, 3, 4), latent_dim=5).projectors
    teacher = EMATeacherBank(students)
    full_features = torch.randn(3, 2, 9)

    latents = teacher(full_features)

    start = 0
    for name, width in zip(MODALITIES, (2, 3, 4)):
        expected = teacher.projectors[name](full_features[..., start : start + width])
        torch.testing.assert_close(latents[name], expected)
        assert latents[name].shape == (3, 2, 5)
        start += width


def test_ema_teacher_updates_parameters_and_float_buffers_exactly():
    students = StudentAdapterBank((2, 3, 4), latent_dim=5).projectors
    for projector in students.values():
        projector.register_buffer("ema_probe", torch.tensor([2.0]))
    teacher = EMATeacherBank(students)
    old_state = copy.deepcopy(teacher.state_dict())
    with torch.no_grad():
        for value in students.state_dict().values():
            value.add_(4.0)

    teacher.update_from(students, tau=0.25)

    for name, value in teacher.state_dict().items():
        student_name = name[len("projectors.") :]
        expected = (
            old_state[name] * 0.25
            + students.state_dict()[student_name] * 0.75
        )
        torch.testing.assert_close(value, expected)
    assert all(not parameter.requires_grad for parameter in teacher.parameters())


@pytest.mark.parametrize(
    "tau", [-0.01, 1.0, 1.1, float("nan"), float("inf"), -float("inf"), "bad", 1j]
)
def test_ema_teacher_rejects_invalid_tau(tau):
    students = StudentAdapterBank((2, 3, 4), latent_dim=5).projectors
    teacher = EMATeacherBank(students)
    with pytest.raises(ValueError, match="tau"):
        teacher.update_from(students, tau)
