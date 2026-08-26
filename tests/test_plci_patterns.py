import pytest
import torch

from gcnet_plci_jepa.patterns import (
    ACTIVE_PATTERNS,
    expand_modality_mask,
    sample_balanced_patterns,
)


def test_active_patterns_have_fixed_modality_mapping():
    assert ACTIVE_PATTERNS == (
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
        (1, 0, 1),
        (0, 1, 1),
    )


def test_balanced_sampler_uses_only_active_patterns_and_zeros_padding():
    generator = torch.Generator().manual_seed(66)
    umask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

    availability = sample_balanced_patterns(umask, generator)

    assert availability.shape == (3, 2, 3)
    assert torch.count_nonzero(availability[~umask.T.bool()]) == 0
    assert all(
        tuple(row.tolist()) in ACTIVE_PATTERNS
        for row in availability[umask.T.bool()]
    )


def test_balanced_sampler_is_uniform_over_independent_utterance_draws():
    generator = torch.Generator().manual_seed(66)
    availability = sample_balanced_patterns(torch.ones(1, 6000), generator)
    rows = availability[:, 0]

    counts = torch.tensor(
        [
            torch.all(rows == torch.tensor(pattern), dim=-1).sum()
            for pattern in ACTIVE_PATTERNS
        ]
    )

    assert torch.all((counts >= 800) & (counts <= 1200)), counts


def test_balanced_sampler_replays_generator_state_without_global_rng():
    umask = torch.ones(2, 5)
    first_generator = torch.Generator().manual_seed(314)
    second_generator = torch.Generator().manual_seed(314)
    global_state = torch.random.get_rng_state().clone()

    first = sample_balanced_patterns(umask, first_generator)
    second = sample_balanced_patterns(umask, second_generator)

    assert torch.equal(first, second)
    assert torch.equal(torch.random.get_rng_state(), global_state)


@pytest.mark.parametrize("umask", [torch.ones(3), torch.ones(2, 3, 1)])
def test_balanced_sampler_rejects_non_matrix_umask(umask):
    with pytest.raises(ValueError, match=r"\[B, L\]"):
        sample_balanced_patterns(umask, torch.Generator().manual_seed(66))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_balanced_sampler_accepts_cpu_generator_with_cuda_umask():
    generator = torch.Generator().manual_seed(66)
    umask = torch.tensor([[1.0, 1.0, 0.0]], device="cuda")

    availability = sample_balanced_patterns(umask, generator)

    assert availability.device == umask.device
    assert torch.count_nonzero(availability[~umask.T.bool()]) == 0
    assert all(
        tuple(row.tolist()) in ACTIVE_PATTERNS
        for row in availability[umask.T.bool()]
    )


def test_expand_modality_mask_repeats_each_modality_by_its_dimension():
    availability = torch.tensor(
        [[[1, 0, 1], [0, 1, 1]], [[1, 1, 0], [0, 0, 0]]]
    )

    expanded = expand_modality_mask(availability, (2, 1, 3))

    expected = torch.tensor(
        [
            [[1, 1, 0, 1, 1, 1], [0, 0, 1, 1, 1, 1]],
            [[1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
        ]
    )
    assert torch.equal(expanded, expected)


@pytest.mark.parametrize(
    "availability",
    [torch.ones(2, 3), torch.ones(2, 3, 2), torch.ones(2, 3, 3, 1)],
)
def test_expand_modality_mask_rejects_invalid_availability_shape(availability):
    with pytest.raises(ValueError, match=r"\[L, B, 3\]"):
        expand_modality_mask(availability, (2, 3, 4))


@pytest.mark.parametrize("dimensions", [(2, 3), (2, 0, 4), (2, -1, 4)])
def test_expand_modality_mask_rejects_invalid_dimensions(dimensions):
    with pytest.raises(ValueError, match="three positive"):
        expand_modality_mask(torch.ones(2, 3, 3), dimensions)
