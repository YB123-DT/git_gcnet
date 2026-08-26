"""Availability patterns and masks for PLCI auxiliary examples."""

from typing import Tuple

import torch


ACTIVE_PATTERNS = (
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1),
)


def sample_balanced_patterns(
    umask: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """Return ``[L, B, 3]`` with one uniform active pattern per utterance."""
    if umask.ndim != 2:
        raise ValueError("umask must have shape [B, L]")

    valid = umask.T.bool()
    availability = torch.zeros(
        (*valid.shape, 3), dtype=umask.dtype, device=umask.device
    )
    pattern_ids = torch.randint(
        len(ACTIVE_PATTERNS),
        (int(valid.sum().item()),),
        generator=generator,
        device="cpu",
    ).to(umask.device)
    patterns = torch.tensor(
        ACTIVE_PATTERNS, dtype=umask.dtype, device=umask.device
    )
    availability[valid] = patterns[pattern_ids]
    return availability


def expand_modality_mask(
    availability: torch.Tensor,
    dimensions: Tuple[int, int, int],
) -> torch.Tensor:
    """Expand ``[L, B, 3]`` availability across modality feature widths."""
    if availability.ndim != 3 or availability.shape[-1] != 3:
        raise ValueError("availability must have shape [L, B, 3]")
    if (
        not isinstance(dimensions, tuple)
        or len(dimensions) != 3
        or any(
            not isinstance(dimension, int) or dimension <= 0
            for dimension in dimensions
        )
    ):
        raise ValueError("dimensions must contain three positive integers")

    repeats = torch.tensor(dimensions, device=availability.device)
    return torch.repeat_interleave(availability, repeats, dim=-1)
