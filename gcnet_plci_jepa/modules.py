"""Projection, residual-adapter, and EMA-teacher modules for PLCI-JEPA."""

import copy
import math
from numbers import Real
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F


MODALITIES = ("audio", "text", "visual")


def _is_finite_real(value: object) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def normalize_latent(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Apply non-affine layer normalization and L2 normalization."""
    normalized = F.layer_norm(value, (value.shape[-1],), eps=eps)
    return F.normalize(normalized, dim=-1, eps=eps)


def bounded_residual(
    value: torch.Tensor,
    kappa: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Smoothly bound each last-dimension vector to radius ``kappa``."""
    if not _is_finite_real(kappa) or kappa <= 0:
        raise ValueError("kappa must be a finite positive real number")
    norm = torch.norm(value, dim=-1, keepdim=True)
    return kappa * value / (norm + eps) * torch.tanh(norm)


def _validate_dimensions(dimensions: Tuple[int, int, int]) -> None:
    if (
        not isinstance(dimensions, tuple)
        or len(dimensions) != 3
        or any(
            not isinstance(dimension, int) or dimension <= 0
            for dimension in dimensions
        )
    ):
        raise ValueError("dimensions must contain three positive integers")


class ModalityProjector(nn.Module):
    """Map one modality into the shared PLCI latent space."""

    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        if input_dim <= 0 or latent_dim <= 0:
            raise ValueError("input_dim and latent_dim must be positive")
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, latent_dim)
        self.activation = nn.GELU()
        self.latent_projection = nn.Linear(latent_dim, latent_dim)
        self.output_norm = nn.LayerNorm(latent_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.input_norm(value)
        value = self.input_projection(value)
        value = self.activation(value)
        value = self.latent_projection(value)
        return self.output_norm(value)


class StudentAdapterBank(nn.Module):
    """Project observed incomplete modalities and inject zero-init residuals."""

    def __init__(
        self,
        dimensions: Tuple[int, int, int],
        latent_dim: int,
    ) -> None:
        super().__init__()
        _validate_dimensions(dimensions)
        if not isinstance(latent_dim, int) or latent_dim <= 0:
            raise ValueError("latent_dim must be a positive integer")
        self.dimensions = dimensions
        self.latent_dim = latent_dim
        self.projectors = nn.ModuleDict(
            {
                name: ModalityProjector(dimension, latent_dim)
                for name, dimension in zip(MODALITIES, dimensions)
            }
        )
        self.adapters = nn.ModuleDict(
            {
                name: nn.Linear(latent_dim, dimension)
                for name, dimension in zip(MODALITIES, dimensions)
            }
        )
        for adapter in self.adapters.values():
            nn.init.zeros_(adapter.weight)
            nn.init.zeros_(adapter.bias)

    def forward(
        self,
        masked_features: torch.Tensor,
        availability: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if masked_features.ndim != 3:
            raise ValueError("masked_features must have shape [L, B, sumD]")
        if masked_features.shape[-1] != sum(self.dimensions):
            raise ValueError("masked_features has the wrong feature dimension")
        if availability.ndim != 3 or availability.shape[-1] != 3:
            raise ValueError("availability must have shape [L, B, 3]")
        if masked_features.shape[:2] != availability.shape[:2]:
            raise ValueError(
                "masked_features and availability leading dimensions differ"
            )

        pattern_size = availability.sum(dim=-1)
        incomplete = (pattern_size == 1) | (pattern_size == 2)
        full = pattern_size == 3
        adapted_blocks = []
        latents = {}
        start = 0
        for index, (name, dimension) in enumerate(
            zip(MODALITIES, self.dimensions)
        ):
            block = masked_features[..., start : start + dimension]
            selected = incomplete & availability[..., index].bool()
            latent = masked_features.new_zeros(
                *masked_features.shape[:2], self.latent_dim
            )
            adapted = torch.zeros_like(block)
            adapted[full] = block[full]
            if bool(selected.any()):
                projected = self.projectors[name](block[selected])
                latent[selected] = projected
                adapted[selected] = block[selected] + self.adapters[name](projected)
            latents[name] = latent
            adapted_blocks.append(adapted)
            start += dimension
        return torch.cat(adapted_blocks, dim=-1), latents


class EMATeacherBank(nn.Module):
    """Frozen exponential-moving-average copies of student projectors."""

    def __init__(self, student_projectors: nn.ModuleDict) -> None:
        super().__init__()
        if not isinstance(student_projectors, nn.ModuleDict):
            raise ValueError("student_projectors must be an nn.ModuleDict")
        if tuple(student_projectors.keys()) != MODALITIES:
            raise ValueError("student_projectors must contain audio, text, and visual")
        self.dimensions = tuple(
            student_projectors[name].input_dim for name in MODALITIES
        )
        self.projectors = copy.deepcopy(student_projectors)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "EMATeacherBank":
        super().train(False)
        return self

    def forward(self, full_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        if full_features.ndim != 3:
            raise ValueError("full_features must have shape [L, B, sumD]")
        if full_features.shape[-1] != sum(self.dimensions):
            raise ValueError("full_features has the wrong feature dimension")
        latents = {}
        start = 0
        for name, dimension in zip(MODALITIES, self.dimensions):
            block = full_features[..., start : start + dimension]
            latents[name] = self.projectors[name](block)
            start += dimension
        return latents

    @torch.no_grad()
    def update_from(self, students: nn.ModuleDict, tau: float) -> None:
        if not _is_finite_real(tau) or tau < 0 or tau >= 1:
            raise ValueError("tau must be finite and satisfy 0 <= tau < 1")
        teacher_state = self.projectors.state_dict()
        student_state = students.state_dict()
        if tuple(teacher_state.keys()) != tuple(student_state.keys()):
            raise ValueError("student projector structure does not match teacher")
        for name, teacher_value in teacher_state.items():
            student_value = student_state[name]
            if torch.is_floating_point(teacher_value):
                teacher_value.mul_(tau).add_(student_value, alpha=1.0 - tau)
            else:
                teacher_value.copy_(student_value)
