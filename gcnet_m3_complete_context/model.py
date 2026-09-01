"""Exact complete-M3 inference model plus an optional zero-init context residual."""

from __future__ import annotations

from typing import Mapping, Sequence, Tuple

import torch
from torch import Tensor, nn


class ModalityProjector(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(latent_dim, latent_dim)
        self.output_norm = nn.LayerNorm(latent_dim)

    def forward(self, value: Tensor) -> Tensor:
        value = self.input_norm(value)
        value = torch.nn.functional.gelu(self.fc1(value))
        value = self.dropout(value)
        return self.output_norm(self.fc2(value))


class FusionHead(nn.Module):
    def __init__(self, latent_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(3 * latent_dim),
            nn.Linear(3 * latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, 1),
        )


class TemporalResidual(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.depthwise = nn.Conv1d(
            width,
            width,
            kernel_size=3,
            padding=1,
            groups=width,
        )
        self.pointwise = nn.Conv1d(width, width, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(width, width)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, hidden: Tensor, valid: Tensor) -> Tensor:
        mask = valid.to(hidden.dtype).unsqueeze(-1)
        value = self.norm(hidden) * mask
        value = value.permute(1, 2, 0)
        value = torch.nn.functional.gelu(self.depthwise(value))
        value = self.dropout(self.pointwise(value)).permute(2, 0, 1)
        return self.output(value) * mask


class CompleteM3Regressor(nn.Module):
    """Reproduce the locked M3 classifier and optionally add sequence context."""

    def __init__(
        self,
        dimensions: Tuple[int, int, int],
        *,
        latent_dim: int = 256,
        projector_dropout: float = 0.1,
        dropout: float = 0.2,
        temporal_context: bool = False,
    ) -> None:
        super().__init__()
        self.dimensions = tuple(int(value) for value in dimensions)
        self.projectors = nn.ModuleDict(
            {
                name: ModalityProjector(input_dim, latent_dim, projector_dropout)
                for name, input_dim in zip(
                    ("audio", "text", "visual"), self.dimensions
                )
            }
        )
        self.fusion = FusionHead(latent_dim, dropout)
        self.use_temporal_context = bool(temporal_context)
        if self.use_temporal_context:
            self.temporal_context = TemporalResidual(latent_dim, dropout)

    def load_baseline_state_dict(self, state: Mapping[str, Tensor]) -> None:
        expected = {
            key for key in self.state_dict() if not key.startswith("temporal_context.")
        }
        if set(state) != expected:
            missing = sorted(expected - set(state))
            extra = sorted(set(state) - expected)
            raise ValueError("baseline state mismatch: missing={} extra={}".format(missing, extra))
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys or set(result.missing_keys) != (
            set(self.state_dict()) - expected
        ):
            raise RuntimeError("baseline compatibility load failed")

    def forward(self, features: Sequence[Tensor], umask: Tensor) -> Tensor:
        if len(features) != 3:
            raise ValueError("features must contain audio, text, and visual")
        leading = features[0].shape[:2]
        if umask.shape != (leading[1], leading[0]):
            raise ValueError("umask must have shape [B,L]")
        projected = []
        for name, value, input_dim in zip(
            ("audio", "text", "visual"), features, self.dimensions
        ):
            if value.ndim != 3 or value.shape[:2] != leading or value.shape[-1] != input_dim:
                raise ValueError("feature shape mismatch")
            projected.append(self.projectors[name](value))
        valid = umask.transpose(0, 1).bool()
        combined = torch.cat(projected, dim=-1)
        hidden = self.fusion.network[:4](combined)
        if self.use_temporal_context:
            hidden = hidden + self.temporal_context(hidden, valid)
        prediction = self.fusion.network[4](hidden)
        return prediction * valid.to(prediction.dtype).unsqueeze(-1)


__all__ = ["CompleteM3Regressor"]
