"""Compact mask-aware SAM-style conversation backbone."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
from torch import Tensor, nn

from .attention import MaskedTrackPooling, SafeDirectedAttention


class ModalityTemporalEncoder(nn.Module):
    """Project one modality and model only its observed conversation tokens."""

    def __init__(
        self,
        input_dim: int,
        width: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.input_dim = input_dim
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, width)
        self.activation = nn.GELU()
        self.input_dropout = nn.Dropout(dropout)
        self.self_attention = SafeDirectedAttention(width, heads, dropout)
        self.ff_norm = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
            nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(width)

    def forward(self, features: Tensor, valid: Tensor) -> Tensor:
        if features.ndim != 3 or features.shape[-1] != self.input_dim:
            raise ValueError("modality feature shape does not match input_dim")
        if valid.shape != features.shape[:2]:
            raise ValueError("modality validity must have shape [L,B]")
        valid = valid.bool()
        mask = valid.to(features.dtype).unsqueeze(-1)
        observed_features = features * mask
        projected = self.input_dropout(
            self.activation(self.input_projection(self.input_norm(observed_features)))
        ) * mask
        attended, _ = self.self_attention(
            projected,
            projected,
            valid,
            valid,
        )
        output = self.output_norm(
            attended + self.feed_forward(self.ff_norm(attended))
        )
        return output * mask


class MaskAwareSAMModel(nn.Module):
    """Intra-modal temporal and directed cross-modal utterance model."""

    DIRECTIONS: Tuple[Tuple[int, int], ...] = (
        (0, 1),
        (1, 0),
        (0, 2),
        (2, 0),
        (1, 2),
        (2, 1),
    )

    def __init__(
        self,
        audio_dim: int,
        text_dim: int,
        visual_dim: int,
        *,
        width: int = 120,
        heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if width <= 0 or heads <= 0 or width % heads:
            raise ValueError("width must be positive and divisible by heads")
        self.width = width
        self.input_dims = (audio_dim, text_dim, visual_dim)
        self.encoders = nn.ModuleList(
            ModalityTemporalEncoder(dim, width, heads, dropout)
            for dim in self.input_dims
        )
        self.cross_attention = nn.ModuleDict(
            {
                self._direction_key(source, target): SafeDirectedAttention(
                    width,
                    heads,
                    dropout,
                )
                for source, target in self.DIRECTIONS
            }
        )
        self.pool = MaskedTrackPooling(width)
        self.output_block = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
            nn.Dropout(dropout),
            nn.LayerNorm(width),
        )
        self.regressor = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, 1),
        )

    @staticmethod
    def _direction_key(source: int, target: int) -> str:
        return "{}-{}".format(source, target)

    def _validate_inputs(
        self,
        features: Sequence[Tensor],
        availability: Tensor,
        umask: Tensor,
    ) -> Tensor:
        if len(features) != 3:
            raise ValueError("features must contain audio, text, and visual")
        first_shape = features[0].shape[:2]
        for value, expected_dim in zip(features, self.input_dims):
            if value.ndim != 3 or value.shape[:2] != first_shape:
                raise ValueError("all modality features must share [L,B]")
            if value.shape[-1] != expected_dim:
                raise ValueError("modality feature width is incorrect")
        if availability.shape != first_shape + (3,):
            raise ValueError("availability must have shape [L,B,3]")
        if umask.shape != (first_shape[1], first_shape[0]):
            raise ValueError("umask must have shape [B,L]")
        valid = umask.transpose(0, 1).bool()
        observed_count = availability.gt(0).sum(dim=-1)
        if bool((valid & observed_count.eq(0)).any()):
            raise ValueError("each valid utterance needs at least one observed modality")
        return valid

    def forward(
        self,
        features: Sequence[Tensor],
        availability: Tensor,
        umask: Tensor,
    ) -> Tuple[Tensor, Tensor, Dict[str, object]]:
        valid = self._validate_inputs(features, availability, umask)
        modality_valid = availability.gt(0) & valid.unsqueeze(-1)
        encoded: List[Tensor] = [
            encoder(value, modality_valid[..., index])
            for index, (encoder, value) in enumerate(
                zip(self.encoders, features)
            )
        ]
        tracks = list(encoded)
        track_valid = [modality_valid[..., index] for index in range(3)]
        attention_maps: Dict[str, Tensor] = {}
        for source, target in self.DIRECTIONS:
            key = self._direction_key(source, target)
            output, active, weights = self.cross_attention[key](
                encoded[source],
                encoded[target],
                modality_valid[..., source],
                modality_valid[..., target],
                return_weights=True,
            )
            tracks.append(output)
            track_valid.append(active)
            attention_maps[key] = weights
        stacked_valid = torch.stack(track_valid, dim=2)
        hidden, pooling = self.pool(
            torch.stack(tracks, dim=2),
            stacked_valid,
        )
        valid_float = valid.to(hidden.dtype).unsqueeze(-1)
        hidden = self.output_block(hidden) * valid_float
        prediction = self.regressor(hidden) * valid_float
        return prediction, hidden, {
            "cross_attention": attention_maps,
            "track_pooling": pooling,
            "track_valid": stacked_valid,
        }


__all__ = ["MaskAwareSAMModel", "ModalityTemporalEncoder"]
