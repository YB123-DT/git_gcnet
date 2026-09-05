"""Observed-set node construction and training-only M3 prediction."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from gcnet_modality_jepa.model import GraphModel


MODALITIES = ("audio", "text", "visual")


def _validate_observed_inputs(
    features: torch.Tensor,
    availability: torch.Tensor,
    umask: torch.Tensor,
    dimensions: Tuple[int, int, int],
) -> torch.Tensor:
    if features.ndim != 3 or features.shape[-1] != sum(dimensions):
        raise ValueError("features must have shape [L, B, sumD]")
    if availability.ndim != 3 or availability.shape[-1] != 3:
        raise ValueError("availability must have shape [L, B, 3]")
    if features.shape[:2] != availability.shape[:2]:
        raise ValueError("features and availability leading dimensions differ")
    length, batch = features.shape[:2]
    if umask.shape != (batch, length):
        raise ValueError("umask must have shape [B, L]")
    if not bool(((availability == 0) | (availability == 1)).all()):
        raise ValueError("availability must be binary")
    valid = umask.T.bool()
    if bool((availability[~valid] != 0).any()):
        raise ValueError("padding availability must be zero")
    if bool((availability[valid].sum(dim=-1) == 0).any()):
        raise ValueError("valid utterances require at least one observed modality")
    return valid


class AvailabilityConditionedLowRankReadout(nn.Module):
    """Low-rank output residual selected by explicit modality availability."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        rank: int = 8,
        route_type: str = "availability-low-rank",
    ) -> None:
        super().__init__()
        if int(input_dim) <= 0 or int(output_dim) <= 0:
            raise ValueError("input_dim and output_dim must be positive")
        if int(rank) <= 0:
            raise ValueError("rank must be positive")
        if route_type not in {
            "availability-low-rank",
            "shared-low-rank-parammatch",
        }:
            raise ValueError("unsupported route_type")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.rank = int(rank)
        self.route_type = route_type
        self.input_norm = nn.LayerNorm(
            self.input_dim,
            elementwise_affine=False,
        )
        self.basis = nn.Linear(self.input_dim, self.rank, bias=False)
        self.pattern_factor = nn.Parameter(
            torch.zeros(7, self.rank, self.output_dim)
        )
        self.pattern_bias = nn.Parameter(torch.zeros(7, self.output_dim))

    def forward(
        self,
        hidden: torch.Tensor,
        availability: torch.Tensor,
        umask: torch.Tensor,
    ) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.shape[-1] != self.input_dim:
            raise ValueError("hidden must have shape [L, B, input_dim]")
        length, batch = hidden.shape[:2]
        if availability.shape != (length, batch, 3):
            raise ValueError("availability must have shape [L, B, 3]")
        if umask.shape != (batch, length):
            raise ValueError("umask must have shape [B, L]")
        if not bool(((availability == 0) | (availability == 1)).all()):
            raise ValueError("availability must be binary")
        valid = umask.T.bool()
        if bool((availability[~valid] != 0).any()):
            raise ValueError("padding availability must be zero")
        if bool((availability[valid].sum(dim=-1) == 0).any()):
            raise ValueError("valid utterances require a nonempty pattern")

        residual = hidden.new_zeros(length, batch, self.output_dim)
        if not bool(valid.any()):
            return residual
        pattern_id = (
            availability[..., 0].long() * 4
            + availability[..., 1].long() * 2
            + availability[..., 2].long()
        )
        basis = self.basis(self.input_norm(hidden[valid]))
        if self.route_type == "availability-low-rank":
            row = pattern_id[valid] - 1
            factor = self.pattern_factor[row]
            bias = self.pattern_bias[row]
        else:
            factor = self.pattern_factor.mean(dim=0).expand(
                basis.shape[0], -1, -1
            )
            bias = self.pattern_bias.mean(dim=0).expand(basis.shape[0], -1)
        residual[valid] = torch.einsum("nr,nro->no", basis, factor) + bias
        return residual


class AvailabilityConditionedAffineReadout(nn.Module):
    """Pattern-specific affine residual before the unchanged shared head."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        if int(hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        self.hidden_dim = int(hidden_dim)
        self.input_norm = nn.LayerNorm(
            self.hidden_dim,
            elementwise_affine=False,
        )
        self.gamma = nn.Parameter(torch.zeros(7, self.hidden_dim))
        self.beta = nn.Parameter(torch.zeros(7, self.hidden_dim))

    def forward(
        self,
        hidden: torch.Tensor,
        availability: torch.Tensor,
        umask: torch.Tensor,
    ) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_dim:
            raise ValueError("hidden must have shape [L, B, hidden_dim]")
        length, batch = hidden.shape[:2]
        if availability.shape != (length, batch, 3):
            raise ValueError("availability must have shape [L, B, 3]")
        if umask.shape != (batch, length):
            raise ValueError("umask must have shape [B, L]")
        if not bool(((availability == 0) | (availability == 1)).all()):
            raise ValueError("availability must be binary")
        valid = umask.T.bool()
        if bool((availability[~valid] != 0).any()):
            raise ValueError("padding availability must be zero")
        if bool((availability[valid].sum(dim=-1) == 0).any()):
            raise ValueError("valid utterances require a nonempty pattern")

        residual = hidden.new_zeros(hidden.shape)
        if not bool(valid.any()):
            return residual
        pattern_id = (
            availability[..., 0].long() * 4
            + availability[..., 1].long() * 2
            + availability[..., 2].long()
        )
        row = pattern_id[valid] - 1
        normalized = self.input_norm(hidden[valid])
        residual[valid] = self.gamma[row] * normalized + self.beta[row]
        return residual


class ModalityProjector(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.input_norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(latent_dim, latent_dim)
        self.output_norm = nn.LayerNorm(latent_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.input_norm(value)
        value = F.gelu(self.fc1(value))
        value = self.dropout(value)
        return self.output_norm(self.fc2(value))


class EMATeacherProjectors(nn.ModuleDict):
    """Frozen EMA copies with state-dict keys matching the student bank."""

    def __init__(self, students: nn.ModuleDict) -> None:
        super().__init__(copy.deepcopy(dict(students.items())))
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.train(False)

    def train(self, mode: bool = True) -> "EMATeacherProjectors":
        super().train(False)
        return self

    @torch.no_grad()
    def update_from(self, students: nn.ModuleDict, tau: float) -> None:
        if not 0.0 <= float(tau) < 1.0:
            raise ValueError("tau must be in [0, 1)")
        for name in MODALITIES:
            for teacher_value, student_value in zip(
                self[name].parameters(), students[name].parameters()
            ):
                teacher_value.mul_(tau).add_(student_value, alpha=1.0 - tau)


class ObservedSetEncoder(nn.Module):
    """Build one fixed-width node from any non-empty observed modality set."""

    def __init__(
        self,
        dimensions: Tuple[int, int, int],
        latent_dim: int,
        dropout: float = 0.1,
        fusion_type: str = "mean",
    ) -> None:
        super().__init__()
        if len(dimensions) != 3 or any(int(value) <= 0 for value in dimensions):
            raise ValueError("dimensions must contain three positive integers")
        if fusion_type not in {"mean", "slot", "text-anchor-residual"}:
            raise ValueError(
                "fusion_type must be 'mean', 'slot', or "
                "'text-anchor-residual'"
            )
        self.dimensions = tuple(int(value) for value in dimensions)
        self.latent_dim = int(latent_dim)
        self.fusion_type = fusion_type
        self.projectors = nn.ModuleDict(
            {
                name: ModalityProjector(width, latent_dim, dropout)
                for name, width in zip(MODALITIES, self.dimensions)
            }
        )
        self.modality_embedding = nn.Embedding(3, latent_dim)
        self.pattern_embedding = nn.Embedding(8, latent_dim, padding_idx=0)
        fusion_input_dim = latent_dim if fusion_type == "mean" else 4 * latent_dim
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_input_dim),
            nn.Linear(fusion_input_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if fusion_type == "text-anchor-residual":
            residual_rank = min(64, self.latent_dim)
            with torch.random.fork_rng(devices=[]):
                self.anchor_residual = nn.Sequential(
                    nn.LayerNorm(3 * self.latent_dim),
                    nn.Linear(3 * self.latent_dim, residual_rank),
                    nn.GELU(),
                    nn.Linear(residual_rank, self.latent_dim),
                )
            nn.init.zeros_(self.anchor_residual[-1].weight)
            nn.init.zeros_(self.anchor_residual[-1].bias)
            self.anchor_residual_ratio = 0.25

    def _validate(
        self,
        features: torch.Tensor,
        availability: torch.Tensor,
        umask: torch.Tensor,
    ) -> torch.Tensor:
        return _validate_observed_inputs(
            features, availability, umask, self.dimensions
        )

    def forward(
        self,
        features: torch.Tensor,
        availability: torch.Tensor,
        umask: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        valid = self._validate(features, availability, umask)
        latent_shape = (*features.shape[:2], self.latent_dim)
        latents: Dict[str, torch.Tensor] = {}
        evidence = (
            features.new_zeros(latent_shape)
            if self.fusion_type == "mean"
            else None
        )
        slots = [] if self.fusion_type != "mean" else None
        start = 0
        for index, (name, width) in enumerate(zip(MODALITIES, self.dimensions)):
            block = features[..., start : start + width]
            selected = valid & availability[..., index].bool()
            latent = features.new_zeros(latent_shape)
            if bool(selected.any()):
                projected = self.projectors[name](block[selected])
                latent[selected] = projected
                if evidence is not None:
                    evidence[selected] += (
                        projected + self.modality_embedding.weight[index]
                    )
            if slots is not None:
                slot = latent.clone()
                slot[selected] += self.modality_embedding.weight[index]
                slots.append(slot)
            latents[name] = latent
            start += width

        count = availability.sum(dim=-1, keepdim=True).clamp_min(1).to(features.dtype)
        pattern_id = (
            availability[..., 0].long() * 4
            + availability[..., 1].long() * 2
            + availability[..., 2].long()
        )
        pattern = self.pattern_embedding(pattern_id)
        if self.fusion_type == "mean":
            assert evidence is not None
            fusion_input = evidence / count + pattern
        else:
            assert slots is not None
            fusion_input = torch.cat([*slots, pattern], dim=-1)
        node = features.new_zeros(latent_shape)
        node[valid] = self.fusion(fusion_input[valid])
        if self.fusion_type == "text-anchor-residual":
            assert slots is not None
            has_extra = (
                valid
                & availability[..., 1].bool()
                & (
                    availability[..., 0].bool()
                    | availability[..., 2].bool()
                )
            )
            if bool(has_extra.any()):
                zeros = torch.zeros_like(slots[0])
                text_pattern_id = torch.full_like(pattern_id, 2)
                anchor_input = torch.cat(
                    [
                        zeros,
                        slots[1],
                        zeros,
                        self.pattern_embedding(text_pattern_id),
                    ],
                    dim=-1,
                )
                anchor = self.fusion(anchor_input[has_extra])
                observed_set_value = node[has_extra]
                difference = observed_set_value - anchor
                raw_residual = self.anchor_residual(
                    torch.cat(
                        [anchor, difference, anchor * difference], dim=-1
                    )
                )
                residual_norm = raw_residual.norm(dim=-1, keepdim=True)
                residual_limit = (
                    self.anchor_residual_ratio
                    * anchor.norm(dim=-1, keepdim=True)
                )
                residual_scale = torch.minimum(
                    torch.ones_like(residual_norm),
                    residual_limit / residual_norm.clamp_min(1e-12),
                )
                node[has_extra] = anchor + raw_residual * residual_scale
        return node, latents


class ModalityTrackEncoder(nn.Module):
    """Keep observed modality evidence in three fixed-width graph tracks."""

    def __init__(
        self,
        dimensions: Tuple[int, int, int],
        latent_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        reference_encoder = ObservedSetEncoder(
            dimensions,
            latent_dim,
            dropout,
            fusion_type="slot",
        )
        self.dimensions = reference_encoder.dimensions
        self.latent_dim = reference_encoder.latent_dim
        self.projectors = reference_encoder.projectors
        self.modality_embedding = reference_encoder.modality_embedding

    def forward(
        self,
        features: torch.Tensor,
        availability: torch.Tensor,
        umask: torch.Tensor,
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        valid = _validate_observed_inputs(
            features, availability, umask, self.dimensions
        )
        latent_shape = (*features.shape[:2], self.latent_dim)
        tracks: Dict[str, torch.Tensor] = {}
        latents: Dict[str, torch.Tensor] = {}
        start = 0
        for index, (name, width) in enumerate(zip(MODALITIES, self.dimensions)):
            block = features[..., start : start + width]
            selected = valid & availability[..., index].bool()
            latent = features.new_zeros(latent_shape)
            if bool(selected.any()):
                latent[selected] = self.projectors[name](block[selected])
            track = latent.clone()
            track[selected] += self.modality_embedding.weight[index]
            latents[name] = latent
            tracks[name] = track
            start += width
        return tracks, latents


class PostGraphTrackFusion(nn.Module):
    """Fuse availability-masked modality tracks after shared graph reasoning."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.pattern_embedding = nn.Embedding(
            8, self.hidden_dim, padding_idx=0
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(4 * self.hidden_dim),
            nn.Linear(4 * self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        track_hidden: Mapping[str, torch.Tensor],
        availability: torch.Tensor,
        umask: torch.Tensor,
    ) -> torch.Tensor:
        if set(track_hidden) != set(MODALITIES):
            raise ValueError("track_hidden must contain audio, text, and visual")
        reference = track_hidden[MODALITIES[0]]
        if reference.ndim != 3 or reference.shape[-1] != self.hidden_dim:
            raise ValueError("each track must have shape [L, B, hidden_dim]")
        length, batch = reference.shape[:2]
        if availability.shape != (length, batch, 3):
            raise ValueError("availability must have shape [L, B, 3]")
        if umask.shape != (batch, length):
            raise ValueError("umask must have shape [B, L]")
        if not bool(((availability == 0) | (availability == 1)).all()):
            raise ValueError("availability must be binary")
        for value in track_hidden.values():
            if value.shape != reference.shape:
                raise ValueError("all modality tracks must share one shape")

        valid = umask.T.bool()
        if bool((availability[~valid] != 0).any()):
            raise ValueError("padding availability must be zero")
        slots = [
            track_hidden[name] * availability[..., index : index + 1]
            for index, name in enumerate(MODALITIES)
        ]
        pattern_id = (
            availability[..., 0].long() * 4
            + availability[..., 1].long() * 2
            + availability[..., 2].long()
        )
        fusion_input = torch.cat(
            [*slots, self.pattern_embedding(pattern_id)], dim=-1
        )
        output = reference.new_zeros(reference.shape)
        output[valid] = self.fusion(fusion_input[valid])
        return output


class PatternConditionedInteractionResidual(nn.Module):
    """Refine a Slot node using observed-only unary and pair evidence."""

    PAIRS = ((0, 1), (0, 2), (1, 2))

    def __init__(
        self,
        latent_dim: int,
        pair_embedding_dim: int = 32,
        pair_rank: int = 64,
        residual_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        values = (
            latent_dim,
            pair_embedding_dim,
            pair_rank,
            residual_hidden_dim,
        )
        if any(int(value) <= 0 for value in values):
            raise ValueError("PCIR dimensions must be positive")
        self.latent_dim = int(latent_dim)
        self.scale_shift = nn.Embedding(
            8, 3 * 2 * self.latent_dim, padding_idx=0
        )
        self.pair_embedding = nn.Embedding(3, int(pair_embedding_dim))
        pair_input_dim = 4 * self.latent_dim + int(pair_embedding_dim)
        self.pair_mlp = nn.Sequential(
            nn.LayerNorm(pair_input_dim),
            nn.Linear(pair_input_dim, int(pair_rank)),
            nn.GELU(),
            nn.Linear(int(pair_rank), self.latent_dim),
        )
        self.pattern_embedding = nn.Embedding(
            8, int(pair_embedding_dim), padding_idx=0
        )
        residual_input_dim = 2 * self.latent_dim + int(pair_embedding_dim)
        self.residual_mlp = nn.Sequential(
            nn.LayerNorm(residual_input_dim),
            nn.Linear(residual_input_dim, int(residual_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(residual_hidden_dim), self.latent_dim),
        )
        nn.init.zeros_(self.scale_shift.weight)
        nn.init.zeros_(self.residual_mlp[-1].weight)
        nn.init.zeros_(self.residual_mlp[-1].bias)

    @staticmethod
    def active_pair_mask(availability: torch.Tensor) -> torch.Tensor:
        if availability.ndim != 3 or availability.shape[-1] != 3:
            raise ValueError("availability must have shape [L, B, 3]")
        observed = availability.bool()
        return torch.stack(
            (
                observed[..., 0] & observed[..., 1],
                observed[..., 0] & observed[..., 2],
                observed[..., 1] & observed[..., 2],
            ),
            dim=-1,
        )

    def forward(
        self,
        latents: Mapping[str, torch.Tensor],
        availability: torch.Tensor,
        umask: torch.Tensor,
    ) -> torch.Tensor:
        if set(latents) != set(MODALITIES):
            raise ValueError("latents must contain audio, text, and visual")
        reference = latents[MODALITIES[0]]
        if reference.ndim != 3 or reference.shape[-1] != self.latent_dim:
            raise ValueError("each latent must have shape [L, B, latent_dim]")
        length, batch = reference.shape[:2]
        if availability.shape != (length, batch, 3):
            raise ValueError("availability must have shape [L, B, 3]")
        if umask.shape != (batch, length):
            raise ValueError("umask must have shape [B, L]")
        for value in latents.values():
            if value.shape != reference.shape:
                raise ValueError("all modality latents must share one shape")
        availability = availability.to(device=reference.device)
        if not bool(((availability == 0) | (availability == 1)).all()):
            raise ValueError("availability must be binary")
        valid = umask.to(device=reference.device).T.bool()
        if bool((availability[~valid] != 0).any()):
            raise ValueError("padding availability must be zero")
        if bool((availability[valid].sum(dim=-1) == 0).any()):
            raise ValueError("valid utterances require an observed modality")

        pattern_id = (
            availability[..., 0].long() * 4
            + availability[..., 1].long() * 2
            + availability[..., 2].long()
        )
        parameters = self.scale_shift(pattern_id).reshape(
            length, batch, 3, 2, self.latent_dim
        )
        scale = parameters[..., 0, :]
        shift = parameters[..., 1, :]
        stacked = torch.stack([latents[name] for name in MODALITIES], dim=2)
        observed = availability.unsqueeze(-1).to(reference.dtype)
        corrected = ((1.0 + scale) * stacked + shift) * observed
        observed_count = observed.sum(dim=2).clamp_min(1.0)
        observed_summary = corrected.sum(dim=2) / observed_count

        pair_mask = self.active_pair_mask(availability)
        pair_values = []
        for pair_index, (left_index, right_index) in enumerate(self.PAIRS):
            left = corrected[..., left_index, :]
            right = corrected[..., right_index, :]
            pair_identity = self.pair_embedding.weight[pair_index].view(
                1, 1, -1
            ).expand(length, batch, -1)
            pair_input = torch.cat(
                (left, right, left * right, (left - right).abs(), pair_identity),
                dim=-1,
            )
            pair_value = self.pair_mlp(pair_input)
            pair_values.append(
                pair_value
                * pair_mask[..., pair_index : pair_index + 1].to(
                    reference.dtype
                )
            )
        pair_stack = torch.stack(pair_values, dim=2)
        pair_count = pair_mask.sum(dim=-1, keepdim=True).clamp_min(1)
        pair_summary = pair_stack.sum(dim=2) / pair_count.to(reference.dtype)
        residual_input = torch.cat(
            (
                observed_summary,
                pair_summary,
                self.pattern_embedding(pattern_id),
            ),
            dim=-1,
        )
        residual = self.residual_mlp(residual_input)
        return residual * valid.unsqueeze(-1).to(reference.dtype)


class RawResidualObservedEncoder(nn.Module):
    """Preserve observed raw blocks while coupling Student latents by residuals."""

    def __init__(
        self,
        dimensions: Tuple[int, int, int],
        latent_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if len(dimensions) != 3 or any(int(value) <= 0 for value in dimensions):
            raise ValueError("dimensions must contain three positive integers")
        self.dimensions = tuple(int(value) for value in dimensions)
        self.latent_dim = int(latent_dim)
        self.projectors = nn.ModuleDict(
            {
                name: ModalityProjector(width, latent_dim, dropout)
                for name, width in zip(MODALITIES, self.dimensions)
            }
        )
        self.adapters = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(latent_dim),
                    nn.Linear(latent_dim, width),
                )
                for name, width in zip(MODALITIES, self.dimensions)
            }
        )
        for adapter in self.adapters.values():
            nn.init.zeros_(adapter[-1].weight)
            nn.init.zeros_(adapter[-1].bias)

    def forward(
        self,
        features: torch.Tensor,
        availability: torch.Tensor,
        umask: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        valid = _validate_observed_inputs(
            features, availability, umask, self.dimensions
        )
        latent_shape = (*features.shape[:2], self.latent_dim)
        output = features.new_zeros(features.shape)
        latents: Dict[str, torch.Tensor] = {}
        start = 0
        for index, (name, width) in enumerate(zip(MODALITIES, self.dimensions)):
            stop = start + width
            block = features[..., start:stop]
            selected = valid & availability[..., index].bool()
            latent = features.new_zeros(latent_shape)
            if bool(selected.any()):
                projected = self.projectors[name](block[selected])
                latent[selected] = projected
                output_block = output[..., start:stop]
                output_block[selected] = (
                    block[selected] + self.adapters[name](projected)
                )
            latents[name] = latent
            start = stop
        return output, latents


class LocalContextResidualFusion(nn.Module):
    """Fuse utterance-local Student slots into a GCNet hidden residual."""

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        input_dim = 3 * int(latent_dim) + 3
        self.latent_dim = int(latent_dim)
        self.context_dim = int(context_dim)
        self.fusion = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, context_dim),
        )
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

    def forward(
        self,
        latents: Mapping[str, torch.Tensor],
        availability: torch.Tensor,
        umask: torch.Tensor,
    ) -> torch.Tensor:
        if set(latents) != set(MODALITIES):
            raise ValueError("latents must contain audio, text, and visual")
        reference = latents[MODALITIES[0]]
        if reference.ndim != 3 or reference.shape[-1] != self.latent_dim:
            raise ValueError("Student latents must have shape [L, B, latent_dim]")
        length, batch = reference.shape[:2]
        expected_shape = (length, batch, self.latent_dim)
        if any(latents[name].shape != expected_shape for name in MODALITIES):
            raise ValueError("all Student latents must have the same shape")
        if availability.shape != (length, batch, 3):
            raise ValueError("availability must have shape [L, B, 3]")
        if umask.shape != (batch, length):
            raise ValueError("umask must have shape [B, L]")

        availability_value = availability.to(dtype=reference.dtype)
        slots = [
            latents[name] * availability_value[..., index : index + 1]
            for index, name in enumerate(MODALITIES)
        ]
        fusion_input = torch.cat([*slots, availability_value], dim=-1)
        residual = reference.new_zeros(length, batch, self.context_dim)
        valid = umask.T.bool()
        if bool(valid.any()):
            residual[valid] = self.fusion(fusion_input[valid])
        return residual


class TargetPrivateExpertResidual(nn.Module):
    """Low-rank target-owned capacity added before target-specific heads."""

    def __init__(self, latent_dim: int, rank: int) -> None:
        super().__init__()
        if int(rank) <= 0:
            raise ValueError("rank must be positive")
        self.down = nn.Linear(latent_dim, int(rank), bias=False)
        self.up = nn.Linear(int(rank), latent_dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.up(F.gelu(self.down(value)))


class DualGateTopKMMoE(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_experts: int,
        top_k: int,
        dropout: float,
        variant: str = "dual-gate",
        target_private_rank: int = 0,
    ) -> None:
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be between one and num_experts")
        if variant not in {"dual-gate", "paper-faithful"}:
            raise ValueError("variant must be 'dual-gate' or 'paper-faithful'")
        if int(target_private_rank) < 0:
            raise ValueError("target_private_rank cannot be negative")
        self.top_k = int(top_k)
        self.variant = variant
        self.num_experts = int(num_experts)
        self.target_private_rank = int(target_private_rank)
        self.source_embedding = nn.Embedding(3, latent_dim)
        self.target_embedding = nn.Embedding(3, latent_dim)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(latent_dim, latent_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(latent_dim, latent_dim),
                )
                for _ in range(num_experts)
            ]
        )
        self.reg_gate = nn.Linear(latent_dim, num_experts)
        self.cl_gate = nn.Linear(latent_dim, num_experts)
        self.reg_heads = nn.ModuleList(
            [nn.Linear(latent_dim, latent_dim) for _ in MODALITIES]
        )
        self.cl_heads = nn.ModuleList(
            [nn.Linear(latent_dim, latent_dim) for _ in MODALITIES]
        )
        if self.target_private_rank:
            self.target_private_experts = nn.ModuleList(
                [
                    TargetPrivateExpertResidual(
                        latent_dim, self.target_private_rank
                    )
                    for _ in MODALITIES
                ]
            )
        if self.variant == "paper-faithful":
            self.reg_task_embedding = nn.Parameter(torch.empty(latent_dim))
            self.cl_task_embedding = nn.Parameter(torch.empty(latent_dim))
            self.reg_norm = nn.LayerNorm(latent_dim)
            self.cl_norm = nn.LayerNorm(latent_dim)
            nn.init.kaiming_uniform_(
                self.reg_task_embedding.view(1, -1), nonlinearity="relu"
            )
            nn.init.kaiming_uniform_(
                self.cl_task_embedding.view(1, -1), nonlinearity="relu"
            )
        self.register_buffer(
            "_routing_selection_count",
            torch.zeros(2, num_experts, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "_routing_probability_mass",
            torch.zeros(2, num_experts, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "_routing_token_count",
            torch.zeros(2, dtype=torch.float64),
            persistent=False,
        )

    def _route(
        self, value: torch.Tensor, gate: nn.Linear, branch_index: int
    ) -> torch.Tensor:
        logits = gate(value)
        if self.variant == "paper-faithful":
            top_weights, top_indices = logits.softmax(dim=-1).topk(
                self.top_k, dim=-1
            )
        else:
            top_values, top_indices = logits.topk(self.top_k, dim=-1)
            top_weights = top_values.softmax(dim=-1)
        route = torch.zeros_like(logits)
        route = route.scatter(-1, top_indices, top_weights)
        with torch.no_grad():
            selected = torch.zeros_like(logits).scatter(
                -1, top_indices, torch.ones_like(top_weights)
            )
            self._routing_selection_count[branch_index].add_(
                selected.sum(dim=0).to(torch.float64)
            )
            self._routing_probability_mass[branch_index].add_(
                route.sum(dim=0).to(torch.float64)
            )
            self._routing_token_count[branch_index].add_(float(value.shape[0]))
        return route

    @torch.no_grad()
    def reset_routing_statistics(self) -> None:
        self._routing_selection_count.zero_()
        self._routing_probability_mass.zero_()
        self._routing_token_count.zero_()

    @torch.no_grad()
    def routing_statistics(self) -> dict[str, torch.Tensor]:
        mass_total = self._routing_probability_mass.sum(dim=-1, keepdim=True)
        distribution = self._routing_probability_mass / mass_total.clamp_min(1e-12)
        entropy = -(
            distribution * distribution.clamp_min(1e-12).log()
        ).sum(dim=-1)
        return {
            "selection_count": self._routing_selection_count.clone(),
            "probability_mass": self._routing_probability_mass.clone(),
            "token_count": self._routing_token_count.clone(),
            "usage": self._routing_selection_count
            / self._routing_selection_count.sum(dim=-1, keepdim=True).clamp_min(1.0),
            "entropy": entropy,
        }

    def _paper_branch(
        self,
        value: torch.Tensor,
        gate: nn.Linear,
        norm: nn.LayerNorm,
        branch_index: int,
    ) -> torch.Tensor:
        experts = torch.stack([expert(value) for expert in self.experts], dim=1)
        hidden = torch.einsum(
            "ne,ned->nd",
            self._route(value, gate, branch_index=branch_index),
            experts,
        )
        return value + F.gelu(norm(hidden))

    def forward(
        self, value: torch.Tensor, source_index: int, target_index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        conditioned = (
            value
            + self.source_embedding.weight[source_index]
            + self.target_embedding.weight[target_index]
        )
        if self.variant == "paper-faithful":
            reg_hidden = self._paper_branch(
                conditioned + self.reg_task_embedding,
                self.reg_gate,
                self.reg_norm,
                branch_index=0,
            )
            cl_hidden = self._paper_branch(
                conditioned + self.cl_task_embedding,
                self.cl_gate,
                self.cl_norm,
                branch_index=1,
            )
        else:
            experts = torch.stack(
                [expert(conditioned) for expert in self.experts], dim=1
            )
            reg_hidden = torch.einsum(
                "ne,ned->nd",
                self._route(conditioned, self.reg_gate, branch_index=0),
                experts,
            )
            cl_hidden = torch.einsum(
                "ne,ned->nd",
                self._route(conditioned, self.cl_gate, branch_index=1),
                experts,
            )
        if self.target_private_rank:
            private_residual = self.target_private_experts[target_index](
                conditioned
            )
            reg_hidden = reg_hidden + private_residual
            cl_hidden = cl_hidden + private_residual
        return self.reg_heads[target_index](reg_hidden), self.cl_heads[target_index](cl_hidden)


@dataclass(frozen=True)
class MissingM3Predictions:
    reg_predictions: torch.Tensor
    cl_predictions: torch.Tensor
    target_mask: torch.Tensor
    source_counts: torch.Tensor


class MissingLatentResidualFusion(nn.Module):
    """Map predicted missing target latents into the emotion hidden space."""

    def __init__(self, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.target_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(latent_dim),
                    nn.Linear(latent_dim, hidden_dim),
                )
                for _ in MODALITIES
            ]
        )
        for projection in self.target_projections:
            nn.init.zeros_(projection[-1].weight)
            nn.init.zeros_(projection[-1].bias)

    def forward(
        self,
        reg_predictions: torch.Tensor,
        target_mask: torch.Tensor,
        umask: torch.Tensor,
    ) -> torch.Tensor:
        if reg_predictions.ndim != 4 or reg_predictions.shape[2:] != (
            len(MODALITIES),
            self.latent_dim,
        ):
            raise ValueError(
                "reg_predictions must have shape [L, B, 3, latent_dim]"
            )
        length, batch = reg_predictions.shape[:2]
        if target_mask.shape != (length, batch, len(MODALITIES)):
            raise ValueError("target_mask must have shape [L, B, 3]")
        if umask.shape != (batch, length):
            raise ValueError("umask must have shape [B, L]")

        projected = torch.stack(
            [
                projection(reg_predictions[:, :, target_index])
                for target_index, projection in enumerate(self.target_projections)
            ],
            dim=2,
        )
        valid_targets = target_mask.bool() & umask.T.bool().unsqueeze(-1)
        weights = valid_targets.to(dtype=reg_predictions.dtype).unsqueeze(-1)
        residual = (torch.tanh(projected) * weights).sum(dim=2)
        divisor = valid_targets.sum(dim=2).clamp_min(1).to(
            dtype=reg_predictions.dtype
        )
        return residual / divisor.unsqueeze(-1)


class ContextualM3Predictor(nn.Module):
    """Vectorized six-direction M3 predictor conditioned on GCNet context."""

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        num_experts: int = 4,
        top_k: int = 2,
        dropout: float = 0.1,
        mmoe_variant: str = "dual-gate",
        target_private_rank: int = 0,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.context_projection = nn.Linear(context_dim, latent_dim)
        self.input_norm = nn.LayerNorm(latent_dim)
        self.mmoe = DualGateTopKMMoE(
            latent_dim,
            num_experts,
            top_k,
            dropout,
            variant=mmoe_variant,
            target_private_rank=target_private_rank,
        )

    def direction_forward(
        self,
        source: torch.Tensor,
        context: torch.Tensor,
        source_index: int,
        target_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        conditioned = self.input_norm(source + self.context_projection(context))
        return self.mmoe(conditioned, source_index, target_index)

    def forward(
        self,
        latents: Mapping[str, torch.Tensor],
        hidden: torch.Tensor,
        availability: torch.Tensor,
        umask: torch.Tensor,
    ) -> MissingM3Predictions:
        if set(latents) != set(MODALITIES):
            raise ValueError("latents must contain audio, text, and visual")
        length, batch, _ = hidden.shape
        valid = umask.T.bool().reshape(-1)
        flat_availability = availability.reshape(-1, 3).bool()
        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        flat_latents = {
            name: value.reshape(-1, self.latent_dim) for name, value in latents.items()
        }
        reg_targets = []
        cl_targets = []
        target_masks = []
        source_counts = []
        for target_index in range(3):
            target_mask = valid & ~flat_availability[:, target_index]
            reg_sum = hidden.new_zeros(length * batch, self.latent_dim)
            cl_sum = hidden.new_zeros(length * batch, self.latent_dim)
            counts = torch.zeros(length * batch, dtype=torch.long, device=hidden.device)
            for source_index, source_name in enumerate(MODALITIES):
                if source_index == target_index:
                    continue
                selected = target_mask & flat_availability[:, source_index]
                indices = torch.nonzero(selected, as_tuple=False).flatten()
                if indices.numel() == 0:
                    continue
                reg, cl = self.direction_forward(
                    flat_latents[source_name][indices],
                    flat_hidden[indices],
                    source_index,
                    target_index,
                )
                reg_sum = reg_sum.index_add(0, indices, reg)
                cl_sum = cl_sum.index_add(0, indices, cl)
                counts = counts.index_add(0, indices, torch.ones_like(indices))
            divisor = counts.clamp_min(1).to(hidden.dtype).unsqueeze(-1)
            reg_targets.append(reg_sum / divisor)
            cl_targets.append(cl_sum / divisor)
            target_masks.append(target_mask & (counts > 0))
            source_counts.append(counts)
        return MissingM3Predictions(
            reg_predictions=torch.stack(reg_targets, dim=1).reshape(
                length, batch, 3, self.latent_dim
            ),
            cl_predictions=torch.stack(cl_targets, dim=1).reshape(
                length, batch, 3, self.latent_dim
            ),
            target_mask=torch.stack(target_masks, dim=1).reshape(length, batch, 3),
            source_counts=torch.stack(source_counts, dim=1).reshape(length, batch, 3),
        )


class MissingM3GraphModel(GraphModel):
    """Observed-set node encoder followed by the original GCNet graph core."""

    def __init__(
        self,
        base_model,
        adim,
        tdim,
        vdim,
        D_e,
        graph_hidden_size,
        n_speakers,
        window_past,
        window_future,
        n_classes,
        dropout=0.5,
        time_attn=True,
        no_cuda=False,
        latent_dim=256,
        num_experts=4,
        top_k=2,
        projector_dropout=0.1,
        predictor_dropout=0.1,
        fusion_type="mean",
        local_context_residual=False,
        local_fusion_hidden_dim=256,
        local_fusion_dropout=0.2,
        graph_branch_mode="both",
        mmoe_variant="dual-gate",
        target_private_rank=0,
        classification_completion=False,
        representation_type="slot",
        node_interaction_residual=False,
        readout_type="shared",
        readout_rank=8,
        recurrent_padding_mode="legacy",
        postgraph_sequence_mode="independent",
        graph_message_calibration="none",
        graph_second_layer="graphconv",
        postgraph_bilstm_ablation="none",
    ) -> None:
        if readout_type not in {
            "shared",
            "availability-low-rank",
            "shared-low-rank-parammatch",
            "availability-affine",
        }:
            raise ValueError("unsupported readout_type")
        if int(readout_rank) <= 0:
            raise ValueError("readout_rank must be positive")
        if representation_type not in {"slot", "track"}:
            raise ValueError("representation_type must be 'slot' or 'track'")
        if representation_type == "track" and fusion_type != "slot":
            raise ValueError("track representation requires fusion_type='slot'")
        if representation_type == "track" and local_context_residual:
            raise ValueError(
                "track representation cannot use local_context_residual"
            )
        if representation_type == "track" and classification_completion:
            raise ValueError(
                "track representation cannot use classification_completion"
            )
        if local_context_residual and fusion_type != "slot":
            raise ValueError("local_context_residual requires fusion_type='slot'")
        if node_interaction_residual and fusion_type != "slot":
            raise ValueError(
                "node_interaction_residual requires fusion_type='slot'"
            )
        if node_interaction_residual and representation_type == "track":
            raise ValueError(
                "node_interaction_residual cannot use track representation"
            )
        if node_interaction_residual and local_context_residual:
            raise ValueError(
                "node_interaction_residual cannot use local_context_residual"
            )
        if node_interaction_residual and classification_completion:
            raise ValueError(
                "node_interaction_residual cannot use classification_completion"
            )
        super().__init__(
            base_model,
            adim,
            tdim,
            vdim,
            D_e,
            graph_hidden_size,
            n_speakers,
            window_past,
            window_future,
            n_classes,
            dropout,
            time_attn,
            no_cuda,
            enable_reconstruction=False,
            graph_branch_mode=graph_branch_mode,
            recurrent_padding_mode=recurrent_padding_mode,
            postgraph_sequence_mode=postgraph_sequence_mode,
            graph_message_calibration=graph_message_calibration,
            graph_second_layer=graph_second_layer,
            postgraph_bilstm_ablation=postgraph_bilstm_ablation,
        )
        self.dimensions = (adim, tdim, vdim)
        self.latent_dim = int(latent_dim)
        self.representation_type = representation_type
        if representation_type == "track":
            self.observed_set = ModalityTrackEncoder(
                self.dimensions,
                latent_dim,
                projector_dropout,
            )
        elif fusion_type == "raw-residual":
            self.observed_set = RawResidualObservedEncoder(
                self.dimensions,
                latent_dim,
                projector_dropout,
            )
        else:
            self.observed_set = ObservedSetEncoder(
                self.dimensions,
                latent_dim,
                projector_dropout,
                fusion_type=fusion_type,
            )
        self.teacher = EMATeacherProjectors(self.observed_set.projectors)
        if fusion_type != "raw-residual":
            if base_model == "LSTM":
                self.lstm = nn.LSTM(
                    input_size=latent_dim,
                    hidden_size=D_e,
                    num_layers=2,
                    bidirectional=True,
                    dropout=dropout,
                )
            elif base_model == "GRU":
                self.gru = nn.GRU(
                    input_size=latent_dim,
                    hidden_size=D_e,
                    num_layers=2,
                    bidirectional=True,
                    dropout=dropout,
                )
        hidden_dim = 2 * D_e + graph_hidden_size
        self.missing_predictor = ContextualM3Predictor(
            latent_dim,
            hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
            dropout=predictor_dropout,
            mmoe_variant=mmoe_variant,
            target_private_rank=target_private_rank,
        )
        if representation_type == "track":
            self.track_fusion = PostGraphTrackFusion(
                hidden_dim, projector_dropout
            )
        self.classification_completion = bool(classification_completion)
        if self.classification_completion:
            self.missing_latent_fusion = MissingLatentResidualFusion(
                latent_dim, hidden_dim
            )
        self.ema_step = 0
        self.local_context_residual = bool(local_context_residual)
        if self.local_context_residual:
            self.local_context_fusion = LocalContextResidualFusion(
                latent_dim,
                hidden_dim,
                hidden_dim=local_fusion_hidden_dim,
                dropout=local_fusion_dropout,
            )
        self.node_interaction_residual = bool(node_interaction_residual)
        if self.node_interaction_residual:
            self.node_interaction = PatternConditionedInteractionResidual(
                latent_dim
            )
        self.readout_type = readout_type
        self.readout_rank = int(readout_rank)
        if self.readout_type in {
            "availability-low-rank",
            "shared-low-rank-parammatch",
        }:
            with torch.random.fork_rng(devices=[]):
                self.conditioned_readout = AvailabilityConditionedLowRankReadout(
                    hidden_dim,
                    n_classes,
                    rank=self.readout_rank,
                    route_type=self.readout_type,
                )
        elif self.readout_type == "availability-affine":
            self.affine_readout = AvailabilityConditionedAffineReadout(
                hidden_dim
            )

    @staticmethod
    def _feature_tensor(inputfeats) -> torch.Tensor:
        if torch.is_tensor(inputfeats):
            return inputfeats
        if isinstance(inputfeats, (tuple, list)) and len(inputfeats) == 1:
            return inputfeats[0]
        raise ValueError("inputfeats must be a tensor or one-element list")

    def forward(
        self,
        inputfeats,
        availability,
        qmask,
        umask,
        seq_lengths,
        predict_missing=False,
    ):
        features = self._feature_tensor(inputfeats)
        encoded, latents = self.observed_set(features, availability, umask)
        if self.node_interaction_residual:
            encoded = encoded + self.node_interaction(
                latents, availability, umask
            )
        if self.representation_type == "track":
            track_hidden = {
                name: self.encode_hidden(
                    [encoded[name]], qmask, umask, seq_lengths
                )
                for name in MODALITIES
            }
            graph_hidden = self.track_fusion(
                track_hidden, availability, umask
            )
        else:
            graph_hidden = self.encode_hidden(
                [encoded], qmask, umask, seq_lengths
            )
        internal_predictions = (
            self.missing_predictor(latents, graph_hidden, availability, umask)
            if predict_missing or self.classification_completion
            else None
        )
        classification_hidden = graph_hidden
        if self.local_context_residual:
            classification_hidden = graph_hidden + self.local_context_fusion(
                latents, availability, umask
            )
        if self.classification_completion:
            classification_hidden = classification_hidden + self.missing_latent_fusion(
                internal_predictions.reg_predictions,
                internal_predictions.target_mask,
                umask,
            )
        readout_hidden = classification_hidden
        if self.readout_type == "availability-affine":
            readout_hidden = readout_hidden + self.affine_readout(
                classification_hidden,
                availability,
                umask,
            )
        logits = self.smax_fc(readout_hidden)
        if self.readout_type in {
            "availability-low-rank",
            "shared-low-rank-parammatch",
        }:
            logits = logits + self.conditioned_readout(
                classification_hidden,
                availability,
                umask,
            )
        returned_predictions = internal_predictions if predict_missing else None
        return logits, classification_hidden, latents, returned_predictions

    @torch.no_grad()
    def encode_teacher_targets(self, complete_features) -> Dict[str, torch.Tensor]:
        features = self._feature_tensor(complete_features)
        parts = torch.split(features, self.dimensions, dim=-1)
        return {
            name: self.teacher[name](part)
            for name, part in zip(MODALITIES, parts)
        }

    @torch.no_grad()
    def update_teacher(self, tau: float) -> None:
        self.teacher.update_from(self.observed_set.projectors, tau)
        self.ema_step += 1

    def train(self, mode: bool = True) -> "MissingM3GraphModel":
        super().train(mode)
        self.teacher.train(False)
        return self
