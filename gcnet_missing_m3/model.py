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
        if fusion_type not in {"mean", "slot"}:
            raise ValueError("fusion_type must be 'mean' or 'slot'")
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
        slots = [] if self.fusion_type == "slot" else None
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
        return node, latents


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


class DualGateTopKMMoE(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_experts: int,
        top_k: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be between one and num_experts")
        self.top_k = int(top_k)
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

    def _route(self, value: torch.Tensor, gate: nn.Linear) -> torch.Tensor:
        logits = gate(value)
        top_values, top_indices = logits.topk(self.top_k, dim=-1)
        top_weights = top_values.softmax(dim=-1)
        route = torch.zeros_like(logits)
        return route.scatter(-1, top_indices, top_weights)

    def forward(
        self, value: torch.Tensor, source_index: int, target_index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        conditioned = (
            value
            + self.source_embedding.weight[source_index]
            + self.target_embedding.weight[target_index]
        )
        experts = torch.stack([expert(conditioned) for expert in self.experts], dim=1)
        reg_hidden = torch.einsum(
            "ne,ned->nd", self._route(conditioned, self.reg_gate), experts
        )
        cl_hidden = torch.einsum(
            "ne,ned->nd", self._route(conditioned, self.cl_gate), experts
        )
        return self.reg_heads[target_index](reg_hidden), self.cl_heads[target_index](cl_hidden)


@dataclass(frozen=True)
class MissingM3Predictions:
    reg_predictions: torch.Tensor
    cl_predictions: torch.Tensor
    target_mask: torch.Tensor
    source_counts: torch.Tensor


class ContextualM3Predictor(nn.Module):
    """Vectorized six-direction M3 predictor conditioned on GCNet context."""

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        num_experts: int = 4,
        top_k: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.context_projection = nn.Linear(context_dim, latent_dim)
        self.input_norm = nn.LayerNorm(latent_dim)
        self.mmoe = DualGateTopKMMoE(
            latent_dim, num_experts, top_k, dropout
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
    ) -> None:
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
        )
        self.dimensions = (adim, tdim, vdim)
        self.latent_dim = int(latent_dim)
        if fusion_type == "raw-residual":
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
        )
        self.ema_step = 0

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
        node, latents = self.observed_set(features, availability, umask)
        hidden = self.encode_hidden([node], qmask, umask, seq_lengths)
        logits = self.smax_fc(hidden)
        predictions = (
            self.missing_predictor(latents, hidden, availability, umask)
            if predict_missing
            else None
        )
        return logits, hidden, latents, predictions

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
