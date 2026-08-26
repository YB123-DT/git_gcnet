"""Projection, residual-adapter, and EMA-teacher modules for PLCI-JEPA."""

import copy
import math
from dataclasses import dataclass
from numbers import Real
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F


MODALITIES = ("audio", "text", "visual")


@dataclass
class PLCITargetPrediction:
    utterance_index: int
    target_modality: int
    source_pattern: int
    anchor_modalities: Tuple[int, ...]
    paths: torch.Tensor
    context_norm: torch.Tensor
    innovation_norms: torch.Tensor


@dataclass
class PLCIPredictions:
    targets: List[PLCITargetPrediction]


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
    return kappa * value / torch.sqrt(kappa * kappa + norm * norm + eps)


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


class SourceAnchoredPredictor(nn.Module):
    """Predict missing latents from each observed source and current context."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        source_dim: int,
        context_rank: int,
        innovation_rank: int,
        context_cap: float,
        innovation_cap: float,
        embedding_dim: int = 32,
    ) -> None:
        super().__init__()
        dimensions = (
            latent_dim,
            hidden_dim,
            source_dim,
            context_rank,
            innovation_rank,
            embedding_dim,
        )
        if any(not isinstance(value, int) or value <= 0 for value in dimensions):
            raise ValueError("predictor dimensions must be positive integers")
        if not _is_finite_real(context_cap) or context_cap <= 0:
            raise ValueError("context_cap must be a finite positive real number")
        if not _is_finite_real(innovation_cap) or innovation_cap <= 0:
            raise ValueError("innovation_cap must be a finite positive real number")

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.context_cap = float(context_cap)
        self.innovation_cap = float(innovation_cap)
        self.canonicalizers = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(latent_dim),
                    nn.Linear(latent_dim, source_dim),
                    nn.GELU(),
                )
                for name in MODALITIES
            }
        )
        self.anchor_embedding = nn.Embedding(3, embedding_dim)
        self.added_embedding = nn.Embedding(3, embedding_dim)
        self.target_embedding = nn.Embedding(3, embedding_dim)
        self.pattern_embedding = nn.Embedding(6, embedding_dim)

        base_width = source_dim + 3 * embedding_dim
        self.base_trunk = nn.Sequential(
            nn.Linear(base_width, hidden_dim),
            nn.GELU(),
        )
        self.base_outputs = nn.ModuleDict(
            {name: nn.Linear(hidden_dim, latent_dim) for name in MODALITIES}
        )
        self.context_projection = nn.Sequential(
            nn.Linear(hidden_dim, context_rank),
            nn.GELU(),
        )
        self.context_outputs = nn.ModuleDict(
            {name: nn.Linear(context_rank, latent_dim) for name in MODALITIES}
        )

        relation_width = 4 * source_dim + 4 * embedding_dim
        self.innovation_trunk = nn.Sequential(
            nn.Linear(relation_width, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, innovation_rank),
            nn.GELU(),
        )
        self.innovation_outputs = nn.ModuleDict(
            {name: nn.Linear(innovation_rank, latent_dim) for name in MODALITIES}
        )
        for output in list(self.context_outputs.values()) + list(
            self.innovation_outputs.values()
        ):
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)

    def _embedding(self, table: nn.Embedding, index: int, like: torch.Tensor) -> torch.Tensor:
        return table(torch.tensor(index, dtype=torch.long, device=like.device))

    def _base(
        self,
        canonical: torch.Tensor,
        anchor: int,
        target: int,
        pattern: int,
    ) -> torch.Tensor:
        value = torch.cat(
            (
                canonical,
                self._embedding(self.anchor_embedding, anchor, canonical),
                self._embedding(self.target_embedding, target, canonical),
                self._embedding(self.pattern_embedding, pattern, canonical),
            )
        )
        return self.base_outputs[MODALITIES[target]](self.base_trunk(value))

    def _innovation(
        self,
        anchor_value: torch.Tensor,
        added_value: torch.Tensor,
        anchor: int,
        added: int,
        target: int,
        pattern: int,
    ) -> torch.Tensor:
        relation = torch.cat(
            (
                anchor_value,
                added_value,
                added_value - anchor_value,
                anchor_value * added_value,
                self._embedding(self.anchor_embedding, anchor, anchor_value),
                self._embedding(self.added_embedding, added, anchor_value),
                self._embedding(self.target_embedding, target, anchor_value),
                self._embedding(self.pattern_embedding, pattern, anchor_value),
            )
        )
        raw = self.innovation_outputs[MODALITIES[target]](
            self.innovation_trunk(relation)
        )
        return bounded_residual(raw, self.innovation_cap)

    def forward(
        self,
        student_latents: Dict[str, torch.Tensor],
        graph_hidden: torch.Tensor,
        availability: torch.Tensor,
        umask: torch.Tensor,
    ) -> PLCIPredictions:
        if set(student_latents) != set(MODALITIES):
            raise ValueError("student_latents must contain audio, text, and visual")
        if graph_hidden.ndim != 3:
            raise ValueError("graph_hidden must have shape [L, B, H]")
        if availability.ndim != 3 or availability.shape[-1] != 3:
            raise ValueError("availability must have shape [L, B, 3]")
        length, batch = graph_hidden.shape[:2]
        if graph_hidden.shape[-1] != self.hidden_dim:
            raise ValueError("graph_hidden has the wrong hidden dimension")
        if availability.shape[:2] != (length, batch):
            raise ValueError("availability leading dimensions differ")
        if umask.ndim != 2 or umask.shape != (batch, length):
            raise ValueError("umask must have shape [B, L]")
        for name in MODALITIES:
            value = student_latents[name]
            if value.ndim != 3 or value.shape != (length, batch, self.latent_dim):
                raise ValueError(
                    "student_latents values must have shape [L, B, latent_dim]"
                )
        if not bool(torch.isfinite(availability).all()) or not bool(
            ((availability == 0) | (availability == 1)).all()
        ):
            raise ValueError("availability must be binary")

        active_patterns = (
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 1, 0),
            (1, 0, 1),
            (0, 1, 1),
        )
        pattern_ids = {pattern: index for index, pattern in enumerate(active_patterns)}
        records = []  # type: List[PLCITargetPrediction]
        for time in range(length):
            for item in range(batch):
                if not bool(umask[item, time]):
                    continue
                pattern_tuple = tuple(
                    int(value) for value in availability[time, item].tolist()
                )
                if pattern_tuple not in pattern_ids:
                    raise ValueError("valid utterances require one of the six active patterns")
                pattern = pattern_ids[pattern_tuple]
                anchors = tuple(index for index, value in enumerate(pattern_tuple) if value)
                targets = tuple(index for index, value in enumerate(pattern_tuple) if not value)
                canonical = {
                    index: self.canonicalizers[MODALITIES[index]](
                        student_latents[MODALITIES[index]][time, item]
                    )
                    for index in anchors
                }
                for target in targets:
                    target_name = MODALITIES[target]
                    context_raw = self.context_outputs[target_name](
                        self.context_projection(graph_hidden[time, item])
                    )
                    context = bounded_residual(context_raw, self.context_cap)
                    paths = []  # type: List[torch.Tensor]
                    innovations = []  # type: List[torch.Tensor]
                    if len(anchors) == 1:
                        anchor = anchors[0]
                        innovation = torch.zeros_like(context)
                        paths.append(
                            normalize_latent(
                                self._base(canonical[anchor], anchor, target, pattern)
                                + context
                            )
                        )
                        innovations.append(innovation)
                    else:
                        for anchor, added in (anchors, tuple(reversed(anchors))):
                            innovation = self._innovation(
                                canonical[anchor],
                                canonical[added],
                                anchor,
                                added,
                                target,
                                pattern,
                            )
                            paths.append(
                                normalize_latent(
                                    self._base(canonical[anchor], anchor, target, pattern)
                                    + context
                                    + innovation
                                )
                            )
                            innovations.append(innovation)
                    records.append(
                        PLCITargetPrediction(
                            utterance_index=time * batch + item,
                            target_modality=target,
                            source_pattern=pattern,
                            anchor_modalities=anchors,
                            paths=torch.stack(paths),
                            context_norm=torch.norm(context),
                            innovation_norms=torch.stack(
                                [torch.norm(value) for value in innovations]
                            ),
                        )
                    )
        predictions = PLCIPredictions(records)
        predictions._reference = graph_hidden
        return predictions


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
