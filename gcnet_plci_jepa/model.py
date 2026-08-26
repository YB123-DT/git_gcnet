"""GCNet integration for pattern-lattice conditional innovation JEPA."""

from typing import Dict, Tuple

import torch
from torch import nn

from gcnet_modality_jepa.model import GraphModel

from .modules import (
    EMATeacherBank,
    MODALITIES,
    SourceAnchoredPredictor,
    StudentAdapterBank,
    normalize_latent,
)
from .patterns import ACTIVE_PATTERNS, expand_modality_mask


ATV_PATTERN = (1, 1, 1)


class PLCIJEPAGraphModel(GraphModel):
    """Original GCNet plus training-only PLCI student and EMA teacher paths."""

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
        source_dim=256,
        context_rank=32,
        innovation_rank=32,
        context_cap=0.25,
        innovation_cap=0.25,
        pattern_embedding_dim=32,
        predictor_embedding_dim=32,
    ):
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
        )
        self.dimensions = (adim, tdim, vdim)
        self.latent_dim = latent_dim
        self.recurrent_dim = 2 * D_e
        self.hidden_dim = self.recurrent_dim + graph_hidden_size
        self.student_adapter = StudentAdapterBank(self.dimensions, latent_dim)
        self.teacher = EMATeacherBank(self.student_adapter.projectors)
        self.pattern_embedding = nn.Embedding(
            len(ACTIVE_PATTERNS) + 1, pattern_embedding_dim
        )
        self.pattern_projection = nn.Linear(
            pattern_embedding_dim, self.recurrent_dim
        )
        nn.init.zeros_(self.pattern_projection.weight)
        nn.init.zeros_(self.pattern_projection.bias)
        self.predictor = SourceAnchoredPredictor(
            latent_dim=latent_dim,
            hidden_dim=self.hidden_dim,
            source_dim=source_dim,
            context_rank=context_rank,
            innovation_rank=innovation_rank,
            context_cap=context_cap,
            innovation_cap=innovation_cap,
            embedding_dim=predictor_embedding_dim,
        )
        self.ema_step = 0
        self.last_teacher_tau = None

    @staticmethod
    def _feature_tensor(features, name):
        if torch.is_tensor(features):
            return features
        if isinstance(features, (list, tuple)) and len(features) == 1:
            if torch.is_tensor(features[0]):
                return features[0]
        raise ValueError("{} must be a tensor or one-element tensor list".format(name))

    def _validate_availability(self, availability, umask, allow_atv):
        if availability.ndim != 3 or availability.shape[-1] != 3:
            raise ValueError("availability must have shape [L, B, 3]")
        length, batch = availability.shape[:2]
        if umask.ndim != 2 or umask.shape != (batch, length):
            raise ValueError("umask must have shape [B, L]")
        if not bool(torch.isfinite(availability).all()) or not bool(
            ((availability == 0) | (availability == 1)).all()
        ):
            raise ValueError("availability must be binary")
        valid = umask.T.bool()
        if bool((availability[~valid] != 0).any()):
            raise ValueError("padding availability must use the zero pattern")
        allowed = set(ACTIVE_PATTERNS)
        if allow_atv:
            allowed.add(ATV_PATTERN)
        for pattern in availability[valid].detach().cpu().tolist():
            if tuple(int(value) for value in pattern) not in allowed:
                raise ValueError("valid utterance has an invalid availability pattern")
        return valid

    def _zero_latents(self, features):
        return {
            name: features.new_zeros(
                features.shape[0], features.shape[1], self.latent_dim
            )
            for name in MODALITIES
        }

    def pattern_residual(self, availability, umask, allow_atv=False):
        valid = self._validate_availability(availability, umask, allow_atv)
        pattern_to_id = {
            pattern: index for index, pattern in enumerate(ACTIVE_PATTERNS)
        }
        pattern_to_id[ATV_PATTERN] = len(ACTIVE_PATTERNS)
        ids = torch.zeros(
            availability.shape[:2], dtype=torch.long, device=availability.device
        )
        active = torch.zeros_like(valid)
        for pattern, pattern_id in pattern_to_id.items():
            matches = (availability == availability.new_tensor(pattern)).all(dim=-1)
            selected = matches & valid
            ids[selected] = pattern_id
            if pattern != ATV_PATTERN:
                active |= selected
        residual = self.pattern_projection.weight.new_zeros(
            availability.shape[0], availability.shape[1], self.recurrent_dim
        )
        if bool(active.any()):
            residual[active] = self.pattern_projection(
                self.pattern_embedding(ids[active])
            )
        return residual

    def forward_natural(
        self, inputfeats, availability, qmask, umask, seq_lengths
    ):
        features = self._feature_tensor(inputfeats, "inputfeats")
        valid = self._validate_availability(availability, umask, allow_atv=True)
        if bool((availability[valid] == 1).all()):
            log_prob, rec_outputs, hidden = super().forward(
                inputfeats, qmask, umask, seq_lengths
            )
            return log_prob, rec_outputs, hidden, self._zero_latents(features)

        adapted, latents = self.student_adapter(features, availability)
        residual = self.pattern_residual(availability, umask, allow_atv=True)
        hidden = self.encode_hidden(
            [adapted], qmask, umask, seq_lengths, residual
        )
        log_prob = self.smax_fc(hidden)
        rec_outputs = [self.linear_rec(hidden)] if self.enable_reconstruction else []
        return log_prob, rec_outputs, hidden, latents

    def forward_auxiliary(
        self, source_features, availability, qmask, umask, seq_lengths
    ):
        source = self._feature_tensor(source_features, "source_features")
        self._validate_availability(availability, umask, allow_atv=False)
        if source.ndim != 3 or source.shape[-1] != sum(self.dimensions):
            raise ValueError("source_features must have shape [L, B, sumD]")
        if source.shape[:2] != availability.shape[:2]:
            raise ValueError("source_features and availability leading dimensions differ")
        expanded = expand_modality_mask(availability, self.dimensions).to(
            dtype=source.dtype
        )
        if bool(torch.count_nonzero(source * (1 - expanded))):
            raise ValueError("missing modality blocks must be zero")

        adapted, latents = self.student_adapter(source, availability)
        residual = self.pattern_residual(availability, umask, allow_atv=False)
        hidden = self.encode_hidden(
            [adapted], qmask, umask, seq_lengths, residual
        )
        predictions = self.predictor(latents, hidden, availability, umask)
        return predictions, hidden, latents

    @torch.no_grad()
    def encode_teacher_targets(self, teacher_features) -> Dict[str, torch.Tensor]:
        features = self._feature_tensor(teacher_features, "teacher_features")
        return {
            name: normalize_latent(value)
            for name, value in self.teacher(features).items()
        }

    @torch.no_grad()
    def update_teacher(self, tau):
        self.teacher.update_from(self.student_adapter.projectors, tau)
        self.ema_step += 1
        self.last_teacher_tau = float(tau)

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self
