"""OSRAM: a compact bidirectional block-delta memory for conversations.

The implementation keeps the three modality slots separate until the write
operator.  Reads are performed before the current utterance is written, so a
node can choose a memory address using its own content without immediately
reading its own value back.  The write is a single block update, rather than a
sequence of A/T/V writes, which makes it invariant to modality column order.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


MODALITIES = ("audio", "text", "visual")
QUERY_TYPES = ("base", "audio", "text", "visual")
OSRAM_ABLATIONS = ("full", "local-only", "local-base")
OSRAM_EMOTION_ABLATIONS = (*OSRAM_ABLATIONS, "local-gap")
OSRAM_GAP_READ_MODES = ("residual", "raw")
OSRAM_BETA_MODES = ("embedded", "external-head")


def _logit(probability: float) -> float:
    probability = float(probability)
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be strictly between zero and one")
    return math.log(probability / (1.0 - probability))


class LocalCrossAttentionFusion(nn.Module):
    """One local query attends to four explicitly masked context slots only."""

    def __init__(self, local_dim, context_dim, output_dim, dropout=0.5):
        super().__init__()
        self.local_dim, self.context_dim = local_dim, context_dim
        self.attention_dim, self.num_heads = 512, 4
        self.local_norm = nn.LayerNorm(local_dim)
        self.context_norm = nn.LayerNorm(context_dim)
        self.evidence_type = nn.Embedding(4, 16)
        self.query = nn.Linear(local_dim, 512)
        self.key = nn.Linear(context_dim + 16, 512)
        self.value = nn.Linear(context_dim, 512)
        self.context_out = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, output_dim))
        self.local_skip = nn.Linear(local_dim, output_dim)
        self.emotion_norm = nn.LayerNorm(output_dim)
        nn.init.zeros_(self.context_out[-1].weight)
        nn.init.zeros_(self.context_out[-1].bias)
        self.last_diagnostics = {}

    def forward(self, local, base_context, gap_context, availability, umask):
        if local.ndim != 3 or local.shape[-1] != self.local_dim:
            raise ValueError("local must be [L,B,local_dim]")
        length, batch = local.shape[:2]
        if (base_context.shape != (length, batch, self.context_dim)
                or gap_context.shape != (length, batch, 3, self.context_dim)
                or availability.shape != (length, batch, 3) or umask.shape != (batch, length)):
            raise ValueError("fusion context/mask shapes do not match local")
        valid = umask.T.bool()
        if bool(((availability[valid] != 0) & (availability[valid] != 1)).any()):
            raise ValueError("availability must be binary on valid utterances")
        active = torch.cat([valid[..., None], valid[..., None] & ~availability.bool()], -1)
        evidence = torch.cat([base_context.unsqueeze(2), gap_context], 2)
        evidence = torch.where(active[..., None], evidence, torch.zeros_like(evidence))
        # Evaluate only valid rows: padding must never produce an all-masked softmax.
        normalized = self.context_norm(evidence[valid])
        count = normalized.shape[0]
        types = self.evidence_type.weight.unsqueeze(0).expand(count, -1, -1)
        q = self.query(self.local_norm(local[valid])).reshape(count, 4, 128)
        k = self.key(torch.cat([normalized, types], -1)).reshape(count, 4, 4, 128).transpose(1, 2)
        v = self.value(normalized).reshape(count, 4, 4, 128).transpose(1, 2)
        scores = (q.unsqueeze(2) * k).sum(-1) / math.sqrt(128)
        attention = scores.masked_fill(~active[valid].unsqueeze(1), -torch.inf).softmax(-1)
        context = (attention[..., None] * v).sum(2).reshape(count, 512)
        residual = self.context_out(context)
        subject = self.local_skip(local[valid])
        hidden = local.new_zeros(length, batch, self.local_skip.out_features)
        hidden[valid] = self.emotion_norm(subject + residual)
        with torch.no_grad():
            self.last_active = active.detach()
            weights = local.new_zeros(length, batch, 4, 4)
            weights[valid] = attention.detach()
            self.last_attention = weights
            names = ('base', 'gap_audio', 'gap_text', 'gap_visual')
            means = {}
            for i, name in enumerate(names):
                selected = active[valid][:, i]
                means[name] = float(attention[:, :, i][selected].mean()) if bool(selected.any()) else None
            rn, sn = residual.norm(dim=-1), subject.norm(dim=-1)
            self.last_diagnostics = dict(mean_attention=means,
                active_evidence_count_mean=float(active[valid].sum(-1).float().mean()) if count else None,
                context_residual_norm=float(rn.mean()) if count else None,
                local_subject_norm=float(sn.mean()) if count else None,
                residual_subject_norm_ratio=float((rn/sn.clamp_min(1e-8)).mean()) if count else None)
        return hidden


class ModalityTrackFlatFusion(nn.Module):
    """Flat emotion readout over modality-local tracks and OSRAM evidence.

    The OSRAM scan remains responsible for the fused node, keys, values and
    queries.  This readout only replaces the classification-side local input:
    each observed modality gets a shared local transform, while missing tracks
    are hard-zeroed before concatenating Base, masked Gap slots and availability
    into one flat MLP input.
    """

    def __init__(self, local_dim, context_dim, output_dim, dropout=0.5):
        super().__init__()
        self.local_dim = int(local_dim)
        self.context_dim = int(context_dim)
        input_dim = 3 * self.local_dim + 4 * self.context_dim + 3
        self.emotion_adapter = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, int(output_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(output_dim), int(output_dim)),
        )
        self.emotion_norm = nn.LayerNorm(int(output_dim))
        self.last_diagnostics = {}

    def forward(self, tracks, base_context, gap_context, availability, umask):
        if tracks.ndim != 4 or tracks.shape[2:] != (3, self.local_dim):
            raise ValueError("tracks must be [L,B,3,local_dim]")
        length, batch = tracks.shape[:2]
        if (base_context.shape != (length, batch, self.context_dim)
                or gap_context.shape != (length, batch, 3, self.context_dim)
                or availability.shape != (length, batch, 3)
                or umask.shape != (batch, length)):
            raise ValueError("track fusion context/mask shapes do not match tracks")
        valid = umask.transpose(0, 1).bool()
        if bool(((availability[valid] != 0) & (availability[valid] != 1)).any()):
            raise ValueError("availability must be binary on valid utterances")
        observed = availability.to(dtype=tracks.dtype)
        missing = 1.0 - observed
        masked_gap = torch.where(
            missing.bool().unsqueeze(-1), gap_context, torch.zeros_like(gap_context)
        )
        safe_tracks = torch.where(
            observed.bool().unsqueeze(-1), tracks, torch.zeros_like(tracks)
        )
        fusion_input = torch.cat(
            (
                safe_tracks.reshape(length, batch, -1),
                base_context,
                masked_gap.reshape(length, batch, -1),
                observed,
            ),
            dim=-1,
        )
        hidden = self.emotion_norm(self.emotion_adapter(fusion_input))
        hidden = hidden * valid.unsqueeze(-1).to(hidden.dtype)
        with torch.no_grad():
            self.last_diagnostics = {
                "local_track_norm": {
                    name: float(safe_tracks[..., index, :][valid].norm(dim=-1).mean())
                    if bool(valid.any()) else 0.0
                    for index, name in enumerate(MODALITIES)
                },
                "active_evidence_count_mean": float(
                    (1 + missing.sum(dim=-1))[valid].mean()
                ) if bool(valid.any()) else None,
            }
        return hidden


class ModalityTrackResidualFusion(nn.Module):
    """Add a zero-initialized modality-track residual to the flat anchor.

    The flat OSRAM readout remains the task representation.  This module only
    sees the three modality-local slots and availability, so it can test for
    incremental modality identity information without replacing the stable
    fused/local skip path.
    """

    def __init__(self, local_dim, output_dim, dropout=0.5):
        super().__init__()
        self.local_dim = int(local_dim)
        self.output_dim = int(output_dim)
        self.input_dim = 3 * self.local_dim + 3
        self.residual = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.output_dim, self.output_dim),
        )
        self.gate = nn.Linear(self.input_dim, 1)
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)
        self.last_diagnostics = {}

    def forward(self, latents, availability, umask, modality_embeddings):
        if set(latents) != set(MODALITIES):
            raise ValueError("latents must contain audio, text, and visual")
        reference = latents[MODALITIES[0]]
        if reference.ndim != 3 or reference.shape[-1] != self.local_dim:
            raise ValueError("modality latents must be [L,B,local_dim]")
        length, batch = reference.shape[:2]
        if (availability.shape != (length, batch, 3)
                or umask.shape != (batch, length)
                or modality_embeddings.shape != (3, self.local_dim)):
            raise ValueError("modality-track residual shapes do not match")
        valid = umask.transpose(0, 1).bool()
        observed = availability.to(dtype=reference.dtype)
        slots = []
        for index, name in enumerate(MODALITIES):
            slot = latents[name] + modality_embeddings[index].view(1, 1, -1)
            slots.append(torch.where(
                observed[..., index : index + 1].bool(), slot, torch.zeros_like(slot)
            ))
        track_input = torch.cat((*slots, observed), dim=-1)
        track_input = track_input * valid.unsqueeze(-1).to(track_input.dtype)
        residual = self.residual(track_input)
        gate = torch.sigmoid(self.gate(track_input))
        residual = residual * gate
        residual = residual * valid.unsqueeze(-1).to(residual.dtype)
        with torch.no_grad():
            valid_rows = valid
            self.last_diagnostics = {
                "gate_mean": float(gate[valid_rows].mean()) if bool(valid_rows.any()) else None,
                "residual_norm": float(residual[valid_rows].norm(dim=-1).mean())
                if bool(valid_rows.any()) else None,
                "active_track_count_mean": float(observed[valid_rows].sum(dim=-1).mean())
                if bool(valid_rows.any()) else None,
            }
        return residual


class LocalCenteredContextFusion(nn.Module):
    """Local subject plus an active-count-averaged, sigmoid-gated residual.

    Evidence projections are shared across Base and the three target Gaps.
    This module never reads or writes the recurrent memory or calls a predictor.
    """

    def __init__(self, local_dim, context_dim, output_dim,
                 interaction_dim=128, type_dim=16, dropout=0.5):
        super().__init__()
        if any(int(d) <= 0 for d in (local_dim, context_dim, output_dim, interaction_dim, type_dim)):
            raise ValueError("fusion dimensions must be positive")
        self.local_dim, self.context_dim = int(local_dim), int(context_dim)
        self.local_norm = nn.LayerNorm(local_dim)
        self.context_norm = nn.LayerNorm(context_dim)
        self.query = nn.Linear(local_dim, interaction_dim)
        self.key = nn.Linear(context_dim, interaction_dim)
        self.value = nn.Linear(context_dim, interaction_dim)
        self.evidence_type = nn.Embedding(4, type_dim)
        self.gate = nn.Sequential(nn.Linear(4 * interaction_dim + type_dim, interaction_dim),
                                  nn.GELU(), nn.Linear(interaction_dim, 1))
        self.context_out = nn.Sequential(nn.Dropout(dropout), nn.Linear(interaction_dim, output_dim))
        self.local_skip = nn.Linear(local_dim, output_dim)
        self.emotion_norm = nn.LayerNorm(output_dim)
        nn.init.zeros_(self.context_out[-1].weight)
        nn.init.zeros_(self.context_out[-1].bias)
        self.last_diagnostics = {}

    def forward(self, local, base_context, gap_context, availability, umask):
        if local.ndim != 3 or local.shape[-1] != self.local_dim:
            raise ValueError("local must be [L,B,local_dim]")
        length, batch = local.shape[:2]
        if (base_context.shape != (length, batch, self.context_dim)
                or gap_context.shape != (length, batch, 3, self.context_dim)
                or availability.shape != (length, batch, 3)
                or umask.shape != (batch, length)):
            raise ValueError("fusion context/mask shapes do not match local")
        valid = umask.transpose(0, 1).bool()
        if bool(((availability[valid] != 0) & (availability[valid] != 1)).any()):
            raise ValueError("availability must be binary on valid utterances")
        active = torch.cat((valid.unsqueeze(-1),
                            valid.unsqueeze(-1) & ~availability.bool()), dim=-1)
        evidence = torch.cat((base_context.unsqueeze(2), gap_context), dim=2)
        # Mask before normalization as well as after sigmoid: even arbitrary
        # inactive-slot values cannot leak through affine biases or NaNs.
        evidence = torch.where(active.unsqueeze(-1), evidence, torch.zeros_like(evidence))
        safe_local = torch.where(valid.unsqueeze(-1), local, torch.zeros_like(local))
        normalized = self.context_norm(evidence)
        q = self.query(self.local_norm(safe_local)).unsqueeze(2).expand(-1, -1, 4, -1)
        k, v = self.key(normalized), self.value(normalized)
        types = self.evidence_type.weight.view(1, 1, 4, -1).expand(length, batch, -1, -1)
        interaction = torch.cat((q, k, q * k, (q - k).abs(), types), dim=-1)
        gates = torch.sigmoid(self.gate(interaction).squeeze(-1))
        gates = gates.masked_fill(~active, 0)
        count = active.sum(-1).clamp_min(1).to(v.dtype)
        delta = (gates.unsqueeze(-1) * v).sum(2) / count.unsqueeze(-1)
        residual = self.context_out(delta)
        subject = self.local_skip(safe_local)
        hidden = self.emotion_norm(subject + residual).masked_fill(~valid.unsqueeze(-1), 0)
        with torch.no_grad():
            self.last_active = active.detach()
            self.last_gates = gates.detach()
            gate_means = {}
            for i, name in enumerate(('base', 'gap_audio', 'gap_text', 'gap_visual')):
                selected = active[..., i]
                gate_means[name] = float(gates[..., i][selected].mean()) if bool(selected.any()) else None
            has_valid = bool(valid.any())
            residual_norm = residual[valid].norm(dim=-1)
            subject_norm = subject[valid].norm(dim=-1)
            self.last_diagnostics = {
                'mean_gate': gate_means,
                'active_evidence_count_mean': float(active.sum(-1)[valid].float().mean()) if has_valid else None,
                'context_residual_norm': float(residual_norm.mean()) if has_valid else None,
                'local_subject_norm': float(subject_norm.mean()) if has_valid else None,
                'residual_subject_norm_ratio': float((residual_norm / subject_norm.clamp_min(1e-8)).mean()) if has_valid else None,
            }
        return hidden


class BaseGapDeltaFusion(nn.Module):
    """Flat readout where each active Gap is represented relative to Base.

    The OSRAM scan is unchanged.  This module only re-parameterizes the
    classification-side context slots as a generic history anchor (Base) and
    missing-modality-specific deviations from that anchor (Gap - Base).
    """

    def __init__(self, local_dim, context_dim, output_dim, dropout=0.5):
        super().__init__()
        self.local_dim = int(local_dim)
        self.context_dim = int(context_dim)
        self.output_dim = int(output_dim)
        self.base_projection = nn.Linear(self.context_dim, self.context_dim)
        self.gap_projection = nn.Linear(self.context_dim, self.context_dim)
        self.last_diagnostics = {}

    def forward(
        self,
        local,
        base_context,
        gap_context,
        availability,
        umask,
        *,
        emotion_adapter,
        local_skip,
        emotion_norm,
    ):
        if local.ndim != 3 or local.shape[-1] != self.local_dim:
            raise ValueError("local must be [L,B,local_dim]")
        length, batch = local.shape[:2]
        if (
            base_context.shape != (length, batch, self.context_dim)
            or gap_context.shape != (length, batch, 3, self.context_dim)
            or availability.shape != (length, batch, 3)
            or umask.shape != (batch, length)
        ):
            raise ValueError("fusion context/mask shapes do not match local")
        valid = umask.transpose(0, 1).bool()
        if bool(((availability[valid] != 0) & (availability[valid] != 1)).any()):
            raise ValueError("availability must be binary on valid utterances")
        valid_float = valid.unsqueeze(-1).to(local.dtype)
        active_missing = valid.unsqueeze(-1) & ~availability.bool()
        safe_local = torch.where(valid.unsqueeze(-1), local, torch.zeros_like(local))
        safe_base = torch.where(
            valid.unsqueeze(-1), base_context, torch.zeros_like(base_context)
        )
        safe_gap = torch.where(
            active_missing.unsqueeze(-1), gap_context, torch.zeros_like(gap_context)
        )

        base = self.base_projection(safe_base)
        gap = self.gap_projection(safe_gap)
        delta = (gap - base.unsqueeze(2)) * active_missing.unsqueeze(-1).to(gap.dtype)
        fusion_input = torch.cat(
            (safe_local, base, delta.reshape(length, batch, -1)), dim=-1
        )
        context_residual = emotion_adapter(fusion_input)
        hidden = emotion_norm(local_skip(safe_local) + context_residual)
        hidden = hidden * valid_float
        with torch.no_grad():
            valid_rows = valid
            base_norm = (
                base[valid_rows].norm(dim=-1)
                if bool(valid_rows.any()) else base.new_empty(0)
            )
            self.last_diagnostics = {
                "base_norm": float(base_norm.mean()) if base_norm.numel() else None,
                "delta_norm": {
                    name: float(delta[..., index, :][valid_rows].norm(dim=-1).mean())
                    if bool(valid_rows.any()) else None
                    for index, name in enumerate(MODALITIES)
                },
                "active_evidence_count_mean": float(
                    (1 + active_missing.sum(dim=-1))[valid_rows].float().mean()
                ) if bool(valid_rows.any()) else None,
                "context_residual_norm": float(
                    context_residual[valid_rows].norm(dim=-1).mean()
                ) if bool(valid_rows.any()) else None,
            }
        return hidden


class MemoryShiftFilter(nn.Module):
    """Zero-initialized readout residual from typed, filtered memory shifts.

    Base and three fixed Gap slots are filtered independently. Normalization
    uses the number of active slots, not the sum of their learned gates.
    """

    SLOT_NAMES = ("base", "gap_audio", "gap_text", "gap_visual")

    def __init__(self, local_dim, context_dim, output_dim, relation_dim=128,
                 filter_width=128, filter_depth=1):
        super().__init__()
        for name, value in (("filter_width", filter_width), ("filter_depth", filter_depth)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.local_dim = int(local_dim)
        self.context_dim = int(context_dim)
        self.output_dim = int(output_dim)
        self.relation_dim = int(relation_dim)
        self.local_relation = nn.Linear(self.local_dim, self.relation_dim)
        self.memory_relation = nn.Linear(self.context_dim, self.relation_dim)
        self.evidence_type = nn.Embedding(4, self.relation_dim)
        layers = []
        in_dim = 6 * self.relation_dim
        for _ in range(filter_depth):
            layers.extend((nn.Linear(in_dim, filter_width), nn.GELU()))
            in_dim = filter_width
        layers.append(nn.Linear(filter_width, 1))
        self.filter = nn.Sequential(*layers)
        self.residual_projector = nn.Linear(self.relation_dim, self.output_dim)
        nn.init.zeros_(self.residual_projector.weight)
        nn.init.zeros_(self.residual_projector.bias)
        self.last_diagnostics = {}

    def forward(self, local, base_context, gap_context, availability, umask, flat_anchor):
        if local.ndim != 3 or local.shape[-1] != self.local_dim:
            raise ValueError("local must be [L,B,local_dim]")
        length, batch = local.shape[:2]
        if (
            base_context.shape != (length, batch, self.context_dim)
            or gap_context.shape != (length, batch, 3, self.context_dim)
            or availability.shape != (length, batch, 3)
            or umask.shape != (batch, length)
            or flat_anchor.shape != (length, batch, self.output_dim)
        ):
            raise ValueError("fusion context/mask shapes do not match local")
        valid = umask.transpose(0, 1).bool()
        if bool(((availability[valid] != 0) & (availability[valid] != 1)).any()):
            raise ValueError("availability must be binary on valid utterances")
        active = torch.cat((valid.unsqueeze(-1),
                            valid.unsqueeze(-1) & ~availability.bool()), dim=-1)
        safe_local = torch.where(valid.unsqueeze(-1), local, torch.zeros_like(local))
        evidence = torch.cat((base_context.unsqueeze(2), gap_context), dim=2)
        evidence = torch.where(active.unsqueeze(-1), evidence, torch.zeros_like(evidence))
        q = self.local_relation(safe_local).unsqueeze(2).expand(-1, -1, 4, -1)
        types = self.evidence_type.weight.view(1, 1, 4, -1).expand_as(q)
        k = self.memory_relation(evidence) + types
        shift = k - q
        gate = torch.sigmoid(self.filter(torch.cat(
            (q, k, shift, shift.abs(), q * k, types), dim=-1
        )))
        gate = torch.where(active.unsqueeze(-1), gate, torch.zeros_like(gate))
        contribution = torch.where(active.unsqueeze(-1), gate * shift, torch.zeros_like(shift))
        filtered_shift = contribution.sum(dim=2) / active.sum(dim=-1).clamp_min(1).unsqueeze(-1)
        residual = self.residual_projector(filtered_shift)
        residual = torch.where(valid.unsqueeze(-1), residual, torch.zeros_like(residual))
        with torch.no_grad():
            count = int(valid.sum())
            active_counts = {name: int(active[..., i].sum())
                             for i, name in enumerate(self.SLOT_NAMES)}
            self.last_diagnostics = {
                "filter_mean": {name: float(gate[..., i, 0][active[..., i]].mean())
                                if active_counts[name] else None
                                for i, name in enumerate(self.SLOT_NAMES)},
                "active_counts": active_counts,
                "valid_count": count,
                "filtered_shift_norm": float(filtered_shift[valid].norm(dim=-1).mean()) if count else None,
                "shift_residual_norm": float(residual[valid].norm(dim=-1).mean()) if count else None,
                "residual_anchor_norm_ratio": float((residual[valid].norm(dim=-1) /
                    flat_anchor[valid].norm(dim=-1).clamp_min(1e-8)).mean()) if count else None,
            }
        return residual


class HierarchicalEvidenceGate(nn.Module):
    """Identity-initialized feature filtering followed by active-evidence competition."""

    def __init__(self, latent_dim, context_dim, feature_only=False):
        super().__init__()
        self.feature_only = bool(feature_only)
        self.local_relation = nn.Linear(latent_dim, 128)
        self.memory_relation = nn.Linear(context_dim, 128)
        self.filtered_relation = nn.Linear(context_dim, 128)
        self.type_embedding = nn.Embedding(4, 8)
        self.feature_input = nn.Linear(128 * 2 + 8 + 3, 128)
        self.feature_output = nn.Linear(128, context_dim)
        self.evidence_input = nn.Linear(128 * 2 + 8 + 3, 128)
        self.evidence_output = nn.Linear(128, 1)
        for layer in (self.feature_output, self.evidence_output):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        # Retain construction order/state compatibility to isolate removal of Level 2.
        if self.feature_only:
            for module in (self.filtered_relation, self.evidence_input, self.evidence_output):
                module.requires_grad_(False)
        self.last_diagnostics = {}

    def forward(self, local, base, gap, availability, umask):
        valid = umask.T.bool()
        active = torch.cat((valid[..., None], valid[..., None] & ~availability.bool()), -1)
        local = torch.where(valid[..., None], local, 0.)
        availability = torch.where(valid[..., None], availability, 0.)
        evidence = torch.where(active[..., None], torch.cat((base.unsqueeze(-2), gap), -2), 0.)
        q = self.local_relation(local).unsqueeze(-2).expand(*evidence.shape[:-1], 128)
        types = self.type_embedding.weight.expand(*evidence.shape[:-2], 4, 8)
        avail = availability.unsqueeze(-2).expand(*evidence.shape[:-1], 3)
        feature_condition = torch.cat((q, self.memory_relation(evidence), types, avail), -1)
        feature = 2. * torch.sigmoid(self.feature_output(torch.nn.functional.elu(self.feature_input(feature_condition))))
        feature = torch.where(active[..., None], feature, 0.)
        filtered = feature * evidence
        if self.feature_only:
            reweight = active.to(filtered.dtype)
            # Uniform alpha is diagnostic bookkeeping only; no score/softmax executes.
            alpha = reweight / active.sum(-1, keepdim=True).clamp_min(1)
        else:
            condition = torch.cat((q, self.filtered_relation(filtered), types, avail), -1)
            scores = self.evidence_output(torch.nn.functional.elu(self.evidence_input(condition))).squeeze(-1)
            scores = scores.masked_fill(~active, -torch.inf)
            scores = torch.where(valid[..., None], scores, 0.)
            alpha = torch.where(active, torch.softmax(scores, -1), 0.)
            reweight = active.sum(-1, keepdim=True).to(alpha.dtype) * alpha
        result = torch.where(active[..., None], reweight[..., None] * filtered, 0.)
        with torch.no_grad():
            entropy = -(alpha * alpha.clamp_min(torch.finfo(alpha.dtype).tiny).log()).sum(-1)
            self.last_diagnostics = {'valid_count': int(valid.sum()),
                'entropy_mean': float(entropy[valid].double().mean()) if valid.any() else None}
            for i, name in enumerate(('base', 'gap_a', 'gap_t', 'gap_v')):
                f = feature[..., i, :][active[..., i]].double()
                a = alpha[..., i][active[..., i]].double()
                r = reweight[..., i][active[..., i]].double()
                self.last_diagnostics[name] = {
                    'active_count': a.numel(),
                    'feature_mean': float(f.mean()) if a.numel() else None,
                    'feature_abs_deviation_mean': float((f - 1.).abs().mean()) if a.numel() else None,
                    'feature_saturation_fraction': float(((f <= .01) | (f >= 1.99)).double().mean()) if a.numel() else None,
                    'alpha_mean': float(a.mean()) if a.numel() else None,
                    'reweight_mean': float(r.mean()) if a.numel() else None,
                }
        return result


class LocalConditionedEvidenceGate(nn.Module):
    """Identity-initialized scalar modulation of each original Flat evidence."""

    def __init__(self, latent_dim, context_dim):
        super().__init__()
        self.local_relation = nn.Linear(latent_dim, 128)
        self.memory_relation = nn.Linear(context_dim, 128)
        self.type_embedding = nn.Embedding(4, 8)
        self.input = nn.Linear(4 * 128 + 2 + 8 + 3, 128)
        self.output = nn.Linear(128, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.regularization = None
        self.regularization_l1 = None
        self.last_diagnostics = {}

    def forward(self, local, base, gap, availability, umask):
        valid = umask.T.bool()
        active = torch.cat((valid[..., None], valid[..., None] & ~availability.bool()), -1)
        local = torch.where(valid[..., None], local, 0.)
        evidence = torch.cat((base.unsqueeze(-2), gap), -2)
        evidence = torch.where(active[..., None], evidence, 0.)
        availability = torch.where(valid[..., None], availability, 0.)
        q = self.local_relation(local).unsqueeze(-2).expand(*evidence.shape[:-1], 128)
        k = self.memory_relation(evidence)
        types = self.type_embedding.weight.expand(*evidence.shape[:-2], 4, 8)
        condition = torch.cat((q, k, q*k, (q-k).abs(),
                               q.norm(dim=-1, keepdim=True), k.norm(dim=-1, keepdim=True),
                               types, availability.unsqueeze(-2).expand(*evidence.shape[:-1], 3)), -1)
        raw = 1. + .2 * torch.tanh(self.output(torch.nn.functional.elu(self.input(condition))).squeeze(-1))
        gates = torch.where(active, raw, 0.)
        penalty = torch.where(active, (raw-1.).square(), 0.)
        self.regularization = penalty.sum() / active.sum().clamp_min(1)
        self.regularization_l1 = torch.where(active, (raw-1.).abs(), 0.).sum() / active.sum().clamp_min(1)
        with torch.no_grad():
            self.last_diagnostics = {'active_count': int(active.sum()),
                                     'regularization': float(self.regularization.detach())}
            for i, name in enumerate(('base', 'gap_a', 'gap_t', 'gap_v')):
                values = gates[..., i][active[..., i]].double()
                self.last_diagnostics[name] = {
                    'active_count': values.numel(),
                    'gate_mean': float(values.mean()) if values.numel() else None,
                    'gate_abs_deviation_mean': float((values-1.).abs().mean()) if values.numel() else None,
                    'gate_near_one_fraction': float(((values-1.).abs() <= .01).double().mean()) if values.numel() else None,
                    'gate_saturation_fraction': float(((values <= .81) | (values >= 1.19)).double().mean()) if values.numel() else None,
                }
        return gates


class HistoryInputGate(nn.Module):
    """One bounded history-retention coefficient per utterance; no memory edits."""

    def __init__(self, latent_dim, context_dim):
        super().__init__()
        self.input = nn.Linear(latent_dim + 4 * context_dim + 3, 128)
        self.output = nn.Linear(128, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.last_diagnostics = {}

    def forward(self, local, base, gap, availability, umask):
        valid = umask.T.bool()
        local = torch.where(valid[..., None], local, 0.)
        base = torch.where(valid[..., None], base, 0.)
        gap = torch.where((valid[..., None] & ~availability.bool())[..., None], gap, 0.)
        availability = torch.where(valid[..., None], availability, 0.)
        condition = torch.cat((local, base, gap.flatten(2), availability), -1)
        alpha = 1. + .2 * torch.tanh(self.output(torch.nn.functional.elu(self.input(condition))))
        alpha = torch.where(valid[..., None], alpha, 0.)
        with torch.no_grad():
            # Preserve small near-one variation in diagnostic second moments.
            values = alpha[valid].flatten().double()
            count = values.numel()
            self.last_diagnostics = {
                'valid_count': count,
                'alpha_mean': float(values.mean()) if count else None,
                'alpha_second_moment': float(values.square().mean()) if count else None,
                'alpha_below_one_fraction': float((values < 1).float().mean()) if count else None,
                'alpha_above_one_fraction': float((values > 1).float().mean()) if count else None,
                'alpha_saturation_fraction': float(((values <= .81) | (values >= 1.19)).float().mean()) if count else None,
            }
        return alpha


class PostGRN(nn.Module):
    """Optional identity-initialized residual after the unchanged Flat norm."""

    def __init__(self, latent_dim, context_dim, output_dim, dropout=0.5):
        super().__init__()
        condition_dim = latent_dim + 4 * context_dim + 3
        self.condition_norm = nn.LayerNorm(condition_dim)
        self.linear_x = nn.Linear(output_dim, 128)
        self.linear_c = nn.Linear(condition_dim, 128)
        self.linear2 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(dropout)
        self.linear_gate = nn.Linear(128, output_dim)
        self.linear_value = nn.Linear(128, output_dim)
        nn.init.zeros_(self.linear_gate.bias)
        nn.init.zeros_(self.linear_value.weight)
        nn.init.zeros_(self.linear_value.bias)
        self.last_diagnostics = {}

    def forward(self, x, local, emotion_base, emotion_gap, availability, umask):
        valid = umask.transpose(0, 1).bool()
        active_gap = valid.unsqueeze(-1) & ~availability.bool()
        x = torch.where(valid.unsqueeze(-1), x, torch.zeros_like(x))
        local = torch.where(valid.unsqueeze(-1), local, torch.zeros_like(local))
        base = torch.where(valid.unsqueeze(-1), emotion_base, torch.zeros_like(emotion_base))
        gap = torch.where(active_gap.unsqueeze(-1), emotion_gap, torch.zeros_like(emotion_gap))
        available = torch.where(valid.unsqueeze(-1), availability, torch.zeros_like(availability)).to(x.dtype)
        condition = torch.cat((local, base, gap.flatten(2), available), dim=-1)
        z = self.dropout(self.linear2(F.elu(self.linear_x(x) + self.linear_c(self.condition_norm(condition)))))
        gate = torch.sigmoid(self.linear_gate(z))
        residual = gate * self.linear_value(z)
        hidden = torch.where(valid.unsqueeze(-1), x + residual, torch.zeros_like(x))
        with torch.no_grad():
            count = int(valid.sum().item())
            self.last_diagnostics = {
                "valid_count": count,
                "gate_mean": float(gate[valid].mean().item()) if count else None,
                "gate_saturation_fraction": float(
                    ((gate[valid] <= 0.05) | (gate[valid] >= 0.95)).float().mean().item()
                ) if count else None,
                "gated_residual_flat_norm_ratio": float(
                    (residual[valid].norm(dim=-1) / x[valid].norm(dim=-1).clamp_min(1e-8)).mean().item()
                ) if count else None,
            }
        return hidden


class CurrentHistoryRelationBlock(nn.Module):
    """Pairwise Local/history representation; no weighting or memory updates."""

    def __init__(self, latent_dim, context_dim, output_dim, relation_dim=128,
                 relation_out_dim=64, dropout=0.5, mode='pairwise'):
        super().__init__()
        if mode not in ('pairwise', 'control') or min(latent_dim, context_dim, output_dim, relation_dim, relation_out_dim) <= 0 or context_dim % 2:
            raise ValueError('relation dimensions/mode require positive dimensions and even context_dim')
        self.mode = mode
        self.forward_dim = context_dim // 2
        self.local_relation = nn.Linear(latent_dim, relation_dim)
        self.memory_relation = nn.Linear(self.forward_dim, relation_dim)
        pair_input = 4 * relation_dim + 16
        if mode == 'pairwise':
            self.evidence_type = nn.Embedding(4, 16)
            input_dim, width = pair_input, relation_dim
        else:
            input_dim = 2 * relation_dim
            # Match type embedding, LayerNorm and shared MLP parameter budget.
            pair_budget = 64 + 2 * pair_input + (pair_input + 1) * relation_dim + (relation_dim + 1) * relation_out_dim
            width = max(1, round((pair_budget - 2 * input_dim - relation_out_dim) / (input_dim + 1 + relation_out_dim)))
        self.relation_mlp = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, width),
                                          nn.GELU(), nn.Dropout(dropout), nn.Linear(width, relation_out_dim))
        self.relation_out = nn.Linear(relation_out_dim, output_dim)
        nn.init.zeros_(self.relation_out.weight)
        nn.init.zeros_(self.relation_out.bias)
        self.last_diagnostics = {}

    def forward(self, local, base, gap, availability, umask, flat_anchor):
        valid = umask.T.bool()
        if availability.shape != (*valid.shape, 3) or gap.shape != (*valid.shape, 3, 2*self.forward_dim) or base.shape != (*valid.shape, 2*self.forward_dim):
            raise ValueError('incompatible relation input shapes')
        if not torch.all((availability[valid] == 0) | (availability[valid] == 1)):
            raise ValueError('availability must be binary')
        has_history = valid & ((valid.long().cumsum(0) - valid.long()) > 0)
        active = torch.cat((valid[...,None], valid[...,None] & ~availability.bool()), -1)
        active = active & has_history[...,None]
        safe_local = torch.where(has_history[...,None], local, torch.zeros_like(local))
        evidence = torch.cat((base[...,None,:self.forward_dim],gap[...,:self.forward_dim]),2)
        evidence = torch.where(active[...,None],evidence,torch.zeros_like(evidence))
        q = self.local_relation(safe_local)
        k = self.memory_relation(evidence)
        count = active.sum(-1).clamp_min(1)[...,None]
        slot_relation = None
        # Extra training dropout must not perturb downstream baseline RNG.
        devices = [local.device.index] if local.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            if self.mode == 'pairwise':
                q = q[...,None,:].expand_as(k)
                types = self.evidence_type.weight.view(1,1,4,16).expand(*k.shape[:-1],16)
                z = torch.cat((q,k,q*k,(q-k).abs(),types),-1)
                z = torch.where(active[...,None],z,torch.zeros_like(z))
                slot_relation = self.relation_mlp(z)
                slot_relation = torch.where(active[...,None],slot_relation,torch.zeros_like(slot_relation))
                relation = slot_relation.sum(2) / count
            else:
                pooled = torch.where(active[...,None],k,torch.zeros_like(k)).sum(2) / count
                z = torch.cat((q,pooled),-1)
                z = torch.where(has_history[...,None],z,torch.zeros_like(z))
                relation = self.relation_mlp(z)
        relation = torch.where(has_history[...,None],relation,torch.zeros_like(relation))
        residual = self.relation_out(relation)
        residual = torch.where(has_history[...,None],residual,torch.zeros_like(residual))
        with torch.no_grad():
            n = int(valid.sum())
            self.last_diagnostics = {
                'valid_count': n, 'history_count': int(has_history.sum()),
                'relation_residual_norm': float(residual[valid].norm(dim=-1).mean()) if n else 0.,
                'relation_anchor_norm_ratio': float((residual[valid].norm(dim=-1) / flat_anchor[valid].norm(dim=-1).clamp_min(1e-8)).mean()) if n else 0.,
                'active_evidence_count_mean': float(active.sum(-1)[valid].float().mean()) if n else 0.,
                'per_slot_relation_norm': {}, 'active_counts': {},
            }
            for i,name in enumerate(('base','gap_audio','gap_text','gap_visual')):
                selected = active[...,i]
                self.last_diagnostics['active_counts'][name] = int(selected.sum())
                self.last_diagnostics['per_slot_relation_norm'][name] = float(slot_relation[...,i,:][selected].norm(dim=-1).mean()) if slot_relation is not None and selected.any() else 0.
        return residual


class GapIncrementFilter(nn.Module):
    """Scalar modulation of the shared adapter's full-minus-base increment."""

    def __init__(self, output_dim):
        super().__init__()
        self.base_norm = nn.LayerNorm(output_dim)
        self.delta_norm = nn.LayerNorm(output_dim)
        self.gate_mlp = nn.Sequential(nn.Linear(2 * output_dim + 3, 128),
                                      nn.GELU(), nn.Linear(128, 1))
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.zeros_(self.gate_mlp[-1].bias)
        self.last_diagnostics = {}

    def forward(self, full_anchor, base_anchor, delta, availability, valid):
        mask = valid.unsqueeze(-1)
        base_anchor = torch.where(mask, base_anchor, torch.zeros_like(base_anchor))
        delta = torch.where(mask, delta, torch.zeros_like(delta))
        available = torch.where(mask, availability, torch.zeros_like(availability))
        gate = 1 + torch.tanh(self.gate_mlp(torch.cat((
            self.base_norm(base_anchor), self.delta_norm(delta), available), -1)))
        modulation = (gate - 1) * delta
        with torch.no_grad():
            n = int(valid.sum())
            g = gate[valid]
            mod_norm = modulation[valid].norm(dim=-1)
            self.last_diagnostics = {
                'valid_tokens': n,
                'gate_mean': float(g.mean()) if n else 0.,
                'gate_saturation_fraction': float(((g < .1) | (g > 1.9)).float().mean()) if n else 0.,
                'gate_low_fraction': float((g < .1).float().mean()) if n else 0.,
                'gate_high_fraction': float((g > 1.9).float().mean()) if n else 0.,
                'delta_norm': float(delta[valid].norm(dim=-1).mean()) if n else 0.,
                'modulation_norm': float(mod_norm.mean()) if n else 0.,
                'modulation_full_anchor_ratio': float((mod_norm / full_anchor[valid].norm(dim=-1).clamp_min(1e-8)).mean()) if n else 0.,
            }
        return torch.where(mask, full_anchor + modulation, torch.zeros_like(full_anchor))


class OSRAMBackbone(nn.Module):
    """Bidirectional slot-conditioned associative memory.

    Parameters follow the experiment specification: four heads with 32-wide
    keys and values, a 256-dimensional context after concatenating the two
    directions, and a fixed 500-dimensional emotion representation.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        output_dim: int = 500,
        num_heads: int = 4,
        key_dim: int = 32,
        value_dim: int = 32,
        n_speakers: int = 1,
        dropout: float = 0.5,
        read_ridge: float = 1e-3,
        write_ridge: float = 1e-3,
        alpha_init: float = 0.98,
        beta_init: float = 0.5,
        osram_ablation: str = "full",
        query_use_availability: bool = True,
        bidirectional: bool = True,
        forward_slot_reuse: bool = False,
        write_step: float = 1.0,
        osram_emotion_ablation: str = "full",
        osram_readout_fusion: str = "flat",
        osram_gap_read: str = "residual",
        gap_residual_strength: float = 1.0,
        beta_mode: str = "embedded",
        history_query_adapter: bool = False,
        osram_post_grn: bool = False,
        osram_local_skip_gate: bool = False,
        osram_memory_only_adapter: bool = False,
        osram_history_input_gate: bool = False,
        osram_local_evidence_gate: bool = False,
        osram_hierarchical_evidence_gate: bool = False,
        osram_hierarchical_feature_only: bool = False,
        osram_shift_filter_width: int = 128,
        osram_shift_filter_depth: int = 1,
        osram_relation_block: bool = False,
        osram_relation_dual_readout: bool = False,
        osram_relation_mode: str = 'pairwise',
        osram_relation_dim: int = 128,
        osram_relation_out_dim: int = 64,
        osram_gap_increment_filter: bool = False,
        osram_decision_correction: bool = False,
        osram_readout_candidate: str = 'none',
        osram_meaningful_block: str = 'none',
    ) -> None:
        super().__init__()
        self.osram_meaningful_block = osram_meaningful_block
        self.meaningful_input_mode = False
        if osram_meaningful_block != 'none':
            from .meaningful_blocks import MEANINGFUL_METHODS
            from .meaningful_input import INPUT_METHODS
            self.meaningful_input_mode = osram_meaningful_block in INPUT_METHODS
            if (osram_meaningful_block not in MEANINGFUL_METHODS or osram_readout_candidate != 'none'
                    or (num_heads,value_dim) != (8,64)):
                raise ValueError('meaningful block requires original cfg84 heads and a known independent method')
        self.osram_readout_candidate = osram_readout_candidate
        if (self.osram_readout_candidate != 'none' or osram_meaningful_block != 'none') and (
            bidirectional or forward_slot_reuse or osram_readout_fusion != 'flat'
            or osram_post_grn or osram_local_skip_gate or osram_memory_only_adapter
            or osram_history_input_gate or osram_local_evidence_gate
            or osram_hierarchical_evidence_gate or osram_hierarchical_feature_only
            or history_query_adapter or osram_relation_block or osram_relation_dual_readout
            or osram_gap_increment_filter or osram_decision_correction
        ):
            raise ValueError('readout candidates require causal Flat without other adaptations')
        self.osram_decision_correction = bool(osram_decision_correction)
        self.last_decision_evidence = None
        if self.osram_decision_correction and (
            bidirectional or forward_slot_reuse or osram_readout_fusion != 'flat'
            or osram_post_grn or osram_local_skip_gate or osram_memory_only_adapter
            or osram_history_input_gate or osram_local_evidence_gate
            or osram_hierarchical_evidence_gate or osram_hierarchical_feature_only
            or history_query_adapter or osram_relation_block or osram_relation_dual_readout
            or osram_gap_increment_filter
        ):
            raise ValueError('decision correction requires causal Flat without other adaptations')
        self.osram_gap_increment_filter = bool(osram_gap_increment_filter)
        if self.osram_gap_increment_filter and (
            bidirectional or forward_slot_reuse or osram_readout_fusion != 'flat'
            or osram_post_grn or osram_local_skip_gate or osram_memory_only_adapter
            or osram_history_input_gate or osram_local_evidence_gate
            or osram_hierarchical_evidence_gate or osram_hierarchical_feature_only
            or history_query_adapter or osram_relation_block or osram_relation_dual_readout
            or osram_ablation != 'full' or osram_emotion_ablation != 'full'
        ):
            raise ValueError('Gap increment filter requires original causal Flat without other adaptations')
        self.osram_relation_block = bool(osram_relation_block)
        self.osram_relation_dual_readout = bool(osram_relation_dual_readout)
        self.last_relation_base_hidden = None
        if self.osram_relation_dual_readout and not self.osram_relation_block:
            raise ValueError('dual readout requires relation block')
        if self.osram_relation_block and (bidirectional or forward_slot_reuse or osram_readout_fusion != 'flat' or osram_post_grn or osram_local_skip_gate or osram_memory_only_adapter or osram_history_input_gate or osram_local_evidence_gate or osram_hierarchical_evidence_gate or history_query_adapter):
            raise ValueError('relation block requires causal Flat without other adaptations')
        if osram_post_grn and osram_readout_fusion != "flat":
            raise ValueError("osram_post_grn requires flat readout")
        self.osram_post_grn = bool(osram_post_grn)
        self.osram_local_skip_gate = bool(osram_local_skip_gate)
        self.osram_memory_only_adapter = bool(osram_memory_only_adapter)
        if self.osram_memory_only_adapter and (bidirectional or osram_readout_fusion != 'flat' or osram_post_grn or history_query_adapter or osram_history_input_gate or osram_local_evidence_gate or osram_hierarchical_evidence_gate or osram_local_skip_gate):
            raise ValueError('memory-only adapter requires causal Flat without other adaptations')
        if self.osram_local_skip_gate and (bidirectional or osram_readout_fusion != 'flat' or osram_post_grn or history_query_adapter or osram_history_input_gate or osram_local_evidence_gate or osram_hierarchical_evidence_gate):
            raise ValueError('local-skip gate requires causal Flat without other adaptations')
        self.osram_history_input_gate = bool(osram_history_input_gate)
        self.osram_local_evidence_gate = bool(osram_local_evidence_gate)
        self.osram_hierarchical_evidence_gate = bool(osram_hierarchical_evidence_gate)
        if osram_hierarchical_feature_only and not self.osram_hierarchical_evidence_gate:
            raise ValueError('hierarchical feature-only requires hierarchical-evidence gate')
        if self.osram_hierarchical_evidence_gate and (bidirectional or osram_readout_fusion != 'flat' or osram_post_grn or history_query_adapter or osram_history_input_gate or osram_local_evidence_gate):
            raise ValueError('hierarchical-evidence gate requires Flat without other adaptations')
        if self.osram_local_evidence_gate and (osram_readout_fusion != 'flat' or osram_post_grn or history_query_adapter or osram_history_input_gate):
            raise ValueError('local-evidence gate requires Flat without other readout/query adaptations')
        if self.osram_history_input_gate and (osram_readout_fusion != 'flat' or osram_post_grn or history_query_adapter):
            raise ValueError('history-input gate requires flat without post-GRN or query adaptation')
        if osram_readout_fusion not in (
            "flat", "local-gated", "local-cross-attn", "modality-tracks",
            "modality-track-residual", "base-gap-delta", "memory-shift-residual"
        ):
            raise ValueError(
                "osram_readout_fusion must be flat, local-gated, local-cross-attn, "
                "modality-tracks, modality-track-residual, base-gap-delta, or memory-shift-residual"
            )
        if osram_readout_fusion != "flat" and (osram_ablation != "full" or osram_emotion_ablation != "full"):
            raise ValueError("local-gated cannot combine with readout ablations")
        if osram_ablation not in OSRAM_ABLATIONS:
            raise ValueError(
                "osram_ablation must be 'full', 'local-only', or 'local-base'"
            )
        if osram_gap_read not in OSRAM_GAP_READ_MODES:
            raise ValueError("osram_gap_read must be 'residual' or 'raw'")
        if beta_mode not in OSRAM_BETA_MODES:
            raise ValueError("beta_mode must be 'embedded' or 'external-head'")
        gap_residual_strength = float(gap_residual_strength)
        if not math.isfinite(gap_residual_strength) or not 0.0 <= gap_residual_strength <= 1.0:
            raise ValueError("gap_residual_strength must be finite and between zero and one")
        if osram_emotion_ablation not in OSRAM_EMOTION_ABLATIONS:
            raise ValueError("unsupported osram_emotion_ablation")
        if osram_ablation != "full" and osram_emotion_ablation != "full":
            raise ValueError("legacy and emotion-only OSRAM ablations cannot be combined")
        integer_values = {
            "latent_dim": latent_dim,
            "output_dim": output_dim,
            "num_heads": num_heads,
            "key_dim": key_dim,
            "value_dim": value_dim,
            "n_speakers": n_speakers,
        }
        if any(int(value) <= 0 for value in integer_values.values()):
            raise ValueError("OSRAM dimensions must be positive")
        if float(read_ridge) <= 0.0 or float(write_ridge) <= 0.0:
            raise ValueError("OSRAM ridge values must be positive")
        write_step = float(write_step)
        if not math.isfinite(write_step) or not 0.0 <= write_step <= 1.0:
            raise ValueError("write_step must be finite and between zero and one")
        self.latent_dim = int(latent_dim)
        self.output_dim = int(output_dim)
        self.num_heads = int(num_heads)
        self.key_dim = int(key_dim)
        self.value_dim = int(value_dim)
        self.n_speakers = int(n_speakers)
        self.read_ridge = float(read_ridge)
        self.write_ridge = float(write_ridge)
        self.write_step = write_step
        self.osram_ablation = osram_ablation
        self.osram_emotion_ablation = osram_emotion_ablation
        self.osram_readout_fusion = osram_readout_fusion
        self.osram_gap_read = osram_gap_read
        self.gap_residual_strength = gap_residual_strength
        self.beta_mode = beta_mode
        self.query_use_availability = bool(query_use_availability)
        self.bidirectional = bool(bidirectional)
        self.history_query_adapter = bool(history_query_adapter)
        if self.history_query_adapter and self.bidirectional:
            raise ValueError("history query adaptation requires a causal scan")
        self.forward_slot_reuse = bool(forward_slot_reuse)
        if self.bidirectional and self.forward_slot_reuse:
            raise ValueError("forward_slot_reuse requires bidirectional=False")
        self.context_dim = 2 * self.num_heads * self.value_dim

        self.latent_norm = nn.LayerNorm(self.latent_dim)
        self.node_norm = nn.LayerNorm(self.latent_dim)
        self.availability_embedding = nn.Linear(3, self.latent_dim)
        self.speaker_embedding = nn.Embedding(self.n_speakers, self.latent_dim)
        self.query_type_embedding = nn.Embedding(4, self.latent_dim)

        key_input_dim = 4 * self.latent_dim
        self.key_projectors = nn.ModuleDict(
            {
                name: nn.Linear(key_input_dim, self.num_heads * self.key_dim)
                for name in MODALITIES
            }
        )
        self.value_projectors = nn.ModuleDict(
            {
                name: nn.Linear(
                    self.latent_dim, self.num_heads * self.value_dim
                )
                for name in MODALITIES
            }
        )
        query_input_dim = 4 * self.latent_dim
        self.query_projector = nn.Linear(
            query_input_dim, self.num_heads * self.key_dim
        )

        self.alpha_logits = nn.Parameter(
            torch.full((self.num_heads,), _logit(alpha_init))
        )
        self.beta_logits = nn.Parameter(
            torch.full((self.num_heads, len(MODALITIES)), _logit(beta_init))
        )

        local_hidden = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.local_path = local_hidden
        self.emotion_adapter = nn.Sequential(
            nn.LayerNorm(self.latent_dim + self.context_dim + 3 * self.context_dim),
            nn.Linear(
                self.latent_dim + self.context_dim + 3 * self.context_dim,
                self.output_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.output_dim, self.output_dim),
        )
        self.local_skip = nn.Linear(self.latent_dim, self.output_dim)
        self.emotion_norm = nn.LayerNorm(self.output_dim)

        # A zero final adapter leaves the local skip as a stable initial path,
        # while still giving the contextual branch a real gradient.
        nn.init.zeros_(self.emotion_adapter[-1].weight)
        nn.init.zeros_(self.emotion_adapter[-1].bias)

        if self.osram_decision_correction:
            # Preserve legacy state keys and initialization; these replaced
            # modules never execute and are excluded by the optimizer builder.
            self.emotion_adapter.requires_grad_(False)
            self.local_skip.requires_grad_(False)
            self.emotion_norm.requires_grad_(False)

        if self.osram_memory_only_adapter:
            # Keep the original initialization stream for Skip and all later
            # modules. Only the adapter input shape changes in this ablation.
            with torch.random.fork_rng(devices=[]):
                self.emotion_adapter = nn.Sequential(
                    nn.LayerNorm(4 * self.context_dim),
                    nn.Linear(4 * self.context_dim, self.output_dim),
                    nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(self.output_dim, self.output_dim),
                )
                nn.init.zeros_(self.emotion_adapter[-1].weight)
                nn.init.zeros_(self.emotion_adapter[-1].bias)

        if osram_readout_fusion != "flat":
            # Preserve RNG for downstream Student/Teacher/MMoE construction.
            with torch.random.fork_rng(devices=[]):
                if osram_readout_fusion == "local-gated":
                    self.local_centered_fusion = LocalCenteredContextFusion(
                        self.latent_dim, self.context_dim, self.output_dim, dropout=dropout)
                elif osram_readout_fusion == "local-cross-attn":
                    self.local_centered_fusion = LocalCrossAttentionFusion(
                        self.latent_dim, self.context_dim, self.output_dim, dropout=dropout)
                elif osram_readout_fusion == "modality-tracks":
                    self.modality_track_fusion = ModalityTrackFlatFusion(
                        self.latent_dim, self.context_dim, self.output_dim, dropout=dropout)
                elif osram_readout_fusion == "modality-track-residual":
                    self.modality_track_residual = ModalityTrackResidualFusion(
                        self.latent_dim, self.output_dim, dropout=dropout)
                elif osram_readout_fusion == "memory-shift-residual":
                    self.memory_shift_filter = MemoryShiftFilter(
                        self.latent_dim, self.context_dim, self.output_dim,
                        filter_width=osram_shift_filter_width,
                        filter_depth=osram_shift_filter_depth)
                else:
                    self.base_gap_delta_fusion = BaseGapDeltaFusion(
                        self.latent_dim, self.context_dim, self.output_dim, dropout=dropout)
            if osram_readout_fusion not in (
                "modality-tracks", "modality-track-residual", "base-gap-delta", "memory-shift-residual"
            ):
                self.local_centered_fusion.local_skip.load_state_dict(self.local_skip.state_dict())
                self.local_centered_fusion.emotion_norm.load_state_dict(self.emotion_norm.state_dict())
            # Historical flat keys remain available for readout interventions.
            # The residual mode intentionally keeps them trainable by default;
            # the frozen-backbone experiment freezes them explicitly in the trainer.
            if osram_readout_fusion not in ("modality-track-residual", "base-gap-delta", "memory-shift-residual"):
                self.emotion_adapter.requires_grad_(False)
                self.local_skip.requires_grad_(False)
                self.emotion_norm.requires_grad_(False)

        if self.osram_hierarchical_evidence_gate:
            with torch.random.fork_rng(devices=[]):
                self.hierarchical_evidence_gate = HierarchicalEvidenceGate(
                    self.latent_dim, self.context_dim, feature_only=osram_hierarchical_feature_only)
        if self.osram_local_evidence_gate:
            with torch.random.fork_rng(devices=[]):
                self.local_evidence_gate = LocalConditionedEvidenceGate(self.latent_dim, self.context_dim)
        if self.osram_local_skip_gate:
            with torch.random.fork_rng(devices=[]):
                self.local_skip_gate = HistoryInputGate(self.latent_dim, self.context_dim)
        if self.osram_history_input_gate:
            with torch.random.fork_rng(devices=[]):
                self.history_input_gate = HistoryInputGate(self.latent_dim, self.context_dim)
        if self.osram_post_grn:
            # Leave the original Flat and memory trainable, and preserve shared
            # initialization / downstream head RNG independently of the switch.
            with torch.random.fork_rng(devices=[]):
                self.post_grn = PostGRN(self.latent_dim, self.context_dim, self.output_dim, dropout)
        self.last_diagnostics: dict[str, object] = {}
        if self.osram_meaningful_block != 'none':
            from .meaningful_blocks import MeaningfulReadoutResidual
            from .meaningful_input import MeaningfulInputAdapter
            with torch.random.fork_rng(devices=[]):
                if self.osram_meaningful_block == 'm30_dyt':
                    from .priority40_common import DynamicTanh, NormalizationOnlyReadout
                    original_norm = self.emotion_adapter[0]
                    self.emotion_adapter[0] = DynamicTanh(original_norm.normalized_shape[0])
                    with torch.no_grad():
                        self.emotion_adapter[0].weight.copy_(original_norm.weight)
                        self.emotion_adapter[0].bias.copy_(original_norm.bias)
                    self.meaningful_block = NormalizationOnlyReadout()
                else:
                    factory = MeaningfulInputAdapter if self.meaningful_input_mode else MeaningfulReadoutResidual
                    self.meaningful_block = factory(
                        self.latent_dim,self.context_dim,self.output_dim,self.osram_meaningful_block,
                        self.num_heads,self.value_dim)
        if self.osram_readout_candidate != 'none':
            from .readout_candidates import ExternalReadoutResidual
            with torch.random.fork_rng(devices=[]):
                self.readout_candidate = ExternalReadoutResidual(
                    self.latent_dim, self.context_dim, self.output_dim, self.osram_readout_candidate)
        if self.osram_gap_increment_filter:
            with torch.random.fork_rng(devices=[]):
                self.gap_increment_filter = GapIncrementFilter(self.output_dim)
        if self.osram_relation_block:
            with torch.random.fork_rng(devices=[]):
                self.relation_block = CurrentHistoryRelationBlock(
                    self.latent_dim, self.context_dim, self.output_dim,
                    osram_relation_dim, osram_relation_out_dim, dropout, osram_relation_mode)
        if self.history_query_adapter:
            # New parameters must not perturb existing initialization or the
            # downstream classifier's RNG sequence.
            with torch.random.fork_rng(devices=[]):
                self.history_support_embedding = nn.Embedding(8, 16)
                self.history_query_adapters = nn.ModuleDict({
                    name: nn.Sequential(
                        nn.Linear(self.key_dim + 16, 64), nn.GELU(),
                        nn.Linear(64, self.key_dim),
                    ) for name in MODALITIES
                })
                for adapter in self.history_query_adapters.values():
                    nn.init.zeros_(adapter[-1].weight)
                    nn.init.zeros_(adapter[-1].bias)

    def _adapt_history_queries(self, queries, availability, valid):
        """Adapt Gap addresses using strictly past, real-observation support.

        Support bits are A/T/V, encoded with weights 1/2/4. The current
        utterance is excluded because the causal scan reads before writing.
        Counts are local tensors, reset for every conversation/forward call.
        """
        observed = availability.bool() & valid.unsqueeze(-1)
        counts = observed.long().cumsum(dim=0) - observed.long()
        support = counts > 0
        # CUDA does not implement integer matrix-vector multiplication.
        codes = (support.long() * support.new_tensor([1, 2, 4], dtype=torch.long)).sum(-1)
        active = ~support & valid.unsqueeze(-1)
        self.last_history_support = support.detach()
        self.last_history_codes = codes.detach()
        self.last_history_active = active.detach()
        embedding = self.history_support_embedding(codes)
        embedding = embedding.unsqueeze(-2).expand(-1, -1, self.num_heads, -1)
        adapted = [queries[:, :, 0]]
        for index, name in enumerate(MODALITIES):
            query = queries[:, :, index + 1]
            delta = self.history_query_adapters[name](torch.cat((query, embedding), dim=-1))
            normalized = F.normalize(query + delta, dim=-1)
            adapted.append(torch.where(active[:, :, index, None, None], normalized, query))
        return torch.stack(adapted, dim=2)

    @staticmethod
    def _validate_inputs(
        node: torch.Tensor,
        latents: Mapping[str, torch.Tensor],
        availability: torch.Tensor,
        qmask: torch.Tensor,
        umask: torch.Tensor,
        seq_lengths: Sequence[int] | None,
    ) -> torch.Tensor:
        if node.ndim != 3:
            raise ValueError("node must have shape [L, B, latent_dim]")
        length, batch = node.shape[:2]
        if set(latents) != set(MODALITIES):
            raise ValueError("latents must contain audio, text, and visual")
        for value in latents.values():
            if value.shape != node.shape:
                raise ValueError("all modality latents must match node shape")
        if availability.shape != (length, batch, 3):
            raise ValueError("availability must have shape [L, B, 3]")
        if qmask.shape != (batch, length):
            raise ValueError("qmask must have shape [B, L]")
        if umask.shape != (batch, length):
            raise ValueError("umask must have shape [B, L]")
        if not bool(((availability == 0) | (availability == 1)).all()):
            raise ValueError("availability must be binary")
        if not bool(((umask == 0) | (umask == 1)).all()):
            raise ValueError("umask must be binary")
        valid = umask.T.bool()
        if bool((availability[~valid] != 0).any()):
            raise ValueError("padding availability must be zero")
        if bool((availability[valid].sum(dim=-1) == 0).any()):
            raise ValueError("valid utterances require an observed modality")
        if seq_lengths is not None:
            lengths = tuple(int(value) for value in seq_lengths)
            if len(lengths) != batch:
                raise ValueError("seq_lengths must contain one value per batch")
            expected = umask.sum(dim=1).long().tolist()
            if lengths != tuple(int(value) for value in expected):
                raise ValueError("seq_lengths and umask disagree")
            if any(value < 1 or value > length for value in lengths):
                raise ValueError("seq_lengths must be within the sequence")
        return valid

    def _project_sequence(
        self,
        node: torch.Tensor,
        latents: Mapping[str, torch.Tensor],
        availability: torch.Tensor,
        qmask: torch.Tensor,
        *,
        read_node: torch.Tensor | None = None,
        write_node: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
        read_node = node if read_node is None else read_node
        write_node = node if write_node is None else write_node
        length, batch = node.shape[:2]
        dtype = node.dtype
        valid_speaker = qmask.to(device=node.device).long()
        if bool((valid_speaker < 0).any()) or bool(
            (valid_speaker >= self.n_speakers).any()
        ):
            raise ValueError("qmask contains a speaker outside n_speakers")
        speaker = self.speaker_embedding(valid_speaker.T)
        availability_value = availability.to(dtype=dtype)
        availability_embed = self.availability_embedding(availability_value)
        write_node_value = self.node_norm(write_node)
        # Share the original normalization graph when the two inputs coincide:
        # the default must preserve both outputs and gradient accumulation exactly.
        read_node_value = (
            write_node_value if read_node is write_node else self.node_norm(read_node)
        )
        keys: dict[str, torch.Tensor] = {}
        values: dict[str, torch.Tensor] = {}
        for name in MODALITIES:
            latent_value = self.latent_norm(latents[name])
            key_input = torch.cat(
                (latent_value, write_node_value, availability_embed, speaker), dim=-1
            )
            key = self.key_projectors[name](key_input).view(
                length, batch, self.num_heads, self.key_dim
            )
            keys[name] = F.normalize(key, dim=-1)
            value = self.value_projectors[name](latent_value).view(
                length, batch, self.num_heads, self.value_dim
            )
            values[name] = value

        # Keep the query projector width fixed for checkpoint compatibility.
        # The ablation removes only the explicit availability condition from
        # the query; keys, values, and the hard missing-slot readout remain
        # unchanged.
        query_availability = (
            availability_embed
            if self.query_use_availability
            else torch.zeros_like(availability_embed)
        )
        common_query = torch.cat(
            (read_node_value, query_availability, speaker), dim=-1
        )
        type_embedding = self.query_type_embedding.weight.view(1, 1, 4, -1)
        query_input = torch.cat(
            (
                common_query.unsqueeze(2).expand(-1, -1, 4, -1),
                type_embedding.expand(length, batch, -1, -1),
            ),
            dim=-1,
        )
        queries = self.query_projector(query_input).view(
            length, batch, 4, self.num_heads, self.key_dim
        )
        return keys, values, F.normalize(queries, dim=-1)

    def _address_residual(
        self, keys: torch.Tensor, query: torch.Tensor
    ) -> torch.Tensor:
        """Return ``(I - K(KᵀK+λI)⁻¹Kᵀ)query`` without an explicit inverse."""

        # keys: [B, H, d_k, 3], query: [B, H, d_k]
        gram = keys.transpose(-1, -2) @ keys
        identity = torch.eye(
            gram.shape[-1], device=gram.device, dtype=gram.dtype
        )
        system = gram + self.read_ridge * identity
        rhs = keys.transpose(-1, -2) @ query.unsqueeze(-1)
        coefficients = torch.linalg.solve(system, rhs).squeeze(-1)
        return query - (keys @ coefficients.unsqueeze(-1)).squeeze(-1)

    def block_write(
        self,
        memory: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        availability: torch.Tensor,
        beta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply one permutation-invariant, multi-slot block-delta update."""

        if memory.ndim != 4 or keys.ndim != 4 or values.ndim != 4:
            raise ValueError("memory, keys, and values must be rank four")
        if keys.shape[-1] != len(MODALITIES) or values.shape[-1] != len(
            MODALITIES
        ):
            raise ValueError("the last dimension must enumerate A/T/V slots")
        if availability.ndim != 2 or availability.shape[-1] != len(MODALITIES):
            raise ValueError("availability must have shape [B, 3]")
        if beta is None:
            beta = torch.sigmoid(self.beta_logits)
        beta = beta.to(device=keys.device, dtype=keys.dtype)
        if beta.shape != (self.num_heads, len(MODALITIES)):
            raise ValueError("beta must have shape [H, 3]")

        # Availability is broadcast over heads.  A zero column is a true
        # absent slot; it cannot contribute to either side of the solve.
        availability_value = availability.to(device=keys.device, dtype=keys.dtype)
        # Factor the binary availability outside the square root.  Computing
        # ``sqrt(availability * beta)`` gives an undefined derivative at
        # absent slots (0 * inf) even though those slots have zero write
        # strength.  The factored form is algebraically identical for the
        # binary mask and keeps gradients finite for missing/padded slots.
        if self.beta_mode == "embedded":
            slot_scale = (
                availability_value.unsqueeze(1)
                * beta.clamp_min(0.0).sqrt().unsqueeze(0)
            ).unsqueeze(2)
        else:
            # Diagnostic alternative: solve the block update using the raw
            # observed addresses, then apply beta directly to the correction.
            # For a single observed slot this is exactly beta_hm; with two or
            # three slots it is the availability-weighted head mean.
            slot_scale = availability_value.unsqueeze(1).unsqueeze(2)
        k_bar = keys * slot_scale
        v_bar = values * slot_scale
        residual = v_bar - memory @ k_bar
        gram = k_bar.transpose(-1, -2) @ k_bar
        identity = torch.eye(
            gram.shape[-1], device=gram.device, dtype=gram.dtype
        )
        system = gram + self.write_ridge * identity
        solved = torch.linalg.solve(system, k_bar.transpose(-1, -2))
        correction = residual @ solved
        if self.beta_mode == "external-head":
            active_beta = (
                availability_value.unsqueeze(1) * beta.unsqueeze(0)
            ).sum(dim=-1)
            active_count = availability_value.sum(dim=-1).clamp_min(1.0)
            correction = correction * (active_beta / active_count.unsqueeze(-1)).view(
                -1, self.num_heads, 1, 1
            )
        # Keep the legacy arithmetic path exact at the default step. The fixed
        # scalar scales the correction, without detaching the write gradients.
        if self.write_step == 1.0:
            return memory + correction
        return memory + self.write_step * correction

    def _scan(
        self,
        keys: Mapping[str, torch.Tensor],
        values: Mapping[str, torch.Tensor],
        queries: torch.Tensor,
        availability: torch.Tensor,
        valid: torch.Tensor,
        reverse: bool,
        retention_diagnostics=None,
        write_completion=None,
        post_write_observer=None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, list[float]]]:
        if hasattr(self, 'core20_memory'):
            if reverse or any(x is not None for x in (retention_diagnostics, write_completion, post_write_observer)):
                raise ValueError('core20 memory requires an independent observed-only forward scan')
            strength = 0. if self.osram_gap_read == 'raw' else self.gap_residual_strength
            def address_residual(k, q):
                return q - strength * (q - self._address_residual(k, q))
            return self.core20_memory.scan(keys, values, queries, availability, valid,
                                           address_residual=address_residual)
        length, batch = availability.shape[:2]
        dtype = queries.dtype
        memory = queries.new_zeros(
            batch, self.num_heads, self.value_dim, self.key_dim
        )
        base = queries.new_zeros(length, batch, self.num_heads * self.value_dim)
        gap = queries.new_zeros(
            length, batch, len(MODALITIES), self.num_heads * self.value_dim
        )
        diagnostics = {
            name: {"rho": [], "eta": [], "cosine": []}
            for name in MODALITIES
        }
        alpha = torch.sigmoid(self.alpha_logits).to(dtype=dtype)
        beta = torch.sigmoid(self.beta_logits).to(dtype=dtype)
        identity_mask = valid.to(dtype=dtype)
        order = range(length - 1, -1, -1) if reverse else range(length)
        for time_index in order:
            active = valid[time_index]
            if not bool(active.any()):
                continue
            slot_keys = torch.stack(
                [keys[name][time_index] for name in MODALITIES], dim=-1
            )
            slot_values = torch.stack(
                [values[name][time_index] for name in MODALITIES], dim=-1
            )
            if retention_diagnostics is not None:
                probe_pre = memory.detach()
                probe_keys, probe_values = slot_keys.detach(), slot_values.detach()
            # Stacking along the final dimension yields [B,H,d,3].
            slot_keys = slot_keys * availability[time_index].to(dtype).unsqueeze(
                1
            ).unsqueeze(2)
            slot_values = slot_values * availability[time_index].to(dtype).unsqueeze(
                1
            ).unsqueeze(2)
            decayed = memory * alpha.view(1, self.num_heads, 1, 1)
            memory = torch.where(active.view(batch, 1, 1, 1), decayed, memory)
            read_memory = memory

            base_read = (
                read_memory
                @ queries[time_index, :, 0].unsqueeze(-1)
            ).squeeze(-1)
            base[time_index] = base_read.reshape(batch, -1) * identity_mask[time_index].unsqueeze(-1)

            missing = 1.0 - availability[time_index].to(dtype)
            for modality_index, name in enumerate(MODALITIES):
                query = queries[time_index, :, modality_index + 1]
                projected_query = self._address_residual(slot_keys, query)
                residual_strength = (
                    0.0
                    if self.osram_gap_read == "raw"
                    else self.gap_residual_strength
                )
                residual_query = query - residual_strength * (query - projected_query)
                original_norm = query.norm(dim=-1)
                residual_norm = residual_query.norm(dim=-1)
                cosine = F.cosine_similarity(
                    query, residual_query, dim=-1, eps=1e-8
                )
                attenuation = 1.0 - cosine * residual_norm / original_norm.clamp_min(
                    1e-8
                )
                valid_values = active.view(batch, 1).expand(-1, self.num_heads)
                diagnostics[name]["rho"].extend(
                    (residual_norm / original_norm.clamp_min(1e-8))[valid_values]
                    .detach()
                    .cpu()
                    .tolist()
                )
                diagnostics[name]["eta"].extend(
                    attenuation[valid_values].detach().cpu().tolist()
                )
                diagnostics[name]["cosine"].extend(
                    cosine[valid_values].detach().cpu().tolist()
                )
                read = (
                    read_memory @ residual_query.unsqueeze(-1)
                ).squeeze(-1)
                gap[time_index, :, modality_index] = (
                    read.reshape(batch, -1)
                    * missing[:, modality_index].unsqueeze(-1)
                    * identity_mask[time_index].unsqueeze(-1)
                )

            write_keys, write_values, write_mask = slot_keys, slot_values, availability[time_index]
            if write_completion is not None:
                write_keys, write_values, write_mask = write_completion(
                    time_index, base_read.reshape(batch,-1).clone(),
                    gap[time_index].clone(), slot_keys, slot_values)
            memory_candidate = self.block_write(
                read_memory,
                write_keys,
                write_values,
                write_mask,
                beta=beta,
            )
            memory = torch.where(
                active.view(batch, 1, 1, 1), memory_candidate, memory
            )
            if post_write_observer is not None:
                # Differentiable read-only observer: its return is ignored and
                # a clone prevents callback mutation of persistent memory.
                post_write_observer(time_index, memory.clone(), active.clone())
            if retention_diagnostics is not None:
                retention_diagnostics.observe(time_index, probe_pre, read_memory,
                                              memory, probe_keys, probe_values,
                                              availability[time_index], active)
        return base, gap, diagnostics

    @staticmethod
    def _summarize(values: list[float]) -> float | None:
        if not values:
            return None
        return float(sum(values) / len(values))

    def forward(
        self,
        node: torch.Tensor,
        latents: Mapping[str, torch.Tensor],
        availability: torch.Tensor,
        qmask: torch.Tensor,
        umask: torch.Tensor,
        seq_lengths: Sequence[int] | None = None,
        *,
        collect_memory_retention_diagnostics: bool = False,
        memory_retention_diagnostics=None,
        read_node: torch.Tensor | None = None,
        write_node: torch.Tensor | None = None,
        write_completion=None,
        post_write_observer=None,
        context_read_residual=None,
        modality_embeddings: torch.Tensor | None = None,
        core20_reconstruction_targets=None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Condition reads/local features separately from real-observation writes.

        Both optional nodes default to ``node``. Completion belongs exclusively
        in ``read_node``; keys and gap-address projection use ``write_node``,
        while values continue to use the supplied real modality latents.
        This describes B2 read completion. The separate opt-in ``write_completion``
        callback instead replaces missing write slots AFTER reads; it never changes
        current queries or the observed addresses used for Gap residualization.
        ``context_read_residual`` is evaluated after the causal memory read and
        before the local path/readout. It can shape the current representation,
        but it cannot change the already-completed persistent write.
        """
        self.last_relation_base_hidden = None
        read_node = node if read_node is None else read_node
        self.last_decision_evidence = None
        if self.osram_decision_correction and (
            write_node is not None or read_node is not node or write_completion is not None
            or context_read_residual is not None or post_write_observer is not None
        ):
            raise ValueError('decision correction requires unchanged observed-only causal memory')
        write_node = node if write_node is None else write_node
        if post_write_observer is not None and self.bidirectional:
            raise ValueError('Post-write state observer requires a causal scan')
        if write_completion is not None and (self.bidirectional or collect_memory_retention_diagnostics):
            raise ValueError('Predicted writes require causal scan without legacy retention collector')
        if collect_memory_retention_diagnostics:
            if self.bidirectional or memory_retention_diagnostics is None:
                raise ValueError('Retention diagnostics require forward-only and an external collector')
            if memory_retention_diagnostics.last_key is not None:
                raise ValueError('Use a fresh retention collector for each forward/batch')
        valid = self._validate_inputs(
            node, latents, availability, qmask, umask, seq_lengths
        )
        keys, values, queries = self._project_sequence(
            node, latents, availability, qmask,
            read_node=read_node, write_node=write_node,
        )
        if self.history_query_adapter:
            if hasattr(self, 'core20_value'):
                raise ValueError('core20 does not combine query adapters with VQ')
            if write_completion is not None:
                raise ValueError("history query adapter requires observed-only writes")
            queries = self._adapt_history_queries(queries, availability, valid)
        if hasattr(self, 'core20_value'):
            values, self.core20_value.auxiliary_loss = self.core20_value.transform_values(
                values, availability, valid, reconstruction_targets=core20_reconstruction_targets)
        base_forward, gap_forward, diag_forward = self._scan(
            keys, values, queries, availability, valid, reverse=False,
            retention_diagnostics=(memory_retention_diagnostics
                                   if collect_memory_retention_diagnostics else None),
            write_completion=write_completion,
            post_write_observer=post_write_observer,
        )
        if self.bidirectional:
            base_backward, gap_backward, diag_backward = self._scan(
                keys, values, queries, availability, valid, reverse=True
            )
        else:
            # Preserve parameter shapes and fusion slots, without future reads.
            base_backward = base_forward if self.forward_slot_reuse else torch.zeros_like(base_forward)
            gap_backward = gap_forward if self.forward_slot_reuse else torch.zeros_like(gap_forward)
            diag_backward = {name: {metric: [] for metric in ("rho", "eta", "cosine")}
                             for name in MODALITIES}
        base_context = torch.cat((base_forward, base_backward), dim=-1)
        gap_context = torch.cat((gap_forward, gap_backward), dim=-1)

        # The ablations disable only the corresponding readout slots.  The
        # scan itself remains identical so the comparison isolates which
        # contextual representation reaches the downstream heads without
        # changing the memory parameterization or training protocol.
        if self.osram_ablation == "local-only":
            active_base_context = torch.zeros_like(base_context)
            active_gap_context = torch.zeros_like(gap_context)
        elif self.osram_ablation == "local-base":
            active_base_context = base_context
            active_gap_context = torch.zeros_like(gap_context)
        else:
            active_base_context = base_context
            active_gap_context = gap_context

        missing = 1.0 - availability.to(dtype=node.dtype)
        # Unlike the legacy switch above, mask ONLY the emotion-fusion inputs.
        # The returned Base/Gap tensors still supervise the structured predictor.
        emotion_base = active_base_context
        emotion_gap = active_gap_context
        if self.osram_emotion_ablation in ("local-only", "local-gap"):
            emotion_base = torch.zeros_like(emotion_base)
        if self.osram_emotion_ablation in ("local-only", "local-base"):
            emotion_gap = torch.zeros_like(emotion_gap)
        local_input = read_node
        if context_read_residual is not None:
            residual = context_read_residual(active_base_context, active_gap_context)
            if not torch.is_tensor(residual) or residual.shape != read_node.shape:
                raise ValueError(
                    "context_read_residual must return [L, B, latent_dim]"
                )
            local_input = read_node + residual
        local = local_input + self.local_path(local_input)
        local = local * valid.unsqueeze(-1).to(local.dtype)
        if hasattr(self, 'core20_readout'):
            hidden = self.core20_readout(local, emotion_base, emotion_gap, availability, umask)
        elif self.osram_decision_correction:
            forward_dim = self.num_heads * self.value_dim
            local = torch.where(valid[..., None], local, torch.zeros_like(local))
            history = valid & (valid.long().cumsum(0) > 1)
            base = emotion_base[..., :forward_dim]
            gap = emotion_gap[..., :forward_dim]
            self.last_decision_evidence = {
                'local': local,
                'base': torch.where(history[..., None], base, torch.zeros_like(base)),
                'gap': torch.where((history[..., None] & ~availability.bool())[..., None], gap, torch.zeros_like(gap)),
            }
            hidden = local
        elif self.osram_readout_fusion == "modality-tracks":
            if modality_embeddings is None:
                raise ValueError("modality-tracks readout requires modality_embeddings")
            if modality_embeddings.shape != (len(MODALITIES), self.latent_dim):
                raise ValueError("modality_embeddings must be [3, latent_dim]")
            tracks = []
            for index, name in enumerate(MODALITIES):
                track_input = latents[name] + modality_embeddings[index].view(1, 1, -1)
                track = self.local_path(track_input)
                track = track * availability[..., index : index + 1].to(track.dtype)
                track = track * valid.unsqueeze(-1).to(track.dtype)
                tracks.append(track)
            track_local = torch.stack(tracks, dim=2)
            hidden = self.modality_track_fusion(
                track_local, base_context, gap_context, availability, umask
            )
        elif self.osram_readout_fusion == "modality-track-residual":
            if modality_embeddings is None:
                raise ValueError("modality-track-residual readout requires modality_embeddings")
            emotion_input = torch.cat(
                (
                    local,
                    emotion_base,
                    (emotion_gap * missing.unsqueeze(-1)).reshape(
                        active_gap_context.shape[0], active_gap_context.shape[1], -1
                    ),
                ),
                dim=-1,
            )
            hidden = self.emotion_norm(
                self.local_skip(local) + self.emotion_adapter(emotion_input)
            )
            hidden = hidden + self.modality_track_residual(
                latents, availability, umask, modality_embeddings
            )
        elif self.osram_readout_fusion == "base-gap-delta":
            hidden = self.base_gap_delta_fusion(
                local,
                emotion_base,
                emotion_gap,
                availability,
                umask,
                emotion_adapter=self.emotion_adapter,
                local_skip=self.local_skip,
                emotion_norm=self.emotion_norm,
            )
        elif self.osram_readout_fusion == "memory-shift-residual":
            # Keep the original flat anchor trainable and unchanged; only add
            # the zero-initialized shift before its existing LayerNorm.
            active_gap = valid.unsqueeze(-1) & ~availability.bool()
            safe_local = torch.where(valid.unsqueeze(-1), local, torch.zeros_like(local))
            safe_base = torch.where(valid.unsqueeze(-1), emotion_base, torch.zeros_like(emotion_base))
            safe_gap = torch.where(active_gap.unsqueeze(-1), emotion_gap, torch.zeros_like(emotion_gap))
            emotion_input = torch.cat((safe_local, safe_base, safe_gap.flatten(2)), dim=-1)
            flat_anchor = self.local_skip(safe_local) + self.emotion_adapter(emotion_input)
            residual = self.memory_shift_filter(
                local, emotion_base, emotion_gap, availability, umask, flat_anchor)
            hidden = self.emotion_norm(flat_anchor + residual)
        elif self.osram_readout_fusion != "flat":
            hidden = self.local_centered_fusion(
                local, base_context, gap_context, availability, umask)
        else:
            # The opt-in path also sanitizes the Flat inputs; the disabled
            # historical path is deliberately byte-for-byte unchanged.
            if self.osram_meaningful_block != 'none' or self.osram_readout_candidate != 'none' or self.osram_gap_increment_filter or self.osram_relation_block or self.osram_memory_only_adapter or self.osram_local_skip_gate or self.osram_post_grn or self.osram_history_input_gate or self.osram_local_evidence_gate or self.osram_hierarchical_evidence_gate:
                local = torch.where(valid.unsqueeze(-1), local, torch.zeros_like(local))
                emotion_base = torch.where(valid.unsqueeze(-1), emotion_base, torch.zeros_like(emotion_base))
                emotion_gap = torch.where(
                    (valid.unsqueeze(-1) & ~availability.bool()).unsqueeze(-1),
                    emotion_gap, torch.zeros_like(emotion_gap))
                if self.osram_gap_increment_filter or self.osram_readout_candidate != 'none' or self.osram_meaningful_block != 'none':
                    # A transposed mask can make where's result noncontiguous.
                    # Preserve the legacy Linear GEMM layout (and CUDA rounding).
                    local = local.contiguous()
            if self.osram_hierarchical_evidence_gate:
                evidence = self.hierarchical_evidence_gate(local, emotion_base, emotion_gap, availability, umask)
                emotion_base, emotion_gap = evidence[..., 0, :], evidence[..., 1:, :]
                missing = torch.where(valid.unsqueeze(-1), missing, torch.zeros_like(missing))
            if self.osram_local_evidence_gate:
                gates = self.local_evidence_gate(local, emotion_base, emotion_gap, availability, umask)
                emotion_base = gates[..., :1] * emotion_base
                emotion_gap = gates[..., 1:, None] * emotion_gap
                missing = torch.where(valid.unsqueeze(-1), missing, torch.zeros_like(missing))
            if self.osram_history_input_gate:
                alpha = self.history_input_gate(local, emotion_base, emotion_gap, availability, umask)
                emotion_base = alpha * emotion_base
                emotion_gap = alpha.unsqueeze(-1) * emotion_gap
                missing = torch.where(valid.unsqueeze(-1), missing, torch.zeros_like(missing))
            readout_local = local
            if self.meaningful_input_mode:
                readout_local, emotion_base, emotion_gap = self.meaningful_block(
                    local,emotion_base,emotion_gap,availability,umask)
            if self.osram_memory_only_adapter:
                emotion_input = torch.cat((emotion_base, emotion_gap.flatten(2)), dim=-1)
            else:
                emotion_input = torch.cat(
                    (
                        readout_local,
                        emotion_base,
                        (emotion_gap * missing.unsqueeze(-1)).reshape(
                            active_gap_context.shape[0], active_gap_context.shape[1], -1
                        ),
                    ),
                    dim=-1,
                )
            if self.meaningful_input_mode:
                hidden = self.emotion_norm(self.local_skip(local) + self.emotion_adapter(emotion_input))
            elif self.osram_meaningful_block != 'none':
                flat_anchor = self.local_skip(local) + self.emotion_adapter(emotion_input)
                residual = self.meaningful_block(
                    local,emotion_base,emotion_gap,availability,umask,flat_anchor)
                hidden = self.emotion_norm(flat_anchor + residual)
            elif self.osram_readout_candidate != 'none':
                flat_anchor = self.local_skip(local) + self.emotion_adapter(emotion_input)
                residual = self.readout_candidate(
                    local, emotion_base, emotion_gap, availability, umask, flat_anchor)
                hidden = self.emotion_norm(flat_anchor + residual)
            elif self.osram_gap_increment_filter:
                # Save the exact adapter dropout stream, then replay it locally.
                # The enclosing stream advances for the original full call only.
                skip = self.local_skip(local)
                cpu_rng = torch.get_rng_state()
                devices = [emotion_input.device.index] if emotion_input.is_cuda else []
                device_rng = torch.cuda.get_rng_state(emotion_input.device) if devices else None
                full_adapter = self.emotion_adapter(emotion_input)
                base_input = torch.cat((local, emotion_base, torch.zeros_like(emotion_gap).flatten(2)), -1)
                with torch.random.fork_rng(devices=devices):
                    torch.set_rng_state(cpu_rng)
                    if devices:
                        torch.cuda.set_rng_state(device_rng, emotion_input.device)
                    base_adapter = self.emotion_adapter(base_input)
                full_anchor = skip + full_adapter
                hidden = self.emotion_norm(self.gap_increment_filter(
                    full_anchor, skip + base_adapter, full_adapter - base_adapter,
                    availability, valid))
            elif self.osram_relation_block:
                flat_anchor = self.local_skip(local) + self.emotion_adapter(emotion_input)
                if self.osram_relation_dual_readout and self.training:
                    base_hidden = self.emotion_norm(flat_anchor)
                    self.last_relation_base_hidden = torch.where(
                        valid.unsqueeze(-1), base_hidden, torch.zeros_like(base_hidden))
                hidden = self.emotion_norm(flat_anchor + self.relation_block(
                    local, emotion_base, emotion_gap, availability, umask, flat_anchor))
            elif self.osram_local_skip_gate:
                skip_gate = self.local_skip_gate(local, emotion_base, emotion_gap, availability, umask)
                hidden = self.emotion_norm(
                    skip_gate * self.local_skip(local) + self.emotion_adapter(emotion_input)
                )
            else:
                hidden = self.emotion_norm(
                    self.local_skip(local) + self.emotion_adapter(emotion_input)
                )
        if getattr(self, 'core20_collect_multilevel', False):
            # ONE trajectory, three final task exits, no second mask/view.
            safe_gap = torch.where((valid[..., None] & ~availability.bool())[..., None], emotion_gap, 0.)
            base_input = torch.cat((local, emotion_base, torch.zeros_like(safe_gap).flatten(2)), -1)
            local_input = torch.cat((local, torch.zeros_like(emotion_base), torch.zeros_like(safe_gap).flatten(2)), -1)
            skip = self.local_skip(local)
            self.core20_multilevel_hidden = {
                'local': self.emotion_norm(skip + self.emotion_adapter(local_input)),
                'base': self.emotion_norm(skip + self.emotion_adapter(base_input)),
            }
        if self.osram_post_grn:
            hidden = self.post_grn(hidden, local, emotion_base, emotion_gap, availability, umask)
        elif self.osram_meaningful_block != 'none' or self.osram_readout_candidate != 'none' or self.osram_decision_correction or self.osram_gap_increment_filter or self.osram_relation_block or self.osram_memory_only_adapter or self.osram_local_skip_gate or self.osram_history_input_gate or self.osram_local_evidence_gate or self.osram_hierarchical_evidence_gate:
            hidden = torch.where(valid.unsqueeze(-1), hidden, torch.zeros_like(hidden))
        else:
            hidden = hidden * valid.unsqueeze(-1).to(hidden.dtype)

        diagnostics: dict[str, object] = {
            "ablation": self.osram_ablation,
            "gap_read": self.osram_gap_read,
            "gap_residual_strength": self.gap_residual_strength,
            "beta_mode": self.beta_mode,
            "emotion_ablation": self.osram_emotion_ablation,
            "query_use_availability": self.query_use_availability,
            "bidirectional": self.bidirectional,
            "forward_slot_reuse": self.forward_slot_reuse,
            "memory_alpha": torch.sigmoid(self.alpha_logits)
            .detach()
            .cpu()
            .tolist(),
            "memory_beta": torch.sigmoid(self.beta_logits).detach().cpu().tolist(),
            "base_context_norm": float(
                active_base_context[valid].norm(dim=-1).mean().item()
            )
            if bool(valid.any())
            else 0.0,
            "gap_context_norm": {
                name: float(
                    active_gap_context[..., index, :][valid]
                    .norm(dim=-1)
                    .mean()
                    .item()
                )
                if bool(valid.any())
                else 0.0
                for index, name in enumerate(MODALITIES)
            },
            "gap_base_ratio": {
                name: float(
                    active_gap_context[..., index, :][valid]
                    .norm(dim=-1)
                    .mean()
                    .item()
                    / active_base_context[valid]
                    .norm(dim=-1)
                    .mean()
                    .clamp_min(1e-8)
                    .item()
                )
                if bool(valid.any())
                else 0.0
                for index, name in enumerate(MODALITIES)
            },
            "memory_frobenius_norm": float(
                torch.stack(
                    [
                        base_forward[valid].norm(dim=-1).mean(),
                        base_backward[valid].norm(dim=-1).mean(),
                    ]
                ).mean()
            )
            if bool(valid.any())
            else 0.0,
            "address_residual": {
                name: {
                    metric: self._summarize(
                        diag_forward[name][metric] + diag_backward[name][metric]
                    )
                    for metric in ("rho", "eta", "cosine")
                }
                for name in MODALITIES
            },
        }
        if self.osram_readout_fusion == "modality-tracks":
            diagnostics["modality_track_fusion"] = self.modality_track_fusion.last_diagnostics
        elif self.osram_readout_fusion == "modality-track-residual":
            diagnostics["modality_track_residual"] = self.modality_track_residual.last_diagnostics
        elif self.osram_readout_fusion == "base-gap-delta":
            diagnostics["base_gap_delta_fusion"] = self.base_gap_delta_fusion.last_diagnostics
        elif self.osram_readout_fusion == "memory-shift-residual":
            diagnostics["memory_shift_filter"] = self.memory_shift_filter.last_diagnostics
        elif self.osram_readout_fusion != "flat":
            diagnostics["local_centered_fusion"] = self.local_centered_fusion.last_diagnostics
        if self.osram_post_grn:
            diagnostics["post_grn"] = self.post_grn.last_diagnostics
        if self.osram_local_skip_gate:
            diagnostics['local_skip_gate'] = self.local_skip_gate.last_diagnostics
        if self.osram_history_input_gate:
            diagnostics['history_input_gate'] = self.history_input_gate.last_diagnostics
        if self.osram_hierarchical_evidence_gate:
            diagnostics['hierarchical_evidence_gate'] = self.hierarchical_evidence_gate.last_diagnostics
        if self.osram_local_evidence_gate:
            diagnostics['local_evidence_gate'] = self.local_evidence_gate.last_diagnostics
        if self.osram_relation_block:
            diagnostics['current_history_relation'] = self.relation_block.last_diagnostics
        if self.osram_gap_increment_filter:
            diagnostics['gap_increment_filter'] = self.gap_increment_filter.last_diagnostics
        if self.osram_readout_candidate != 'none':
            diagnostics['readout_candidate'] = self.readout_candidate.last_diagnostics
        self.last_diagnostics = diagnostics
        if self.osram_meaningful_block != 'none':
            diagnostics['meaningful_block'] = self.meaningful_block.last_diagnostics
        contexts = {
            "base": active_base_context,
            "gap": active_gap_context,
            "local": local,
        }
        return hidden, contexts
