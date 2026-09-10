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
    ) -> None:
        super().__init__()
        if osram_readout_fusion not in ("flat", "local-gated", "local-cross-attn"):
            raise ValueError("osram_readout_fusion must be flat, local-gated or local-cross-attn")
        if osram_readout_fusion != "flat" and (osram_ablation != "full" or osram_emotion_ablation != "full"):
            raise ValueError("local-gated cannot combine with readout ablations")
        if osram_ablation not in OSRAM_ABLATIONS:
            raise ValueError(
                "osram_ablation must be 'full', 'local-only', or 'local-base'"
            )
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
        self.query_use_availability = bool(query_use_availability)
        self.bidirectional = bool(bidirectional)
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

        if osram_readout_fusion != "flat":
            # Preserve RNG for downstream Student/Teacher/MMoE construction.
            with torch.random.fork_rng(devices=[]):
                fusion_cls = LocalCenteredContextFusion if osram_readout_fusion == "local-gated" else LocalCrossAttentionFusion
                self.local_centered_fusion = fusion_cls(
                    self.latent_dim, self.context_dim, self.output_dim, dropout=dropout)
            self.local_centered_fusion.local_skip.load_state_dict(self.local_skip.state_dict())
            self.local_centered_fusion.emotion_norm.load_state_dict(self.emotion_norm.state_dict())
            # Historical flat keys remain available, without unused optimizer parameters.
            self.emotion_adapter.requires_grad_(False)
            self.local_skip.requires_grad_(False)
            self.emotion_norm.requires_grad_(False)

        self.last_diagnostics: dict[str, object] = {}

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
        slot_scale = (
            availability_value.unsqueeze(1)
            * beta.clamp_min(0.0).sqrt().unsqueeze(0)
        ).unsqueeze(2)
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
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, list[float]]]:
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
                residual_query = self._address_residual(slot_keys, query)
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

            memory_candidate = self.block_write(
                read_memory,
                slot_keys,
                slot_values,
                availability[time_index],
                beta=beta,
            )
            memory = torch.where(
                active.view(batch, 1, 1, 1), memory_candidate, memory
            )
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
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Condition reads/local features separately from real-observation writes.

        Both optional nodes default to ``node``. Completion belongs exclusively
        in ``read_node``; keys and gap-address projection use ``write_node``,
        while values continue to use the supplied real modality latents.
        """
        read_node = node if read_node is None else read_node
        write_node = node if write_node is None else write_node
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
        base_forward, gap_forward, diag_forward = self._scan(
            keys, values, queries, availability, valid, reverse=False,
            retention_diagnostics=(memory_retention_diagnostics
                                   if collect_memory_retention_diagnostics else None),
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

        local = read_node + self.local_path(read_node)
        local = local * valid.unsqueeze(-1).to(local.dtype)
        missing = 1.0 - availability.to(dtype=node.dtype)
        # Unlike the legacy switch above, mask ONLY the emotion-fusion inputs.
        # The returned Base/Gap tensors still supervise the structured predictor.
        emotion_base = active_base_context
        emotion_gap = active_gap_context
        if self.osram_emotion_ablation in ("local-only", "local-gap"):
            emotion_base = torch.zeros_like(emotion_base)
        if self.osram_emotion_ablation in ("local-only", "local-base"):
            emotion_gap = torch.zeros_like(emotion_gap)
        if self.osram_readout_fusion != "flat":
            hidden = self.local_centered_fusion(
                local, base_context, gap_context, availability, umask)
        else:
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
        hidden = hidden * valid.unsqueeze(-1).to(hidden.dtype)

        diagnostics: dict[str, object] = {
            "ablation": self.osram_ablation,
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
        if self.osram_readout_fusion != "flat":
            diagnostics["local_centered_fusion"] = self.local_centered_fusion.last_diagnostics
        self.last_diagnostics = diagnostics
        contexts = {
            "base": active_base_context,
            "gap": active_gap_context,
            "local": local,
        }
        return hidden, contexts
