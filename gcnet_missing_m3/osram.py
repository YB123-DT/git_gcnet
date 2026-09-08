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


def _logit(probability: float) -> float:
    probability = float(probability)
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be strictly between zero and one")
    return math.log(probability / (1.0 - probability))


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
    ) -> None:
        super().__init__()
        if osram_ablation not in OSRAM_ABLATIONS:
            raise ValueError(
                "osram_ablation must be 'full', 'local-only', or 'local-base'"
            )
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
        self.latent_dim = int(latent_dim)
        self.output_dim = int(output_dim)
        self.num_heads = int(num_heads)
        self.key_dim = int(key_dim)
        self.value_dim = int(value_dim)
        self.n_speakers = int(n_speakers)
        self.read_ridge = float(read_ridge)
        self.write_ridge = float(write_ridge)
        self.osram_ablation = osram_ablation
        self.query_use_availability = bool(query_use_availability)
        self.bidirectional = bool(bidirectional)
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
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
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
        node_value = self.node_norm(node)
        keys: dict[str, torch.Tensor] = {}
        values: dict[str, torch.Tensor] = {}
        for name in MODALITIES:
            latent_value = self.latent_norm(latents[name])
            key_input = torch.cat(
                (latent_value, node_value, availability_embed, speaker), dim=-1
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
            (node_value, query_availability, speaker), dim=-1
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
        return memory + residual @ solved

    def _scan(
        self,
        keys: Mapping[str, torch.Tensor],
        values: Mapping[str, torch.Tensor],
        queries: torch.Tensor,
        availability: torch.Tensor,
        valid: torch.Tensor,
        reverse: bool,
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
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        valid = self._validate_inputs(
            node, latents, availability, qmask, umask, seq_lengths
        )
        keys, values, queries = self._project_sequence(
            node, latents, availability, qmask
        )
        base_forward, gap_forward, diag_forward = self._scan(
            keys, values, queries, availability, valid, reverse=False
        )
        if self.bidirectional:
            base_backward, gap_backward, diag_backward = self._scan(
                keys, values, queries, availability, valid, reverse=True
            )
        else:
            # Preserve parameter shapes and fusion slots, without future reads.
            base_backward = torch.zeros_like(base_forward)
            gap_backward = torch.zeros_like(gap_forward)
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

        local = node + self.local_path(node)
        local = local * valid.unsqueeze(-1).to(local.dtype)
        missing = 1.0 - availability.to(dtype=node.dtype)
        emotion_input = torch.cat(
            (
                local,
                active_base_context,
                (active_gap_context * missing.unsqueeze(-1)).reshape(
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
            "query_use_availability": self.query_use_availability,
            "bidirectional": self.bidirectional,
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
        self.last_diagnostics = diagnostics
        contexts = {
            "base": active_base_context,
            "gap": active_gap_context,
            "local": local,
        }
        return hidden, contexts
