"""PAM-E: explicit cross-modal episodic Text memory.

This module stores every prior *truly observed* Text latent as its own
``(key, value)`` episode.  Reads use a learned A/V+OSRAM-context query and a
signed ridge linear combination over all historical Text values.  Nothing is
compressed into a recurrent associative matrix, and predicted Text is never
written back into the bank.
"""

from __future__ import annotations

from typing import Mapping, Tuple

import torch
from torch import nn
from torch.nn import functional as F

MODALITIES = ("audio", "text", "visual")


class ExplicitEpisodicTextMemory(nn.Module):
    """Causal episodic memory over all previously observed Text utterances.

    Parameters
    ----------
    latent_dim:
        Width of the Student latent space (256 for MOSI).
    context_dim:
        Width of the causal OSRAM ``Base`` / ``Gap`` contexts.  For a causal
        four-head OSRAM with 32-wide values this is ``2 * 4 * 32 = 256``.
    key_dim:
        Width of the normalized Text keys and cross-modal queries.
    mask_embed_dim:
        Width of the A/V availability embedding.
    ridge:
        Fixed positive ridge used by the signed linear retrieval.
    dropout:
        Dropout used inside the query encoder.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        context_dim: int = 256,
        key_dim: int = 64,
        mask_embed_dim: int = 16,
        ridge: float = 1e-3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        for name, value in (
            ("latent_dim", latent_dim),
            ("context_dim", context_dim),
            ("key_dim", key_dim),
            ("mask_embed_dim", mask_embed_dim),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if float(ridge) <= 0:
            raise ValueError("ridge must be positive")
        self.latent_dim = int(latent_dim)
        self.context_dim = int(context_dim)
        self.key_dim = int(key_dim)
        self.mask_embed_dim = int(mask_embed_dim)
        self.ridge = float(ridge)

        # A/V availability uses only the current visible A/V pattern.  Text is
        # never an input to the query encoder.
        self.mask_embedding = nn.Embedding(4, self.mask_embed_dim)
        self.base_projection = nn.Linear(self.context_dim, self.latent_dim)
        self.gap_projection = nn.Linear(self.context_dim, self.latent_dim)
        query_input_dim = 2 * self.latent_dim + self.mask_embed_dim + 2 * self.latent_dim
        self.query_encoder = nn.Sequential(
            nn.LayerNorm(query_input_dim),
            nn.Linear(query_input_dim, self.latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.latent_dim, self.key_dim),
        )

        # Independent Text key encoder.  Values remain in the full 256-d
        # Student Text latent space.
        self.text_key_encoder = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.key_dim),
            nn.GELU(),
            nn.Linear(self.key_dim, self.key_dim),
        )

        # Kept for numerical compatibility with the explicit identity matrix.
        self.register_buffer(
            "retrieval_identity",
            torch.eye(self.key_dim),
            persistent=False,
        )

    def _validate(
        self,
        latents: Mapping[str, torch.Tensor],
        availability: torch.Tensor,
        umask: torch.Tensor,
        base_context: torch.Tensor,
        gap_text_context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if set(latents) != set(MODALITIES):
            raise ValueError("latents must contain audio, text, and visual")
        reference = latents["audio"]
        if reference.ndim != 3 or reference.shape[-1] != self.latent_dim:
            raise ValueError("latents must have shape [L,B,latent_dim]")
        if any(value.shape != reference.shape for value in latents.values()):
            raise ValueError("all modality latents must share the same shape")
        length, batch = reference.shape[:2]
        if availability.shape != (length, batch, 3):
            raise ValueError("availability must have shape [L,B,3]")
        if umask.shape != (batch, length):
            raise ValueError("umask must have shape [B,L]")
        if not bool(((availability == 0) | (availability == 1)).all()):
            raise ValueError("availability must be binary")
        if not bool(((umask == 0) | (umask == 1)).all()):
            raise ValueError("umask must be binary")
        if base_context.shape != (length, batch, self.context_dim):
            raise ValueError("base_context must have shape [L,B,context_dim]")
        if gap_text_context.shape != (length, batch, self.context_dim):
            raise ValueError("gap_text_context must have shape [L,B,context_dim]")
        valid = umask.transpose(0, 1).bool()
        if bool(availability[~valid].any()):
            raise ValueError("padding availability must be zero")
        if bool((availability[valid].sum(dim=-1) == 0).any()):
            raise ValueError("valid utterances require at least one observed modality")
        return valid, availability

    def _query(
        self,
        latents: Mapping[str, torch.Tensor],
        availability: torch.Tensor,
        base_context: torch.Tensor,
        gap_text_context: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        audio_mask = availability[..., 0]
        visual_mask = availability[..., 2]
        pattern_id = (audio_mask.long() * 2 + visual_mask.long()).clamp(0, 3)
        pattern = self.mask_embedding(pattern_id)
        source = valid & ((audio_mask > 0) | (visual_mask > 0))
        query_input = torch.cat(
            (
                latents["audio"] * audio_mask.unsqueeze(-1),
                latents["visual"] * visual_mask.unsqueeze(-1),
                pattern,
                self.base_projection(base_context),
                self.gap_projection(gap_text_context),
            ),
            dim=-1,
        )
        query = F.normalize(self.query_encoder(query_input), dim=-1)
        return torch.where(source.unsqueeze(-1), query, torch.zeros_like(query))

    @staticmethod
    def _resolve_ridge(
        keys: torch.Tensor,
        values: torch.Tensor,
        query: torch.Tensor,
        ridge: float,
    ) -> torch.Tensor:
        """Signed ridge retrieval over a non-empty episode bank.

        ``keys`` is ``[B,n,K]`` and ``values`` is ``[B,n,D]``.  Padded / never
        observed episodes are zero keys, for which the ridge solution gives
        exactly zero coefficients; coverage is tracked separately.
        """

        # Spec convention: K_t = [k_1, ..., k_n] in R^{key_dim x n}.
        # ``keys`` is its transpose [B,n,key_dim].
        transposed = keys.transpose(1, 2)  # [B,key_dim,n]
        gram = transposed.transpose(1, 2) @ transposed  # [B,n,n]
        identity = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
        system = gram + ridge * identity
        rhs = transposed.transpose(1, 2) @ query.unsqueeze(-1)  # [B,n,1]
        coefficients = torch.linalg.solve(system, rhs).squeeze(-1)  # [B,n]
        return torch.einsum("bnd,bn->bd", values, coefficients)

    def forward(
        self,
        latents: Mapping[str, torch.Tensor],
        availability: torch.Tensor,
        umask: torch.Tensor,
        base_context: torch.Tensor,
        gap_text_context: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        valid, availability = self._validate(
            latents, availability, umask, base_context, gap_text_context
        )
        length, batch = availability.shape[:2]
        text_observed = availability[..., 1].bool() & valid
        source_av = valid & (
            (availability[..., 0] > 0) | (availability[..., 2] > 0)
        )
        text_missing = valid & (~availability[..., 1].bool()) & source_av

        query = self._query(
            latents, availability, base_context, gap_text_context, valid
        )
        query = torch.where(
            text_missing.unsqueeze(-1), query, torch.zeros_like(query)
        )

        # Real observed Text is the only thing that can ever enter the bank.
        value_steps = latents["text"]
        key_steps = F.normalize(self.text_key_encoder(value_steps), dim=-1)
        zero_value = torch.zeros_like(value_steps[0])
        zero_key = torch.zeros_like(key_steps[0])

        z_hat_steps: list[torch.Tensor] = []
        bank_size_steps: list[torch.Tensor] = []
        history_values: list[torch.Tensor] = []
        history_keys: list[torch.Tensor] = []
        bank_count = torch.zeros(batch, device=availability.device, dtype=torch.long)

        for t in range(length):
            bank_size_steps.append(bank_count)
            if history_keys:
                keys = torch.stack(history_keys, dim=1)
                values = torch.stack(history_values, dim=1)
                read = self._resolve_ridge(
                    keys, values, query[t], self.ridge
                )
            else:
                read = query.new_zeros(batch, self.latent_dim)
            # A retrieval for a sample with an empty bank is exactly zero
            # because all of that sample's stacked keys are zero.  Mask here to
            # keep the invariant explicit and to avoid reading at T-present or
            # no-source positions.
            read = torch.where(
                text_missing[t].unsqueeze(-1), read, torch.zeros_like(read)
            )
            z_hat_steps.append(read)

            observed_t = text_observed[t]
            # Store both observed and zero episodes so the ragged bank can be
            # represented as a rectangular tensor.  Zero-key episodes have
            # zero retrieval coefficient; ``bank_count`` records the real
            # episode occupancy used for coverage diagnostics.
            history_keys.append(
                torch.where(observed_t.unsqueeze(-1), key_steps[t], zero_key)
            )
            history_values.append(
                torch.where(observed_t.unsqueeze(-1), value_steps[t], zero_value)
            )
            bank_count = bank_count + observed_t.long()

        z_hat_text = torch.stack(z_hat_steps, dim=0)
        bank_size = torch.stack(bank_size_steps, dim=0)
        return {
            "z_hat_text": z_hat_text,
            "query": query,
            "pam_mask": source_av,
            "text_missing_mask": text_missing,
            "write_mask": text_observed,
            "bank_size": bank_size,
            "bank_nonempty": bank_size > 0,
        }
