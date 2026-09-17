"""PAM-A: explicit episodic Text memory with direct address supervision.

The bank is identical to PAM-E: every prior truly observed Text utterance is
stored as a ``(key, value)`` episode, predicted Text is never written, and the
read is causal.  The difference is the address mechanism:

* PAM-E used one context query plus the closed-form ridge read.
* PAM-A produces a signed score for every historical episode directly from the
  current-observation context, then reads ``z_hat = V_t a_hat``.

During training the module also receives the current Teacher Text target and
computes the oracle least-squares address

    a_star = (V_t^T V_t + lambda I)^-1 V_t^T stopgrad(z_t^T)

on all real history episodes.  The trainer adds a SmoothL1 address loss.
"""

from __future__ import annotations

from typing import Mapping, Tuple

import torch
from torch import nn
from torch.nn import functional as F

MODALITIES = ("audio", "text", "visual")


class ExplicitAddressTextMemory(nn.Module):
    """Explicit episodic Text bank with signed per-episode address prediction."""

    def __init__(
        self,
        latent_dim: int = 256,
        context_dim: int = 256,
        key_dim: int = 64,
        mask_embed_dim: int = 16,
        ridge: float = 1e-3,
        dropout: float = 0.1,
        scorer_hidden: int = 128,
    ) -> None:
        super().__init__()
        for name, value in (
            ("latent_dim", latent_dim),
            ("context_dim", context_dim),
            ("key_dim", key_dim),
            ("mask_embed_dim", mask_embed_dim),
            ("scorer_hidden", scorer_hidden),
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

        self.mask_embedding = nn.Embedding(4, self.mask_embed_dim)
        self.base_projection = nn.Linear(self.context_dim, self.latent_dim)
        self.gap_projection = nn.Linear(self.context_dim, self.latent_dim)
        context_input_dim = 2 * self.latent_dim + self.mask_embed_dim + 2 * self.latent_dim
        self.context_encoder = nn.Sequential(
            nn.LayerNorm(context_input_dim),
            nn.Linear(context_input_dim, self.latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.latent_dim, self.key_dim),
        )

        self.text_key_encoder = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.key_dim),
            nn.GELU(),
            nn.Linear(self.key_dim, self.key_dim),
        )

        scorer_input_dim = 4 * self.key_dim
        self.address_scorer = nn.Sequential(
            nn.LayerNorm(scorer_input_dim),
            nn.Linear(scorer_input_dim, scorer_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(scorer_hidden, 1),
        )
        # Start from zero address, so the initial read is exactly zero and the
        # classification path is unchanged until the address loss/emotion loss
        # move the scorer.
        nn.init.zeros_(self.address_scorer[-1].weight)
        nn.init.zeros_(self.address_scorer[-1].bias)

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

    def _context(
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
        context_input = torch.cat(
            (
                latents["audio"] * audio_mask.unsqueeze(-1),
                latents["visual"] * visual_mask.unsqueeze(-1),
                pattern,
                self.base_projection(base_context),
                self.gap_projection(gap_text_context),
            ),
            dim=-1,
        )
        context = F.normalize(self.context_encoder(context_input), dim=-1)
        return torch.where(source.unsqueeze(-1), context, torch.zeros_like(context))

    @staticmethod
    def _scores(context: torch.Tensor, keys: torch.Tensor, scorer: nn.Module) -> torch.Tensor:
        """Return signed per-episode scores [B,n] from context [B,K] and keys [B,n,K]."""
        batch, count, key_dim = keys.shape
        context_expanded = context.unsqueeze(1).expand(-1, count, -1)
        features = torch.cat(
            (
                context_expanded,
                keys,
                context_expanded * keys,
                (context_expanded - keys).abs(),
            ),
            dim=-1,
        )
        return scorer(features).squeeze(-1)

    @staticmethod
    def _oracle_address(
        values: torch.Tensor,
        target: torch.Tensor,
        ridge: float,
    ) -> torch.Tensor:
        """Solve ``min_a ||V a - target||^2 + ridge ||a||^2``.

        ``values`` is [B,n,latent_dim] and ``target`` is [B,latent_dim].
        """
        gram = values @ values.transpose(1, 2)
        identity = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
        system = gram + ridge * identity
        rhs = values @ target.unsqueeze(-1)
        return torch.linalg.solve(system, rhs).squeeze(-1)

    def forward(
        self,
        latents: Mapping[str, torch.Tensor],
        availability: torch.Tensor,
        umask: torch.Tensor,
        base_context: torch.Tensor,
        gap_text_context: torch.Tensor,
        target_text: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        valid_input, availability = self._validate(
            latents, availability, umask, base_context, gap_text_context
        )
        if target_text is not None and target_text.shape[:2] != latents["audio"].shape[:2]:
            raise ValueError("target_text must have shape [L,B,latent_dim]")
        length, batch = availability.shape[:2]
        source_av = valid_input & (
            (availability[..., 0] > 0) | (availability[..., 2] > 0)
        )
        text_observed = availability[..., 1].bool() & valid_input
        text_missing = source_av & (~availability[..., 1].bool())

        context = self._context(
            latents, availability, base_context, gap_text_context, valid_input
        )
        context = torch.where(
            text_missing.unsqueeze(-1), context, torch.zeros_like(context)
        )

        value_steps = latents["text"]
        key_steps = F.normalize(self.text_key_encoder(value_steps), dim=-1)
        zero_value = torch.zeros_like(value_steps[0])
        zero_key = torch.zeros_like(key_steps[0])

        z_hat_steps: list[torch.Tensor] = []
        bank_size_steps: list[torch.Tensor] = []
        history_keys: list[torch.Tensor] = []
        history_values: list[torch.Tensor] = []
        history_observed: list[torch.Tensor] = []
        bank_count = torch.zeros(batch, device=availability.device, dtype=torch.long)
        address_loss_terms: list[torch.Tensor] = []
        address_count = 0

        for t in range(length):
            bank_size_steps.append(bank_count)
            if history_keys:
                keys_seq = torch.stack(history_keys, dim=1)
                values_seq = torch.stack(history_values, dim=1)
                observed_seq = torch.stack(history_observed, dim=1)
                address = self._scores(context[t], keys_seq, self.address_scorer)
                read = torch.einsum("bnd,bn->bd", values_seq, address)
            else:
                keys_seq = values_seq = observed_seq = None
                address = None
                read = context.new_zeros(batch, self.latent_dim)
            read = torch.where(text_missing[t].unsqueeze(-1), read, torch.zeros_like(read))
            z_hat_steps.append(read)

            if (
                target_text is not None
                and keys_seq is not None
                and bool(text_missing[t].any())
            ):
                oracle = self._oracle_address(
                    values_seq, target_text[t].detach(), self.ridge
                )
                valid_loss = text_missing[t] & (bank_count > 0)
                if bool(valid_loss.any()):
                    selected_address = address[valid_loss]
                    selected_oracle = oracle[valid_loss].detach()
                    selected_observed = observed_seq[valid_loss]
                    if bool(selected_observed.any()):
                        loss = F.smooth_l1_loss(
                            selected_address[selected_observed],
                            selected_oracle[selected_observed],
                        )
                        count = int(selected_observed.sum().item())
                        address_loss_terms.append(loss * count)
                        address_count += count

            observed_t = text_observed[t]
            history_keys.append(
                torch.where(observed_t.unsqueeze(-1), key_steps[t], zero_key)
            )
            history_values.append(
                torch.where(observed_t.unsqueeze(-1), value_steps[t], zero_value)
            )
            history_observed.append(observed_t)
            bank_count = bank_count + observed_t.long()

        z_hat_text = torch.stack(z_hat_steps, dim=0)
        bank_size = torch.stack(bank_size_steps, dim=0)
        if address_count:
            address_loss = sum(address_loss_terms) / address_count
        else:
            address_loss = value_steps.sum() * 0.0
        return {
            "z_hat_text": z_hat_text,
            "pam_mask": source_av,
            "text_missing_mask": text_missing,
            "write_mask": text_observed,
            "bank_size": bank_size,
            "bank_nonempty": bank_size > 0,
            "address_loss": address_loss,
            "address_count": value_steps.new_tensor(float(address_count)),
        }
