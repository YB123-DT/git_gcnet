"""Task-relevant Text state and A/V-conditioned Text prediction."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn


class TextCore(nn.Module):
    """Map Text into a shared 64-d task space and predict it from A/V context.

    The predictor never receives the Text latent.  Its only contextual inputs
    are the observed Audio/Visual student slots, their two-bit availability
    pattern, and the causal OSRAM Base/Text-Gap reads supplied by the caller.
    ``read_residual`` is consumed after the OSRAM memory scan, so it can affect
    the current read representation but cannot affect persistent writes.
    """

    task_dim = 64

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if int(latent_dim) <= 0 or int(context_dim) <= 0:
            raise ValueError("latent_dim and context_dim must be positive")
        self.latent_dim = int(latent_dim)
        self.context_dim = int(context_dim)
        self.encoder = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, self.task_dim),
        )
        predictor_input_dim = 2 * self.latent_dim + 2 * self.context_dim + 2
        self.predictor = nn.Sequential(
            nn.LayerNorm(predictor_input_dim),
            nn.Linear(predictor_input_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, self.task_dim),
        )
        self.task_head = nn.Linear(self.task_dim, 1)
        self.task_slot_adapter = nn.Sequential(
            nn.LayerNorm(self.task_dim),
            nn.Linear(self.task_dim, self.latent_dim),
        )
        # Start with the unchanged OSRAM readout; the task-space route is
        # enabled by its own losses instead of injecting a random residual.
        nn.init.zeros_(self.task_slot_adapter[-1].weight)
        nn.init.zeros_(self.task_slot_adapter[-1].bias)

    def forward(
        self,
        latents: Mapping[str, torch.Tensor],
        availability: torch.Tensor,
        umask: torch.Tensor,
        base_context: torch.Tensor,
        gap_context: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        required = {"audio", "text", "visual"}
        if set(latents) < required:
            raise ValueError("latents must contain audio, text, and visual")
        if availability.ndim != 3 or availability.shape[-1] != 3:
            raise ValueError("availability must have shape [L, B, 3]")
        length, batch = availability.shape[:2]
        if umask.shape != (batch, length):
            raise ValueError("umask must have shape [B, L]")
        if base_context.shape[:2] != (length, batch):
            raise ValueError("base_context leading dimensions do not match")
        if gap_context.shape[:3] != (length, batch, 3):
            raise ValueError("gap_context must have shape [L, B, 3, C]")
        valid = umask.T.bool()
        source_pattern = availability[..., (0, 2)].to(dtype=latents["audio"].dtype)
        predictor_input = torch.cat(
            (
                latents["audio"],
                latents["visual"],
                source_pattern,
                base_context,
                gap_context[..., 1, :],
            ),
            dim=-1,
        )
        u_pred = self.predictor(predictor_input)
        u_true = self.encoder(latents["text"])
        text_observed = availability[..., 1].bool()
        u_use = torch.where(text_observed.unsqueeze(-1), u_true, u_pred)
        u_use = u_use * valid.unsqueeze(-1).to(u_use.dtype)
        read_residual = self.task_slot_adapter(u_use)
        read_residual = read_residual * valid.unsqueeze(-1).to(read_residual.dtype)
        return {
            "u_true": u_true,
            "u_pred": u_pred,
            "u_use": u_use,
            "read_residual": read_residual,
            "real_logits": self.task_head(u_true),
            "pred_logits": self.task_head(u_pred),
            "valid": valid,
            "text_observed": text_observed,
        }
