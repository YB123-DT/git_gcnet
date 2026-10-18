"""Strictly causal side read of recent, actually observed modality latents.

This branch does not read or change the OSRAM state. Its output is an optional
pre-normalization Flat residual, not a missing-modality prediction.
"""

from typing import Dict

import torch
from torch import Tensor, nn


class RecentHistoryAttentionResidual(nn.Module):
    """Read at most ``window * 3`` tokens strictly before each utterance.

    Inputs use time-major [T, B, D] layout; ``umask`` is [T, B]. Only the
    first ``context_dim`` Base coordinates participate in the branch query.
    Initialization isolation from the original model's RNG is the caller's
    responsibility, e.g. instantiate inside ``torch.random.fork_rng``.
    """

    MODALITIES = ("audio", "text", "visual")

    def __init__(
        self,
        local_dim: int = 256,
        latent_dim: int = 256,
        context_dim: int = 512,
        output_dim: int = 1600,
        window: int = 4,
        attention_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if window < 1:
            raise ValueError("window must be a positive number of utterances")
        if num_heads < 1 or attention_dim < 1 or attention_dim % num_heads:
            raise ValueError("attention_dim must be divisible by positive num_heads")
        self.window = window
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.query = nn.Linear(local_dim + context_dim, attention_dim)
        self.key = nn.Linear(latent_dim, attention_dim)
        self.value = nn.Linear(latent_dim, attention_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.output = nn.Linear(attention_dim, output_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.last_diagnostics: Dict[str, Tensor] = {}

    def forward(
        self,
        local: Tensor,
        base: Tensor,
        latents: Dict[str, Tensor],
        availability: Tensor,
        umask: Tensor,
    ) -> Tensor:
        length, batch = local.shape[:2]
        if umask.shape != (length, batch):
            raise ValueError("umask must be time-major [T, B]")
        if availability.shape != (length, batch, 3):
            raise ValueError("availability must be [T, B, 3]")
        if base.shape[:2] != (length, batch) or base.shape[-1] < self.context_dim:
            raise ValueError("Base must contain the configured forward context dimensions")
        valid = umask.bool()
        observed = availability.bool() & valid.unsqueeze(-1)
        tokens = torch.stack([latents[name] for name in self.MODALITIES], dim=2)
        # Sanitize BEFORE projections: NaN * 0 is still NaN, including backward.
        tokens = torch.where(observed.unsqueeze(-1), tokens, torch.zeros_like(tokens))
        key = self.key(tokens)
        value = self.value(tokens)
        prefix_key = key.new_zeros(self.window, batch, 3, key.shape[-1])
        prefix_value = value.new_zeros(self.window, batch, 3, value.shape[-1])
        prefix_mask = observed.new_zeros(self.window, batch, 3)
        padded_key = torch.cat((prefix_key, key), dim=0)
        padded_value = torch.cat((prefix_value, value), dim=0)
        padded_mask = torch.cat((prefix_mask, observed), dim=0)
        # This is a window axis, not a conversation axis: no cross-batch tokens.
        past_key = torch.stack([padded_key[offset:offset + length]
                                for offset in range(self.window)], dim=2)
        past_value = torch.stack([padded_value[offset:offset + length]
                                  for offset in range(self.window)], dim=2)
        past_mask = torch.stack([padded_mask[offset:offset + length]
                                 for offset in range(self.window)], dim=2)
        past_mask = past_mask.reshape(length, batch, self.window * 3)
        past_key = past_key.reshape(length, batch, self.window * 3,
                                    self.num_heads, self.head_dim).transpose(2, 3)
        past_value = past_value.reshape(length, batch, self.window * 3,
                                        self.num_heads, self.head_dim).transpose(2, 3)
        safe_local = torch.where(valid.unsqueeze(-1), local, torch.zeros_like(local))
        forward_base = base[..., :self.context_dim]
        safe_base = torch.where(valid.unsqueeze(-1), forward_base, torch.zeros_like(forward_base))
        query = self.query(torch.cat((safe_local, safe_base), dim=-1))
        query = query.reshape(length, batch, self.num_heads, self.head_dim)
        scores = (past_key * query.unsqueeze(-2)).sum(-1) * self.head_dim ** -0.5
        scores = scores.masked_fill(~past_mask.unsqueeze(-2), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(past_mask.unsqueeze(-2), weights, torch.zeros_like(weights))
        weights = self.attention_dropout(weights)
        recent = (weights.unsqueeze(-1) * past_value).sum(-2)
        recent = recent.reshape(length, batch, self.num_heads * self.head_dim)
        active = valid & past_mask.any(-1)
        residual = self.output(recent)
        residual = torch.where(active.unsqueeze(-1), residual, torch.zeros_like(residual))
        with torch.no_grad():
            count = valid.sum().clamp_min(1)
            self.last_diagnostics = {
                "recent_history_residual_norm": residual.norm(dim=-1).sum() / count,
                "recent_history_token_count": (past_mask.sum(-1) * valid).sum() / count,
                "recent_history_active_fraction": active.sum() / count,
            }
        return residual
