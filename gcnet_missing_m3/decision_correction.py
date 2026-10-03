"""Evidence-centered hierarchical corrections in task/logit space."""
from __future__ import annotations

import torch
from torch import nn


class EvidenceCenteredDecisionHead(nn.Module):
    """Three supervised exits sharing one causal evidence trajectory.

    Each delta compares the same deterministic network and conditioners with
    evidence present versus zero. Both evaluations remain differentiable.
    ``context_dim`` is the forward H*V width, without backward placeholder slots.
    """

    def __init__(self, local_dim: int, context_dim: int, n_classes: int) -> None:
        super().__init__()
        if min(local_dim, context_dim, n_classes) <= 0:
            raise ValueError('decision dimensions must be positive')
        self.local_dim = int(local_dim)
        self.context_dim = int(context_dim)
        self.n_classes = int(n_classes)

        def mlp(width, correction=False):
            network = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 128),
                                    nn.GELU(), nn.Linear(128, self.n_classes, bias=not correction))
            if correction:
                nn.init.zeros_(network[-1].weight)
            return network

        # New head initialization must not perturb the backbone/experiment RNG.
        with torch.random.fork_rng(devices=[]):
            self.local_head = mlp(self.local_dim)
            self.base_delta = mlp(self.local_dim + self.context_dim, correction=True)
            self.gap_delta = mlp(self.local_dim + 4 * self.context_dim + 3, correction=True)

    def forward(self, local, base, gap, availability, umask):
        if local.ndim != 3 or local.shape[-1] != self.local_dim:
            raise ValueError('local must have shape [L, B, local_dim]')
        shape = local.shape[:2]
        if (base.shape != (*shape, self.context_dim)
                or gap.shape != (*shape, 3, self.context_dim)
                or availability.shape != (*shape, 3) or umask.shape != (shape[1], shape[0])):
            raise ValueError('incompatible decision evidence shapes')
        valid = umask.T.bool()
        has_history = valid & (valid.long().cumsum(dim=0) > 1)
        safe_local = torch.where(valid[..., None], local, torch.zeros_like(local))
        safe_base = torch.where(has_history[..., None], base, torch.zeros_like(base))
        safe_availability = torch.where(valid[..., None], availability, torch.zeros_like(availability)).to(local.dtype)
        gap_active = has_history[..., None] & ~safe_availability.bool()
        safe_gap = torch.where(gap_active[..., None], gap, torch.zeros_like(gap))

        local_logits = self.local_head(safe_local)
        base_logits = self.base_delta(torch.cat((safe_local, safe_base), dim=-1))
        base_zero = self.base_delta(torch.cat((safe_local, torch.zeros_like(safe_base)), dim=-1))
        delta_base = torch.where(has_history[..., None], base_logits - base_zero, torch.zeros_like(base_logits))
        gap_condition = (safe_local, safe_base)
        gap_logits = self.gap_delta(torch.cat((*gap_condition, safe_gap.flatten(2), safe_availability), dim=-1))
        gap_zero = self.gap_delta(torch.cat((*gap_condition, torch.zeros_like(safe_gap).flatten(2), safe_availability), dim=-1))
        delta_gap = torch.where(gap_active.any(dim=-1)[..., None], gap_logits - gap_zero, torch.zeros_like(gap_logits))
        local_logits = torch.where(valid[..., None], local_logits, torch.zeros_like(local_logits))
        base_logits = local_logits + delta_base
        return {'local': local_logits, 'base': base_logits, 'full': base_logits + delta_gap,
                'delta_base': delta_base, 'delta_gap': delta_gap}
