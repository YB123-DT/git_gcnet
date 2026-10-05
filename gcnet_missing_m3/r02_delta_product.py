"""Ungated DeltaProduct adapted to OSRAM's historical read-before-write scan.

Source: arXiv:2502.10297v7, Eq. (3), and automl/DeltaProduct commit
d62241a81d07aa32b1b65e7d17377f6a7cd0a5d8. Independently implemented
from the equation; no author kernels, short convolution, or LM output head.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .core20_storage import SequenceStorage


class DeltaProductStorage(SequenceStorage):
    """Multiple ordered delta steps from one observed-only pooled utterance.

    Internal slots have separate learned projections, shared across OSRAM heads.
    They do not correspond to modalities. State shape is [B,H,K,V]. Existing
    OSRAM query magnitudes (including residual addresses) are preserved at read.
    """

    def __init__(self, config, osram=None):
        super().__init__(config.osram_num_heads, config.osram_key_dim,
                         config.osram_value_dim, config.latent_dim)
        self.num_householder = int(getattr(config, 'osram_r02_num_householder', 2))
        if self.num_householder < 1:
            raise ValueError('R02 requires at least one Householder slot')
        width = self.key_dim + self.value_dim
        self.key_projections = nn.ModuleList(
            nn.Linear(width, self.key_dim, bias=False) for _ in range(self.num_householder))
        self.value_projections = nn.ModuleList(
            nn.Linear(width, self.value_dim, bias=False) for _ in range(self.num_householder))
        self.beta_projections = nn.ModuleList(
            nn.Linear(width, 1, bias=False) for _ in range(self.num_householder))

    def initial_state(self, reference, batch):
        return reference.new_zeros(batch, self.num_heads, self.key_dim, self.value_dim)

    def read(self, state, queries, active):
        clean = torch.where(active[:, None, None, None], queries, 0.)
        reads = torch.einsum('bhkv,brhk->brhv', state, clean)
        return torch.where(active[:, None, None, None], reads, 0.), state

    def controls(self, pooled):
        """Return slot-first unit keys, values, and beta in (0,2)."""
        keys, values, betas = [], [], []
        for key_proj, value_proj, beta_proj in zip(
                self.key_projections, self.value_projections, self.beta_projections):
            raw_key = F.silu(key_proj(pooled))
            # A zero projection still yields a unit key, without NaN division.
            fallback = torch.zeros_like(raw_key)
            fallback[..., 0] = 1.
            key = torch.where(raw_key.norm(dim=-1, keepdim=True) > 1e-8,
                              F.normalize(raw_key, dim=-1, eps=1e-8), fallback)
            keys.append(key)
            values.append(F.silu(value_proj(pooled)))
            betas.append(2 * beta_proj(pooled).squeeze(-1).sigmoid())
        return torch.stack(keys), torch.stack(values), torch.stack(betas)

    @staticmethod
    def update(state, keys, values, betas):
        """Sequential residuals realize the ordered product, without dense A."""
        for key, value, beta in zip(keys, values, betas):
            residual = value - torch.einsum('bhkv,bhk->bhv', state, key)
            state = state + beta[..., None, None] * key[..., :, None] * residual[..., None, :]
        return state

    def write(self, state, keys, values, observed, active, queries):
        observed = observed.bool() & active[:, None].bool()
        mask = observed[:, None, None, :]
        clean_keys = torch.where(mask, keys, 0.)
        clean_values = torch.where(mask, values, 0.)
        count = observed.sum(-1).clamp_min(1).to(keys.dtype)
        pooled = torch.cat((clean_keys.sum(-1), clean_values.sum(-1)), -1)
        pooled = pooled / count[:, None, None]
        candidate = self.update(state, *self.controls(pooled))
        enabled = observed.any(-1)
        return torch.where(enabled[:, None, None, None], candidate, state)
