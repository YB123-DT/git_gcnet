"""Bounded, source-preserving coupling of existing Base and Gap reads."""

import math

import torch
from torch import nn


class CoupledSourceRole(nn.Module):
    """Return additive read corrections without owning or updating any memory.

    Roles are Base, Gap-audio, Gap-text, Gap-visual; sources retain the same
    audio/text/visual ordering. ``trajectory`` must describe strict-past
    count, coverage, age and ever-seen state, before the current turn writes.
    """

    def __init__(self, local_dim=256, forward_dim=512, width=64, bound=.25):
        super().__init__()
        for name, value in (('local_dim', local_dim), ('forward_dim', forward_dim),
                            ('width', width)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if not isinstance(bound, (int, float)) or not math.isfinite(bound) or bound <= 0:
            raise ValueError('bound must be finite and positive')
        self.local_dim, self.forward_dim = local_dim, forward_dim
        self.bound = float(bound)
        self.local_proj = nn.Linear(local_dim, width)
        self.read_proj = nn.Linear(forward_dim, width)
        self.partner_proj = nn.Linear(forward_dim, width)
        self.role_embedding = nn.Embedding(4, 8)
        self.source_embedding = nn.Embedding(3, 8)
        feature_dim = 3 * width + 16 + 4 + 3
        self.net = nn.Sequential(nn.LayerNorm(feature_dim),
                                 nn.Linear(feature_dim, width), nn.GELU(),
                                 nn.Linear(width, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.last_gamma = None
        self.last_diagnostics = {}

    def forward(self, local, source_base, source_gap, trajectory, availability, umask):
        if local.ndim != 3 or local.shape[-1] != self.local_dim:
            raise ValueError(f'local must have shape [T,B,{self.local_dim}]')
        t, b, _ = local.shape
        expected = ((source_base, (t, b, 3, self.forward_dim), 'source_base'),
                    (source_gap, (t, b, 3, 3, self.forward_dim), 'source_gap'),
                    (trajectory, (t, b, 3, 4), 'trajectory'),
                    (availability, (t, b, 3), 'availability'),
                    (umask, (b, t), 'umask'))
        for value, shape, name in expected:
            if tuple(value.shape) != shape:
                raise ValueError(f'{name} must have shape {shape}')

        valid = umask.transpose(0, 1) > 0
        seen = trajectory[..., 3] > 0
        has_history = seen.any(dim=-1)
        role_active = torch.cat((has_history.unsqueeze(-1), ~(availability > 0)), dim=-1)
        role_active = role_active & valid.unsqueeze(-1) & has_history.unsqueeze(-1)
        active = role_active.unsqueeze(-1) & seen.unsqueeze(2)

        # Multiplication cannot mask NaN. Sanitize before every projection and
        # reduction, including reads used as partners for another role.
        reads = torch.cat((source_base.unsqueeze(2), source_gap), dim=2)
        reads = torch.where(active.unsqueeze(-1), reads, torch.zeros_like(reads))
        gap_count = active[:, :, 1:].sum(dim=2).clamp_min(1).unsqueeze(-1)
        base_partner = reads[:, :, 1:].sum(dim=2) / gap_count
        partners = torch.cat((base_partner.unsqueeze(2),
                              reads[:, :, :1].expand(-1, -1, 3, -1, -1)), dim=2)
        partners = torch.where(active.unsqueeze(-1), partners, torch.zeros_like(partners))
        local_values = local[:, :, None, None, :].expand(-1, -1, 4, 3, -1)
        local_values = torch.where(active.unsqueeze(-1), local_values,
                                   torch.zeros_like(local_values))
        metadata = trajectory.unsqueeze(2).expand(-1, -1, 4, -1, -1)
        metadata = torch.where(active.unsqueeze(-1), metadata, torch.zeros_like(metadata))
        metadata = torch.stack((metadata[..., 0].clamp_min(0).log1p(),
                                metadata[..., 1],
                                metadata[..., 2].clamp_min(0).log1p(),
                                metadata[..., 3]), dim=-1)
        current = availability[:, :, None, None, :].expand(-1, -1, 4, 3, -1)
        current = torch.where(active.unsqueeze(-1), current, torch.zeros_like(current))
        role_ids = torch.arange(4, device=local.device)
        source_ids = torch.arange(3, device=local.device)
        role = self.role_embedding(role_ids)[None, None, :, None, :].expand(t, b, -1, 3, -1)
        source = self.source_embedding(source_ids)[None, None, None, :, :].expand(t, b, 4, -1, -1)
        features = torch.cat((self.local_proj(local_values), self.read_proj(reads),
                              self.partner_proj(partners), role, source,
                              current.to(local.dtype), metadata.to(local.dtype)), dim=-1)
        features = torch.where(active.unsqueeze(-1), features, torch.zeros_like(features))
        score = self.net(features).squeeze(-1)
        gamma = torch.where(active, self.bound * score.tanh(), torch.zeros_like(score))
        delta = (gamma.unsqueeze(-1) * reads).sum(dim=3)
        delta = torch.where(role_active.unsqueeze(-1), delta, torch.zeros_like(delta))

        with torch.no_grad():
            denominator = active.sum(dim=(0, 1)).clamp_min(1)
            self.last_gamma = (gamma.detach().sum(dim=(0, 1)) / denominator)
            self.last_diagnostics = {
                'gamma_mean': self.last_gamma,
                'gamma_abs_mean': gamma.detach().abs().sum(dim=(0, 1)) / denominator,
                'delta_norm': delta.detach().norm(dim=-1).sum(dim=(0, 1))
                              / role_active.sum(dim=(0, 1)).clamp_min(1),
            }
        return delta, gamma
