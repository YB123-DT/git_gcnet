"""Modality-specific scalar gates on decoded Nested Gap residuals only."""
import torch
from torch import nn

from .meaningful_blocks_common import safe_mask


class NestedGapResidualGate(nn.Module):
    """Keep original Gap content; control only the new Nested correction."""

    def __init__(self, latent_dim, forward_dim, gated_modalities=(0, 1, 2)):
        super().__init__()
        self.gated_modalities = tuple(gated_modalities)
        if not self.gated_modalities or any(i not in (0, 1, 2) for i in self.gated_modalities):
            raise ValueError('Gated modalities must be selected from A/T/V indices0/1/2')
        self.local_projection = nn.Linear(latent_dim, 64)
        self.gap_projection = nn.Linear(forward_dim, 64)
        self.delta_projection = nn.Linear(forward_dim, 64)
        self.type_embedding = nn.Embedding(3, 16)
        self.network = nn.Sequential(nn.LayerNorm(211), nn.Linear(211, 32),
                                     nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        self.last_diagnostics = {}
        self.last_gates = None

    def forward(self, local, gap, delta_gap, active_gap, availability):
        active_gap = active_gap.bool()
        enabled = torch.tensor([i in self.gated_modalities for i in range(3)],
                               device=local.device, dtype=torch.bool)
        active_gate = active_gap & enabled[None]
        valid = active_gate.any(-1)
        q = self.local_projection(safe_mask(local, valid))[:, None].expand(-1, 3, -1)
        k = self.gap_projection(safe_mask(gap, active_gate))
        d = self.delta_projection(safe_mask(delta_gap, active_gate))
        types = self.type_embedding.weight[None].expand(local.shape[0], -1, -1)
        av = safe_mask(availability.to(local.dtype), valid)[:, None].expand(-1, 3, -1)
        features = safe_mask(torch.cat((q, k, d, types, av), -1), active_gate)
        predicted_gates = safe_mask(2 * self.network(features).squeeze(-1).sigmoid(), active_gate)
        gates = torch.where(enabled[None], predicted_gates, active_gap.to(local.dtype))
        filtered = safe_mask(gates[..., None] * safe_mask(delta_gap, active_gap), active_gap)
        with torch.no_grad():
            self.last_gates = gates.detach()
            self.last_diagnostics = {}
            for i, name in enumerate(('A', 'T', 'V')):
                selected = active_gap[:, i]
                count = int(selected.sum())
                self.last_diagnostics[name] = dict(
                    gate_enabled=i in self.gated_modalities,
                    active_count=count,
                    gate_mean=float(gates[selected, i].mean()) if count else 0.,
                    residual_norm=float(delta_gap[selected, i].norm(dim=-1).mean()) if count else 0.,
                    gated_residual_norm=float(filtered[selected, i].norm(dim=-1).mean()) if count else 0.)
        return filtered
