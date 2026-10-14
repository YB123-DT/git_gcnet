"""Small trainable probes over frozen original OSRAM state and projections."""

import torch
from torch import nn
from torch.nn import functional as F


def forward_reads(memory, queries, observed_keys, availability,
                  read_ridge=1e-3, residual_strength=1.):
    """Original pre-write read algebra, with gradients only through queries.

    Inputs are valid utterance rows: memory [N,H,V,K], queries [N,4,H,K],
    observed keys [N,H,K,3], and the original availability [N,3]. Inactive
    keys and Gap queries are selected to zero before arithmetic so their
    unused values cannot contaminate active reads or query gradients.
    """
    memory = memory.detach()
    available = availability.detach().bool()
    keys = torch.where(available[:, None, None, :], observed_keys.detach(), 0.)
    batch = memory.shape[0]
    base = (memory @ queries[:, 0].unsqueeze(-1)).squeeze(-1).reshape(batch, -1)
    gaps = []
    for index in range(3):
        active = ~available[:, index]
        query = torch.where(active[:, None, None], queries[:, index + 1], 0.)
        gram = keys.transpose(-1, -2) @ keys
        identity = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
        system = gram + read_ridge * identity
        rhs = keys.transpose(-1, -2) @ query.unsqueeze(-1)
        coefficients = torch.linalg.solve(system, rhs).squeeze(-1)
        projected_query = query - (keys @ coefficients.unsqueeze(-1)).squeeze(-1)
        residual_query = query - residual_strength * (query - projected_query)
        read = (memory @ residual_query.unsqueeze(-1)).squeeze(-1).reshape(batch, -1)
        gaps.append(torch.where(active[:, None], read, 0.))
    return base, torch.stack(gaps, dim=1)


class QueryCorrector(nn.Module):
    """Current-evidence-only query correction; frozen history is never an input."""

    def __init__(self, local_dim=256, key_dim=64, num_heads=8,
                 hidden_dim=128, type_dim=16, head_dim=16):
        super().__init__()
        self.num_heads = num_heads
        self.type_embedding = nn.Embedding(4, type_dim)
        self.head_embedding = nn.Embedding(num_heads, head_dim)
        self.delta_mlp = nn.Sequential(
            nn.Linear(local_dim + key_dim + 3 + type_dim + head_dim, hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, key_dim),
        )
        nn.init.zeros_(self.delta_mlp[-1].weight)
        nn.init.zeros_(self.delta_mlp[-1].bias)

    def forward(self, local, original_queries, availability):
        local = local.detach()
        original = original_queries.detach()
        availability = availability.detach()
        batch = local.shape[0]
        active = torch.cat((torch.ones(batch, 1, device=local.device, dtype=torch.bool),
                            ~availability.bool()), dim=1)
        safe_queries = torch.where(active[:, :, None, None], original, 0.)
        inputs = torch.cat((
            local[:, None, None, :].expand(-1, 4, self.num_heads, -1),
            safe_queries,
            availability.to(local.dtype)[:, None, None, :].expand(-1, 4, self.num_heads, -1),
            self.type_embedding.weight[None, :, None, :].expand(batch, -1, self.num_heads, -1),
            self.head_embedding.weight[None, None, :, :].expand(batch, 4, -1, -1),
        ), dim=-1)
        delta = self.delta_mlp(inputs)
        corrected = F.normalize(safe_queries + delta, dim=-1)
        return torch.where(active[:, :, None, None], corrected, original)


def pack_features(local, base, gap, availability):
    """Pack Local, Base, three hard-masked Gaps, and original availability."""
    safe_gap = torch.where(~availability.bool().unsqueeze(-1), gap, 0.)
    return torch.cat((local, base, safe_gap.flatten(1), availability.to(local.dtype)), dim=-1)


class LightReadout(nn.Module):
    """Identical small decoder architecture for real, Local, and donor probes."""

    def __init__(self, local_dim=256, context_dim=512, hidden_dim=64, num_classes=7):
        super().__init__()
        self.local_dim = local_dim
        self.context_dim = context_dim
        width = local_dim + 4 * context_dim + 3
        self.register_buffer('feature_mean', torch.zeros(width))
        self.register_buffer('feature_scale', torch.ones(width))
        self.network = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    @torch.no_grad()
    def fit_normalizer(self, train_local, train_base, train_gap, train_availability):
        """Fit only from the caller's training split; forward never updates it."""
        packed = pack_features(train_local, train_base, train_gap, train_availability)
        if packed.shape[0] == 0:
            raise ValueError('Cannot fit the normalizer on empty training data')
        self.feature_mean.copy_(packed.mean(dim=0))
        self.feature_scale.copy_(packed.std(dim=0, unbiased=False).clamp_min(1e-6))
        return self

    def normalized_features(self, local, base, gap, availability):
        normalized = (pack_features(local, base, gap, availability)
                      - self.feature_mean) / self.feature_scale
        gap_start = self.local_dim + self.context_dim
        gap_end = gap_start + 3 * self.context_dim
        normalized_gap = normalized[:, gap_start:gap_end].reshape(-1, 3, self.context_dim)
        normalized_gap = torch.where(~availability.bool().unsqueeze(-1), normalized_gap, 0.)
        return torch.cat((normalized[:, :gap_start], normalized_gap.flatten(1),
                          normalized[:, gap_end:]), dim=-1)

    def forward(self, local, base, gap, availability):
        return self.network(self.normalized_features(local, base, gap, availability))
