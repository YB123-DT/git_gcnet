"""Conditional second access to the same pre-write CHM state.

``free`` and ``span`` share parameters, initialization and output interface.
The latter projects its new address onto the numerical column space of actual
Base/currently-active residualized Gap queries. Its pseudoinverse is
differentiable: the control does not add a stop-gradient into original queries.
Numerical rank is determined by the same explicit relative/absolute thresholds
in both modes. Rank diagnostics alone are detached.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class ConditionalMemoryReread(nn.Module):
    """Return a Base-forward correction; caller owns causal state selection."""

    def __init__(self, local_dim, num_heads, key_dim, value_dim, mode='free',
                 rank_rtol=1e-5, rank_atol=1e-7):
        super().__init__()
        if mode not in ('free', 'span'):
            raise ValueError('mode must be free or span')
        if any(not isinstance(d, int) or isinstance(d, bool) or d < 1
               for d in (local_dim, num_heads, key_dim, value_dim)):
            raise ValueError('all dimensions must be positive integers')
        if any(not math.isfinite(t) or t < 0 for t in (rank_rtol, rank_atol)):
            raise ValueError('rank tolerances must be finite and nonnegative')
        self.mode = mode
        self.local_dim = local_dim
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.rank_rtol = rank_rtol
        self.rank_atol = rank_atol
        evidence_dim = num_heads * value_dim
        self.query_net = nn.Sequential(
            nn.LayerNorm(local_dim + evidence_dim + 3),
            nn.Linear(local_dim + evidence_dim + 3, 128),
            nn.GELU(), nn.Linear(128, num_heads * key_dim))
        for layer in self.query_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        self.output_projection = nn.Linear(evidence_dim, evidence_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self.last_diagnostics = {}

    def expected_gradient_groups(self):
        return ('query_net', 'output_projection')

    def _query_space(self, queries, query):
        inverse = torch.linalg.pinv(
            queries, atol=self.rank_atol, rtol=self.rank_rtol)
        projected = (queries @ (inverse @ query.unsqueeze(-1))).squeeze(-1)
        with torch.no_grad():
            singular = torch.linalg.svdvals(queries.detach())
            threshold = torch.maximum(
                singular[..., :1] * self.rank_rtol,
                singular.new_tensor(self.rank_atol))
            rank = (singular > threshold).sum(-1)
        return projected, rank, inverse

    def forward(self, local, base, availability, memory, active_queries,
                query_active, valid, has_history):
        """Inputs are batch-first and memory is already decayed, before write.

        ``active_queries[B,H,K,4]`` contains actual Base and three effective Gap
        addresses. ``query_active[B,4]`` is the sole permission for those columns;
        hidden/inactive Gap addresses cannot expand the projected control space.
        No-history and padding rows are sanitized before any learned operation.
        """
        enabled = valid.bool() & has_history.bool()
        local = torch.where(enabled[:, None], local, torch.zeros_like(local))
        base = torch.where(enabled[:, None], base, torch.zeros_like(base))
        availability = torch.where(enabled[:, None], availability,
                                   torch.zeros_like(availability)).to(local.dtype)
        state = torch.where(enabled[:, None, None, None], memory,
                            torch.zeros_like(memory))
        columns_active = enabled[:, None] & query_active.bool()
        queries = torch.where(columns_active[:, None, None, :], active_queries,
                              torch.zeros_like(active_queries))
        query = self.query_net(torch.cat((local, base, availability), dim=-1))
        query = F.normalize(query.reshape(-1, self.num_heads, self.key_dim), dim=-1)
        projected, rank, inverse = self._query_space(queries, query)
        actual_query = F.normalize(projected, dim=-1) if self.mode == 'span' else query
        actual_query = torch.where(enabled[:, None, None], actual_query,
                                   torch.zeros_like(actual_query))
        read = (state @ actual_query.unsqueeze(-1)).squeeze(-1).flatten(1)
        delta = self.output_projection(read)
        delta = torch.where(enabled[:, None], delta, torch.zeros_like(delta))
        with torch.no_grad():
            # Reproject the actual normalized query for a mode-comparable
            # out-of-space diagnostic; zero/disabled rows are excluded.
            actual_projected = (queries.detach() @ (
                inverse.detach() @ actual_query.detach().unsqueeze(-1))).squeeze(-1)
            count = enabled.sum().clamp_min(1)
            head_count = count * self.num_heads
            outside = (actual_query - actual_projected).norm(dim=-1)
            self.last_diagnostics = {
                'query_rank_mean': (rank * enabled[:, None]).sum() / head_count,
                'query_out_of_span_norm_mean': (outside * enabled[:, None]).sum() / head_count,
                'new_read_norm': read.norm(dim=-1).sum() / count,
                'correction_norm': delta.norm(dim=-1).sum() / count,
                'active_count': enabled.sum(),
            }
        return delta
