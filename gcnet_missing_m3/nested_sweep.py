"""Bounded Nested GNN ablations and sensitivities on the original Flat slots.

Group count changes graph tokenization only: contiguous actual heads are
concatenated before projection, and decoders reconstruct the complete read.
The OSRAM memory's own heads and read/write algorithm are unaffected.
"""
import torch
from torch import nn

from .meaningful_input_new40 import TokenAdapter
from .meaningful_new40_registry import NESTED_SWEEP


def _mlp(inputs, dim, hidden_dim=None):
    hidden_dim = dim if hidden_dim is None else hidden_dim
    return nn.Sequential(nn.Linear(inputs, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, dim))


class NestedSweepGNN(nn.Module):
    """Original rooted mean pooling, with one explicit experimental switch."""

    def __init__(self, dim=64, depth=3, markers=True, head_edges=True,
                 last_layer=False, plain_gin=False, mlp_hidden=None):
        super().__init__()
        if dim < 1 or depth < 1:
            raise ValueError('Nested dimension and depth must be positive')
        self.markers, self.head_edges = markers, head_edges
        self.last_layer, self.plain_gin = last_layer, plain_gin
        if markers and not plain_gin:
            self.root_embedding = nn.Embedding(2, dim)
            self.distance = nn.Embedding(2, dim)
        self.layers = nn.ModuleList([_mlp(dim, dim, mlp_hidden) for _ in range(depth)])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(depth)])
        self.epsilon = nn.Parameter(torch.zeros(depth))
        self.pool = _mlp(dim if last_layer else depth * dim, dim, mlp_hidden)
        self.readout = _mlp(3 * dim, dim, mlp_hidden)

    def adjacency(self, columns, heads, dtype):
        role = torch.where(columns == 0, -1, (columns - 1) // heads)
        head = (columns - 1) % heads
        adj = ((role[:, None] == role[None, :]) |
               (columns[:, None] == 0) | (columns[None, :] == 0))
        if self.head_edges:
            adj = adj | (head[:, None] == head[None, :])
        adj.fill_diagonal_(False)
        return adj.to(dtype)

    def _states(self, h, adjacency):
        states = []
        for layer, norm, eps in zip(self.layers, self.norms, self.epsilon):
            h = norm(layer((1 + eps) * h + adjacency @ h))
            states.append(h)
        return states[-1:] if self.last_layer else states

    def forward(self, x, columns, heads):
        a = self.adjacency(columns, heads, x.dtype)
        if self.plain_gin:
            # Same graph/layers and matched pool/readout; one entire-graph GIN.
            rooted = self.pool(torch.cat(self._states(x, a), -1))
        else:
            roots = []
            for root in range(x.shape[1]):
                neighbors = (a[root].bool() | (torch.arange(len(columns), device=x.device) == root)).nonzero(as_tuple=True)[0]
                induced = a[neighbors][:, neighbors]
                marker = (neighbors == root).long()
                h = x[:, neighbors]
                if self.markers:
                    h = h + self.root_embedding(marker) + self.distance(1 - marker)
                states = [state.mean(1) for state in self._states(h, induced)]
                roots.append(self.pool(torch.cat(states, -1)))
            rooted = torch.stack(roots, 1)
        return self.readout(torch.cat((x, rooted, rooted.mean(1, keepdim=True).expand_as(x)), -1))


def build(method, latent_dim=256, num_heads=8, value_dim=64):
    if method not in NESTED_SWEEP:
        raise ValueError('Unknown bounded Nested comparison: ' + method)
    options = dict(NESTED_SWEEP[method])
    groups = options.pop('groups', num_heads)
    if groups < 1 or num_heads % groups:
        raise ValueError('Nested groups must divide the actual OSRAM head count')
    grouped_value_dim = (num_heads // groups) * value_dim
    dim = options.get('dim', 64)
    return TokenAdapter(NestedSweepGNN(**options), latent_dim, groups, grouped_value_dim, dim)
