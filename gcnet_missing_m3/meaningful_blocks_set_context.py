"""Complete per-utterance set/context cores; independent mathematical ports.

The source pins, deliberate adaptations and licenses are recorded in
experiments/osram_meaningful20_20261003/set_context.json. In particular GMT
is an equation-based implementation, not vendored/transliterated source.
These modules return features, not logits or a new loss. The shared outer
wrapper owns history/padding exclusion and the zero-initialized Flat bridge.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .meaningful_blocks_common import HeadTokenizer, active_groups


class _EvidenceSetReadout(nn.Module):
    output_dim = 128

    def __init__(self, latent_dim, num_heads, value_dim):
        super().__init__()
        if min(latent_dim, num_heads, value_dim) <= 0:
            raise ValueError('Set/context dimensions must be positive')
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.value_dim = value_dim
        self.tokenizer = HeadTokenizer(latent_dim, num_heads, value_dim,
                                       dim=128, shared_projection=False, normalize=False)

    def forward(self, local, evidence, active, availability):
        if (local.ndim != 2 or local.shape[1] != self.latent_dim
                or evidence.shape != (local.shape[0], 4, self.num_heads * self.value_dim)
                or active.shape != (local.shape[0], 4)
                or availability.shape != (local.shape[0], 3)):
            raise ValueError('Set/context expects compact Local and four real-head evidence roles')
        result = local.new_zeros((local.shape[0], self.output_dim))
        if local.shape[0] == 0 or not active.bool().any():
            return result
        tokens, mask = self.tokenizer(local, evidence, active)
        for rows, columns, packed in active_groups(tokens, mask):
            result[rows] = self._forward_group(local[rows], packed, columns)
        return result


class _PerceiverAttention(nn.Module):
    """Projected multihead attention with head-width, not total-width, scaling."""

    def __init__(self, dim=128, heads=4):
        super().__init__()
        if dim % heads:
            raise ValueError('Attention width must be divisible by heads')
        self.heads, self.head_dim = heads, dim // heads
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, dim)

    def forward(self, query, context):
        def split(value):
            return value.reshape(value.shape[0], value.shape[1], self.heads,
                                 self.head_dim).transpose(1, 2)
        q, k, v = split(self.query(query)), split(self.key(context)), split(self.value(context))
        weights = (q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)).softmax(-1)
        attended = (weights @ v).transpose(1, 2).reshape(query.shape)
        return self.output(attended)


class _PerceiverCrossAttention(nn.Module):
    def __init__(self, dim=128, heads=4, query_residual=True):
        super().__init__()
        self.query_residual = query_residual
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attention = _PerceiverAttention(dim, heads)
        self.feedforward_norm = nn.LayerNorm(dim)
        self.feedforward = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, query, context):
        value = self.attention(self.query_norm(query), self.context_norm(context))
        if self.query_residual:
            value = query + value
        return value + self.feedforward(self.feedforward_norm(value))


class _PerceiverSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_norm = nn.LayerNorm(128)
        self.attention = _PerceiverAttention()
        self.feedforward_norm = nn.LayerNorm(128)
        self.feedforward = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 128))

    def forward(self, latents):
        normalized = self.attention_norm(latents)
        value = latents + self.attention(normalized, normalized)
        return value + self.feedforward(self.feedforward_norm(value))


class PerceiverIOReadout(_EvidenceSetReadout):
    """Eight latent slots, three latent processors, and one Local-derived query."""

    def __init__(self, latent_dim, num_heads, value_dim):
        super().__init__(latent_dim, num_heads, value_dim)
        self.latents = nn.Parameter(torch.empty(8, 128))
        nn.init.normal_(self.latents, std=.02)
        self.encode = _PerceiverCrossAttention(query_residual=True)
        self.process = nn.ModuleList([_PerceiverSelfAttention() for _ in range(3)])
        self.query = nn.Linear(latent_dim, 128, bias=False)
        self.query_bias = nn.Parameter(torch.zeros(128))
        self.decode = _PerceiverCrossAttention(query_residual=False)

    def _forward_group(self, local, tokens, columns):
        latents = self.encode(self.latents[None].expand(tokens.shape[0], -1, -1), tokens)
        for layer in self.process:
            latents = layer(latents)
        query = (self.query(local) + self.query_bias)[:, None]
        return self.decode(query, latents)[:, 0]


def _stable_knn(features, k, canonical_ids):
    """Feature-space neighbors; exact ties follow semantic role/head position."""
    count = features.shape[1]
    k = min(k, count)
    norms = features.square().sum(-1)
    distance = (norms[:, :, None] + norms[:, None, :]
                - 2 * features @ features.transpose(1, 2)).clamp_min(0)
    # Sorting IDs first, then a stable distance sort, avoids perturbing distances
    # with an epsilon that could reorder genuinely different nearby points.
    canonical_order = torch.argsort(canonical_ids, stable=True)
    ordered_distance = distance[:, :, canonical_order]
    ordered_neighbors = torch.argsort(ordered_distance, dim=-1, stable=True)[..., :k]
    return canonical_order[ordered_neighbors]


class _EdgeConv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(2 * input_dim, output_dim, bias=False)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, features, canonical_ids):
        neighbors = _stable_knn(features, 4, canonical_ids)
        batch = torch.arange(features.shape[0], device=features.device)[:, None, None]
        neighbor_features = features[batch, neighbors]
        centers = features[:, :, None].expand_as(neighbor_features)
        edges = torch.cat((neighbor_features - centers, centers), -1)
        return F.leaky_relu(self.norm(self.linear(edges)), .2).max(2).values


class DGCNNReadout(_EvidenceSetReadout):
    """Four recomputed graphs, multiscale concatenation, max and mean readout."""

    def __init__(self, latent_dim, num_heads, value_dim):
        super().__init__(latent_dim, num_heads, value_dim)
        widths = (128, 64, 64, 128, 256)
        self.stages = nn.ModuleList([_EdgeConv(a, b) for a, b in zip(widths, widths[1:])])
        self.multiscale = nn.Sequential(nn.Linear(512, 128, bias=False), nn.LayerNorm(128),
                                        nn.LeakyReLU(.2))
        self.pool_projection = nn.Linear(256, 128)

    def _forward_group(self, local, tokens, columns):
        outputs = []
        for stage in self.stages:
            tokens = stage(tokens, columns)
            outputs.append(tokens)
        tokens = self.multiscale(torch.cat(outputs, -1))
        pooled = torch.cat((tokens.max(1).values, tokens.mean(1)), -1)
        return F.leaky_relu(self.pool_projection(pooled), .2)


def _normalized_graph(role_ids, head_ids, dtype):
    """Local star plus same-head cross-role edges, with self loops exactly once."""
    local = role_ids == 0
    memory = ~local
    same_head = head_ids[:, None] == head_ids[None, :]
    adjacency = (local[:, None] | local[None, :]
                 | (memory[:, None] & memory[None, :] & same_head))
    adjacency = adjacency | torch.eye(role_ids.numel(), device=role_ids.device, dtype=torch.bool)
    adjacency = adjacency.to(dtype=dtype)
    inverse_root_degree = adjacency.sum(-1).rsqrt()
    return inverse_root_degree[:, None] * adjacency * inverse_root_degree[None, :]


class _GraphLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, features, adjacency):
        # Bias is added after propagation, matching normalized GCN semantics.
        return self.linear(adjacency @ features)


class _GMTMAB(nn.Module):
    """GMT's projected-query residual and total-width attention scaling."""

    def __init__(self, input_dim, output_dim, heads=4, graph_keys=False):
        super().__init__()
        if output_dim % heads:
            raise ValueError('GMT width must be divisible by heads')
        self.output_dim, self.heads = output_dim, heads
        self.graph_keys = graph_keys
        self.query = nn.Linear(input_dim, output_dim)
        constructor = _GraphLinear if graph_keys else nn.Linear
        self.key = constructor(input_dim, output_dim)
        self.value = constructor(input_dim, output_dim)
        self.norm1 = nn.LayerNorm(output_dim)
        self.feedforward = nn.Linear(output_dim, output_dim)
        self.norm2 = nn.LayerNorm(output_dim)

    def forward(self, query, context, adjacency=None):
        def split(value):
            return value.reshape(value.shape[0], value.shape[1], self.heads,
                                 self.output_dim // self.heads).transpose(1, 2)
        if self.graph_keys:
            if adjacency is None:
                raise ValueError('Graph-aware pooling requires normalized adjacency')
            keys, values = self.key(context, adjacency), self.value(context, adjacency)
        else:
            keys, values = self.key(context), self.value(context)
        q, k, v = split(self.query(query)), split(keys), split(values)
        weights = (q @ k.transpose(-1, -2) / math.sqrt(self.output_dim)).softmax(-1)
        output = (q + weights @ v).transpose(1, 2).reshape(query.shape[0], query.shape[1], self.output_dim)
        output = self.norm1(output)
        return self.norm2(output + F.relu(self.feedforward(output)))


class GraphMultisetReadout(_EvidenceSetReadout):
    """Two GCNs and GMPool_G -> interseed SelfAtt -> GMPool_I."""

    def __init__(self, latent_dim, num_heads, value_dim):
        super().__init__(latent_dim, num_heads, value_dim)
        self.encoder = nn.ModuleList([_GraphLinear(128, 128) for _ in range(2)])
        self.graph_seeds = nn.Parameter(torch.empty(4, 256))
        self.final_seed = nn.Parameter(torch.empty(1, 128))
        nn.init.xavier_uniform_(self.graph_seeds)
        nn.init.xavier_uniform_(self.final_seed)
        self.pool_graph = _GMTMAB(256, 256, graph_keys=True)
        self.interseed = _GMTMAB(256, 128)
        self.pool_final = _GMTMAB(128, 128)
        self.final = nn.Linear(128, 128)

    def _forward_group(self, local, tokens, columns):
        adjacency = _normalized_graph(self.tokenizer.role_ids[columns],
                                      self.tokenizer.head_ids[columns], tokens.dtype)
        encoded = []
        for layer in self.encoder:
            tokens = F.relu(layer(tokens, adjacency))
            encoded.append(tokens)
        multi_scale = torch.cat(encoded, -1)
        pooled = self.pool_graph(self.graph_seeds[None].expand(tokens.shape[0], -1, -1),
                                 multi_scale, adjacency)
        pooled = self.interseed(pooled, pooled)
        final = self.pool_final(self.final_seed[None].expand(tokens.shape[0], -1, -1), pooled)
        return self.final(final[:, 0])


def build_set_context(method, latent_dim, num_heads, value_dim):
    factories = {'perceiver_io': PerceiverIOReadout,
                 'dgcnn_dynamic_edgeconv': DGCNNReadout,
                 'graph_multiset_transformer': GraphMultisetReadout}
    if method not in factories:
        raise ValueError('Unknown accepted set/context mechanism: ' + str(method))
    return factories[method](latent_dim, num_heads, value_dim)
