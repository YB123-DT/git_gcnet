"""Independent, paper-derived graph processors for already-read evidence.

No OSRAM operation or cross-utterance state lives here. Source/task deviations
are documented in experiments/osram_meaningful20_20261003/verification/graph.md.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .meaningful_blocks_common import HeadTokenizer, active_groups, safe_mask, typed_means


METHODS = ('rrn_evidence', 'egt_evidence', 'residual_gated_graph_evidence', 'pna_evidence')


def graph_structure(num_heads: int):
    """Return target-by-source adjacency and fourteen static relation features."""
    if num_heads < 1:
        raise ValueError('num_heads must be positive')
    roles = torch.cat([torch.zeros(1, dtype=torch.long), torch.arange(1, 5).repeat_interleave(num_heads)])
    heads = torch.cat([torch.full((1,), -1), torch.arange(num_heads).repeat(4)])
    count = len(roles)
    diagonal = torch.eye(count, dtype=torch.bool)
    local = (roles[:, None] == 0) | (roles[None, :] == 0)
    same_role = roles[:, None] == roles[None, :]
    same_head = (heads[:, None] == heads[None, :]) & ~local
    adjacency = ~diagonal & (local | same_role | same_head)
    role_bits = F.one_hot(roles, 5).float()
    relations = torch.cat([
        role_bits[None].expand(count, -1, -1),
        role_bits[:, None].expand(-1, count, -1),
        torch.stack([diagonal, local, same_role, same_head], -1).float(),
    ], -1)
    return adjacency, relations


def _mlp(input_dim: int, dim: int, layers: int):
    modules = []
    for index in range(layers):
        modules.append(nn.Linear(input_dim if index == 0 else dim, dim))
        if index + 1 < layers:
            modules.append(nn.ReLU())
    return nn.Sequential(*modules)


class RRNCore(nn.Module):
    """Shared pair-message / input-reinjection / LSTM reasoning loop."""

    def __init__(self, dim=128, steps=5):
        super().__init__()
        self.dim, self.steps = dim, steps
        self.edge_embedding = nn.Linear(14, 16)
        self.message = _mlp(2 * dim + 16, dim, 4)
        self.post = _mlp(2 * dim, dim, 4)
        self.cell = nn.LSTMCell(dim, dim)

    def forward(self, x, mask, adjacency, relations):
        x = safe_mask(x, mask)
        h = x
        hidden, cell = torch.zeros_like(x), torch.zeros_like(x)
        edge = safe_mask(self.edge_embedding(relations), adjacency)
        nodes = x.shape[1]
        for _ in range(self.steps):
            target = h[:, :, None].expand(-1, -1, nodes, -1)
            source = h[:, None].expand(-1, nodes, -1, -1)
            messages = self.message(torch.cat([source, target, edge], -1))
            incoming = safe_mask(messages, adjacency).sum(2)
            update = self.post(torch.cat([incoming, x], -1))
            hidden, cell = self.cell(update.reshape(-1, self.dim),
                                     (hidden.reshape(-1, self.dim), cell.reshape(-1, self.dim)))
            hidden = safe_mask(hidden.reshape_as(x), mask)
            cell = safe_mask(cell.reshape_as(x), mask)
            h = hidden
        return h


class EGTLayer(nn.Module):
    """Global attention with persistent edge channels and dynamic centrality."""

    def __init__(self, dim=128, edge_dim=32, heads=8, edge_update=True):
        super().__init__()
        if dim % heads:
            raise ValueError('EGT heads must divide node width')
        self.heads, self.head_dim = heads, dim // heads
        self.edge_update = edge_update
        self.node_norm, self.edge_norm = nn.LayerNorm(dim), nn.LayerNorm(edge_dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.edge_bias = nn.Linear(edge_dim, heads)
        self.edge_gate = nn.Linear(edge_dim, heads)
        self.node_output = nn.Linear(dim, dim)
        self.node_ffn_norm = nn.LayerNorm(dim)
        self.node_ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.ELU(), nn.Linear(2 * dim, dim))
        if edge_update:
            self.edge_output = nn.Linear(heads, edge_dim)
            self.edge_ffn_norm = nn.LayerNorm(edge_dim)
            self.edge_ffn = nn.Sequential(nn.Linear(edge_dim, 2 * edge_dim), nn.ELU(), nn.Linear(2 * edge_dim, edge_dim))

    def forward(self, h, e, mask):
        batch, nodes, dim = h.shape
        q, k, v = self.qkv(self.node_norm(h)).reshape(batch, nodes, 3, self.heads, self.head_dim).unbind(2)
        normalized_edges = self.edge_norm(e)
        scores = torch.einsum('bihd,bjhd->bijh', q, k) / math.sqrt(self.head_dim)
        scores = scores.clamp(-5, 5) + self.edge_bias(normalized_edges)
        pair_mask = mask[:, :, None] & mask[:, None, :]
        gates = safe_mask(self.edge_gate(normalized_edges).sigmoid(), pair_mask)
        # Finite sentinels also make deliberately empty padded rows safe.
        logits = scores.masked_fill(~mask[:, None, :, None], torch.finfo(scores.dtype).min)
        attention = logits.softmax(2) * gates
        values = torch.einsum('bijh,bjhd->bihd', attention, v)
        values = values * torch.log1p(gates.sum(2))[..., None]
        h = safe_mask(h + self.node_output(values.reshape(batch, nodes, dim)), mask)
        h = safe_mask(h + self.node_ffn(self.node_ffn_norm(h)), mask)
        if self.edge_update:
            # The finite pre-mask scores feed edge states, never softmax sentinels.
            e = safe_mask(e + self.edge_output(scores), pair_mask)
            e = safe_mask(e + self.edge_ffn(self.edge_ffn_norm(e)), pair_mask)
        return h, e


class EGTCore(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.edge_embedding = nn.Linear(14, 32)
        # A node-ended readout has no consumer for a final edge-only update.
        self.layers = nn.ModuleList([EGTLayer(dim, edge_update=index < 2) for index in range(3)])
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x, mask, adjacency, relations):
        pair_mask = mask[:, :, None] & mask[:, None, :]
        e = safe_mask(self.edge_embedding(relations), pair_mask)
        h = safe_mask(x, mask)
        for layer in self.layers:
            h, e = layer(h, e, mask)
        return safe_mask(self.final_norm(h), mask)


class _GatedConvolution(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.self_map = nn.Linear(dim, dim, bias=False)
        self.neighbor_map = nn.Linear(dim, dim, bias=False)
        self.gate_target = nn.Linear(dim, dim, bias=False)
        self.gate_source = nn.Linear(dim, dim, bias=False)
        self.edge_bias = nn.Parameter(torch.zeros(dim))
        self.node_bias = nn.Parameter(torch.zeros(dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, adjacency):
        gates = (self.gate_target(x)[:, :, None] + self.gate_source(x)[:, None] + self.edge_bias).sigmoid()
        messages = gates * self.neighbor_map(x)[:, None]
        incoming = safe_mask(messages, adjacency).sum(2)
        return self.norm(self.self_map(x) + incoming + self.node_bias)


class GatedGraphCell(nn.Module):
    """Two graph convolutions and the source architecture's block residual."""

    def __init__(self, dim=128):
        super().__init__()
        self.conv1, self.conv2 = _GatedConvolution(dim), _GatedConvolution(dim)
        self.residual = nn.Linear(dim, dim, bias=False)

    def forward(self, x, mask, adjacency):
        h = safe_mask(self.conv1(x, adjacency).relu(), mask)
        return safe_mask((self.conv2(h, adjacency) + self.residual(x)).relu(), mask)


class GatedGraphCore(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.cells = nn.ModuleList([GatedGraphCell(dim) for _ in range(3)])

    def forward(self, x, mask, adjacency, relations):
        for cell in self.cells:
            x = cell(x, mask, adjacency)
        return x


def pna_degree_reference(num_heads: int):
    """Label-free node-weighted log-degree average over seven legal masks."""
    total, log_sum = 0, 0.
    for groups, multiplicity in ((1, 1), (2, 3), (3, 3)):
        memory_count = num_heads * groups
        memory_degree = num_heads + groups - 1
        total += multiplicity * (1 + memory_count)
        log_sum += multiplicity * (math.log1p(memory_count) + memory_count * math.log1p(memory_degree))
    return log_sum / total


def pna_aggregate(messages, adjacency, delta):
    """Mean/std/min/max times three scalers; neighbor axis is dimension 2."""
    original_dtype = messages.dtype
    if original_dtype in (torch.float16, torch.bfloat16):
        messages = messages.float()
    degree = adjacency.sum(2)
    count = degree.clamp_min(1).to(messages.dtype)[..., None, None]
    masked = safe_mask(messages, adjacency)
    mean = masked.sum(2) / count
    variance = (masked.square().sum(2) / count - mean.square()).clamp_min(0)
    std = (variance + 1e-5).sqrt()
    valid = adjacency[..., None, None]
    minimum = messages.masked_fill(~valid, float('inf')).amin(2)
    maximum = messages.masked_fill(~valid, -float('inf')).amax(2)
    statistics = safe_mask(torch.cat([mean, std, minimum, maximum], -1), degree > 0)
    normalizer = torch.as_tensor(delta, dtype=messages.dtype, device=messages.device)
    log_degree = degree.clamp_min(1).to(messages.dtype).log1p()[..., None, None]
    result = torch.cat([statistics, statistics * log_degree / normalizer,
                        statistics * normalizer / log_degree], -1)
    return result.to(original_dtype)


class PNALayer(nn.Module):
    def __init__(self, dim=128, towers=4, delta=2.3785469096888603):
        super().__init__()
        if dim % towers:
            raise ValueError('PNA towers must divide node width')
        self.towers, self.tower_dim = towers, dim // towers
        width = self.tower_dim
        self.edge_embedding = nn.Linear(14, width)
        self.messages = nn.ModuleList([_mlp(3 * width, width, 2) for _ in range(towers)])
        self.updates = nn.ModuleList([_mlp(13 * width, width, 2) for _ in range(towers)])
        self.mixing = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.register_buffer('degree_reference', torch.tensor(delta, dtype=torch.float64))

    def forward(self, x, mask, adjacency, relations):
        batch, nodes, _ = x.shape
        split = x.reshape(batch, nodes, self.towers, self.tower_dim)
        edge = safe_mask(self.edge_embedding(relations), adjacency)
        all_messages = []
        for index, transform in enumerate(self.messages):
            part = split[:, :, index]
            target = part[:, :, None].expand(-1, -1, nodes, -1)
            source = part[:, None].expand(-1, nodes, -1, -1)
            all_messages.append(transform(torch.cat([target, source, edge], -1)))
        statistics = pna_aggregate(torch.stack(all_messages, 3), adjacency, self.degree_reference)
        updated = [transform(torch.cat([split[:, :, index], statistics[:, :, index]], -1))
                   for index, transform in enumerate(self.updates)]
        return safe_mask(self.norm(self.mixing(torch.cat(updated, -1))).relu(), mask)


class PNACore(nn.Module):
    def __init__(self, num_heads, dim=128):
        super().__init__()
        self.layers = nn.ModuleList([PNALayer(dim, delta=pna_degree_reference(num_heads)) for _ in range(4)])

    def forward(self, x, mask, adjacency, relations):
        for layer in self.layers:
            x = layer(x, mask, adjacency, relations)
        return x


class GraphEvidenceBlock(nn.Module):
    output_dim = 128

    def __init__(self, method, latent_dim, num_heads, value_dim):
        super().__init__()
        if method not in METHODS:
            raise ValueError(f'Unknown graph candidate: {method}')
        self.method, self.num_heads = method, num_heads
        self.tokenizer = HeadTokenizer(latent_dim, num_heads, value_dim, dim=128,
                                       shared_projection=False, normalize=True)
        adjacency, relations = graph_structure(num_heads)
        self.register_buffer('adjacency', adjacency)
        self.register_buffer('relations', relations)
        constructors = {
            'rrn_evidence': RRNCore,
            'egt_evidence': EGTCore,
            'residual_gated_graph_evidence': GatedGraphCore,
            'pna_evidence': lambda: PNACore(num_heads),
        }
        self.core = constructors[method]()
        self.pool = nn.Sequential(nn.Linear(5 * 128, 128), nn.LayerNorm(128))

    def forward(self, local, evidence, active, availability):
        if tuple(active.shape) != (local.shape[0], 4) or tuple(availability.shape) != (local.shape[0], 3):
            raise ValueError('Graph active/availability shapes must be [N,4]/[N,3]')
        effective = torch.cat([active[:, :1].bool(), active[:, 1:].bool() & ~availability.bool()], -1)
        alive = effective.any(-1)
        local = safe_mask(local, alive)
        evidence = safe_mask(evidence, effective)
        tokens, mask = self.tokenizer(local, evidence, effective)
        mask = mask & alive[:, None]
        tokens = safe_mask(tokens, mask)
        if local.shape[0] == 0:
            return tokens.new_empty((0, self.output_dim))
        encoded = torch.zeros_like(tokens)
        # Invalid head slots are absent from the computational graph, not just
        # zero messages inside an otherwise dense 33-node computation.
        for rows, columns, packed in active_groups(tokens, mask):
            adjacency = self.adjacency[columns[:, None], columns]
            relations = self.relations[columns[:, None], columns].to(tokens.dtype)
            count = rows.numel()
            packed_mask = torch.ones(packed.shape[:2], dtype=torch.bool, device=packed.device)
            processed = self.core(packed, packed_mask,
                                  adjacency[None].expand(count, -1, -1),
                                  relations[None].expand(count, -1, -1, -1))
            encoded[rows[:, None], columns] = processed
        pooled = typed_means(encoded, mask, self.num_heads).reshape(local.shape[0], 5 * 128)
        return safe_mask(self.pool(pooled), alive)


def build_graph(method, latent_dim, num_heads, value_dim):
    return GraphEvidenceBlock(method, latent_dim, num_heads, value_dim)
