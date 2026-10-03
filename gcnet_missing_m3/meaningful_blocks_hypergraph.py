"""Complete, ephemeral hypergraph cores for already-read OSRAM evidence.

Fixed cards: AllSet (two V-E-V PMA rounds, 4 heads), ED-HNN (three
shared iterations, alpha=.1), Hyper-SAGNN (one static/dynamic discrepancy
encoder, 4 heads), diagonal SheafHyperGNN (two layers, stalk4 x channel32).
Inputs/task head are adaptations; these are not source-task reproductions.
See experiments/osram_meaningful20_20261003/hypergraph.json for pinned
papers/code and the sheaf paper/code operator convention. Independent
PyTorch equations; no vendored source, PyG, dropout, memory or batch state.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def make_incidence(role_ids, head_ids, num_heads):
    """Active nodes x (same-head edges followed by active evidence-type edges)."""
    local = role_ids == 0
    head_edges = local[:, None] | (head_ids[:, None] == torch.arange(num_heads, device=role_ids.device))
    types = torch.unique(role_ids[~local], sorted=True)
    type_edges = local[:, None] | (role_ids[:, None] == types)
    return torch.cat((head_edges, type_edges), dim=1)


def _mlp(input_dim, dim, activation=nn.ReLU):
    return nn.Sequential(nn.Linear(input_dim, dim), activation(), nn.Linear(dim, dim))


class EquivariantDiffusion(nn.Module):
    """Sum phi -> recipient-dependent rho -> sum -> restart -> psi."""
    def __init__(self, dim=128):
        super().__init__()
        self.phi = _mlp(dim, dim)
        self.rho = _mlp(2 * dim, dim)
        self.psi = _mlp(dim, dim)

    def forward(self, x, x0, incidence):
        h = incidence.to(x.dtype)
        edge = torch.einsum('ve,bvd->bed', h, self.phi(x))
        b, n, d = x.shape
        count = edge.shape[1]
        pairs = torch.cat((x[:, :, None].expand(b, n, count, d),
                           edge[:, None].expand(b, n, count, d)), dim=-1)
        returned = torch.where(incidence[None, :, :, None], self.rho(pairs), 0).sum(2)
        return self.psi(.9 * returned + .1 * x0).relu()


class _HypergraphBase(nn.Module):
    output_dim = 128

    def __init__(self, latent_dim, num_heads, value_dim):
        super().__init__()
        from .meaningful_blocks_common import HeadTokenizer
        self.num_heads = num_heads
        self.tokenizer = HeadTokenizer(latent_dim, num_heads, value_dim, dim=128,
                                       shared_projection=False, normalize=False)
        self.readout = nn.Linear(384, 128)

    def summarize(self, x, roles):
        local = x[:, roles == 0].mean(1)
        base = x[:, roles == 1].mean(1)
        gaps = x[:, roles >= 2]
        gap = gaps.mean(1) if gaps.shape[1] else torch.zeros_like(local)
        return self.readout(torch.cat((local, base, gap), -1))

    def forward(self, local, evidence, active, availability):
        from .meaningful_blocks_common import active_groups
        # An all-inactive standalone row is not a Local-only graph. The outer
        # wrapper normally removes these rows together with padding/no-history.
        eligible = active[:, 0].bool()
        result = local.new_zeros((local.shape[0], self.output_dim))
        if not bool(eligible.any()):
            return result
        rows = eligible.nonzero(as_tuple=True)[0]
        tokens, mask = self.tokenizer(local[rows], evidence[rows], active[rows])
        for group_rows, columns, packed in active_groups(tokens, mask):
            roles = self.tokenizer.role_ids[columns]
            heads = self.tokenizer.head_ids[columns]
            incidence = make_incidence(roles, heads, self.num_heads)
            features = self.process(packed, incidence, roles)
            result = result.index_copy(0, rows[group_rows], features)
        return result


class EDHNN(_HypergraphBase):
    iterations = 3

    def __init__(self, latent_dim, num_heads, value_dim):
        super().__init__(latent_dim, num_heads, value_dim)
        self.input = nn.Linear(128, 128)
        self.diffusion = EquivariantDiffusion()

    def process(self, x, incidence, roles):
        x0 = self.input(x).relu()
        x = x0
        for _ in range(self.iterations):
            x = self.diffusion(x, x0, incidence)
        return self.summarize(x, roles)


class SeedPooling(nn.Module):
    """Author-style PMA: learned seed scores, seed residual, two layer norms."""
    def __init__(self, dim=128, heads=4):
        super().__init__()
        if dim % heads:
            raise ValueError('PMA dimension must be divisible by heads')
        self.heads, self.width = heads, dim // heads
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.seed = nn.Parameter(torch.empty(heads, self.width))
        nn.init.xavier_uniform_(self.seed)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = _mlp(dim, dim)

    def forward(self, x, membership):
        # membership[target, source]; graph construction forbids empty targets.
        b, n, _ = x.shape
        k = self.key(x).reshape(b, n, self.heads, self.width)
        v = self.value(x).reshape(b, n, self.heads, self.width)
        logits = F.leaky_relu((k * self.seed).sum(-1), .2)
        weights = logits[:, None].masked_fill(~membership[None, :, :, None], -torch.inf).softmax(2)
        pooled = torch.einsum('btnh,bnhd->bthd', weights, v) + self.seed
        y = self.norm1(pooled.flatten(-2))
        return self.norm2(y + self.ffn(y).relu())


class AllSetTransformer(_HypergraphBase):
    def __init__(self, latent_dim, num_heads, value_dim):
        super().__init__(latent_dim, num_heads, value_dim)
        self.rounds = nn.ModuleList([nn.ModuleList([SeedPooling(), SeedPooling()]) for _ in range(2)])

    def process(self, x, incidence, roles):
        for to_edge, to_node in self.rounds:
            edges = to_edge(x, incidence.T).relu()
            x = to_node(edges, incidence).relu()
        return self.summarize(x, roles)


class StaticDynamicDiscrepancy(nn.Module):
    """Hyper-SAGNN's active static/dynamic paths, excluding dead fc2 code.

    Source EncoderLayer feeds ORIGINAL static inputs into pff_n2 despite
    computing an unused attention-value static projection. We implement that
    effective path, tanh positionwise activations and both final LayerNorms.
    Source sigmoid hyperedge classifier is replaced by vector discrepancies.
    """
    def __init__(self, dim=128, heads=4):
        super().__init__()
        if dim % heads:
            raise ValueError('Attention dimension must be divisible by heads')
        self.heads, self.width = heads, dim // heads
        self.query_norm = nn.LayerNorm(dim)
        self.key_norm = nn.LayerNorm(dim)
        self.value_norm = nn.LayerNorm(dim)
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        self.dynamic_ffn = _mlp(dim, dim, nn.Tanh)
        self.static_ffn = _mlp(dim, dim, nn.Tanh)
        self.dynamic_norm = nn.LayerNorm(dim)
        self.static_norm = nn.LayerNorm(dim)
        self.dynamic_final_norm = nn.LayerNorm(dim)
        self.static_final_norm = nn.LayerNorm(dim)

    def embeddings(self, x):
        b, n, dim = x.shape
        if n < 2:
            raise ValueError('Leave-self-out hyperedge requires at least two nodes')
        def split(value):
            return value.reshape(b, n, self.heads, self.width).transpose(1, 2)
        q = split(self.query(self.query_norm(x)))
        k = split(self.key(self.key_norm(x)))
        v = split(self.value(self.value_norm(x)))
        logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.width)
        logits = logits.masked_fill(torch.eye(n, dtype=torch.bool, device=x.device), -torch.inf)
        attended = (logits.softmax(-1) @ v).transpose(1, 2).reshape(b, n, dim)
        projected = self.output(attended)
        dynamic = self.dynamic_final_norm(self.dynamic_norm(projected + self.dynamic_ffn(projected)))
        static = self.static_final_norm(self.static_norm(self.static_ffn(x)))
        return dynamic, static

    def forward(self, x):
        dynamic, static = self.embeddings(x)
        return (dynamic - static).square().mean(1)


class HyperSAGNN(_HypergraphBase):
    def __init__(self, latent_dim, num_heads, value_dim):
        super().__init__(latent_dim, num_heads, value_dim)
        self.discrepancy = StaticDynamicDiscrepancy()

    def process(self, x, incidence, roles):
        b, _, dim = x.shape
        # Same-head groups all have the same size; evidence-type groups have
        # H+1 nodes. Process each family in one batched encoder invocation.
        summaries = []
        for membership in (incidence[:, :self.num_heads].T, incidence[:, self.num_heads:].T):
            edge_count = membership.shape[0]
            node_count = int(membership[0].sum())
            indices = membership.nonzero(as_tuple=True)[1].reshape(edge_count, node_count)
            gathered = x[:, indices].reshape(b * edge_count, node_count, dim)
            summaries.append(self.discrepancy(gathered).reshape(b, edge_count, dim).mean(1))
        return self.readout(torch.cat((x[:, roles == 0].squeeze(1), *summaries), -1))


def sheaf_propagation(maps, incidence):
    """Pinned author-code P=D^-1/2+A-2*blockdiag(A), not generic I-L.

    maps is [batch,node,edge,stalk]. B uses true hyperedge cardinality;
    D sums squared restriction coefficients. The source applies D^-1/2 to
    the input BEFORE its I+Q-2*blockdiag(Q) operation, so the identity term
    becomes D^-1/2 after combining operators. No cache survives forward.
    """
    maps = torch.where(incidence[None, :, :, None], maps, 0)
    degree = maps.square().sum(2).clamp_min(1e-6)
    normalized = maps * degree.rsqrt()[:, :, None]
    inverse_cardinality = incidence.sum(0).to(maps.dtype).reciprocal()
    a = torch.einsum('bned,bmed,e->bdnm', normalized, normalized, inverse_cardinality)
    n, d = maps.shape[1], maps.shape[-1]
    eye_n = torch.eye(n, dtype=maps.dtype, device=maps.device)
    p = a * (1 - 2 * eye_n) + eye_n * degree.rsqrt().transpose(1, 2)[..., None]
    # Diagonal maps do not mix stalk coordinates in the transport operator.
    eye_d = torch.eye(d, dtype=maps.dtype, device=maps.device)
    return torch.einsum('bdnm,dk->bndmk', p, eye_d).reshape(maps.shape[0], n*d, n*d)


class DiagonalSheafLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.predict = nn.Linear(64, 4)
        self.left = nn.Linear(4, 4, bias=False)
        self.right = nn.Linear(32, 32, bias=False)
        self.bias = nn.Parameter(torch.zeros(32))

    def forward(self, x, incidence):
        b, n, _ = x.shape
        fibers = x.reshape(b, n, 4, 32)
        node = fibers.mean(2)
        h = incidence.to(x.dtype)
        edge = torch.einsum('ve,bvc->bec', h, node) / h.sum(0)[None, :, None]
        edges = edge.shape[1]
        pairs = torch.cat((node[:, :, None].expand(b, n, edges, 32),
                           edge[:, None].expand(b, n, edges, 32)), -1)
        maps = self.predict(pairs).tanh()
        propagation = sheaf_propagation(maps, incidence)
        transformed = self.right(self.left(fibers.transpose(-2, -1)).transpose(-2, -1))
        transported = propagation @ transformed.reshape(b, n * 4, 32)
        return (transported + self.bias).reshape(b, n, 128)


class SheafHyperGNN(_HypergraphBase):
    def __init__(self, latent_dim, num_heads, value_dim):
        super().__init__(latent_dim, num_heads, value_dim)
        self.layers = nn.ModuleList([DiagonalSheafLayer(), DiagonalSheafLayer()])

    def process(self, x, incidence, roles):
        x = F.elu(self.layers[0](x, incidence))
        x = self.layers[1](x, incidence)
        return self.summarize(x, roles)


def build_hypergraph(method, latent_dim=256, num_heads=8, value_dim=64):
    factories = {'ed_hnn': EDHNN, 'allset_transformer': AllSetTransformer,
                 'hyper_sagnn': HyperSAGNN, 'sheaf_hypergnn_diag': SheafHyperGNN}
    if method not in factories:
        raise ValueError(f'Unknown hypergraph method: {method}')
    return factories[method](latent_dim, num_heads, value_dim)
