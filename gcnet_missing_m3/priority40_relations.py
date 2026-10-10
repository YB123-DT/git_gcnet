"""Priority A/H: five REAL role tokens -> one R vector (64 at default).

These are historical mechanism baselines, not ten new literature contributions.
No 33-head tokenizer, batch attention, persistent sample state or auxiliary loss.
The caller owns the zero-initialized raw-slot bridge; these cores are not zeroed.

Source anchors checked against existing implementations and primary source code:
SAB: juho-lee/set_transformer/modules.py; DAT: Awni00/dual-attention,
relational_attention.py at dce218cbf5ec9aa7f90687c1323050a1fba17966;
GATv2: tech-srl/how_attentive_are_gats/gatv2_conv_PyG.py, message;
EdgeConv: Wang et al. arXiv:1801.07829, edge function + symmetric max;
PNA: Corso et al. arXiv:2004.05718, four aggregators/three degree scalers;
Hopfield: Ramsauer et al. arXiv:2008.02217, ONE update = static-pattern attention;
Entmax: deep-spin/entmax/activations.py, Entmax15Function threshold;
SoftMoE: google-research/vmoe/projects/soft_moe/router.py, same-logit D/C;
NODE: Qwicen/node/lib/odst.py, feature selection/path products/data-aware init;
Capsules: Sabour et al. arXiv:1710.09829, Procedure 1, three routing iterations.
Complete graph, five-role flatten readouts and bounded sizes are explicit task
adaptations, not claims to reproduce the original benchmark architectures.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def safe_mask(x, mask):
    return torch.where(mask[..., None], x, torch.zeros_like(x))


def masked_softmax(scores, allowed, dim=-1):
    weights = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min).softmax(dim)
    weights = torch.where(allowed, weights, torch.zeros_like(weights))
    return weights / weights.sum(dim, keepdim=True).clamp_min(torch.finfo(weights.dtype).eps)


def masked_entmax15(scores, allowed):
    """Exact alpha=1.5 threshold; empty rows return zeros, including gradients."""
    allowed = allowed.expand_as(scores).bool()
    nonempty = allowed.any(-1, keepdim=True)
    first = torch.arange(scores.shape[-1], device=scores.device) == 0
    eligible = allowed | (~nonempty & first)
    maximum = scores.masked_fill(~eligible, -torch.inf).max(-1, keepdim=True).values
    centered = (scores - maximum) / 2
    ordered, indices = centered.masked_fill(~eligible, -torch.inf).sort(-1, descending=True)
    valid = eligible.gather(-1, indices)
    ordered = torch.where(valid, ordered, torch.zeros_like(ordered))
    rank = torch.arange(1, scores.shape[-1] + 1, dtype=scores.dtype, device=scores.device)
    mean = ordered.cumsum(-1) / rank
    variance_sum = ordered.square().cumsum(-1) - rank * mean.square()
    threshold = mean - ((1 - variance_sum) / rank).clamp_min(torch.finfo(scores.dtype).tiny).sqrt()
    support = (valid & (threshold <= ordered)).sum(-1, keepdim=True).clamp_min(1)
    tau = threshold.gather(-1, support - 1)
    output = (centered - tau).clamp_min(0).square()
    return torch.where(allowed, output, torch.zeros_like(output))


def mlp(input_dim, output_dim, hidden=None):
    return nn.Sequential(nn.Linear(input_dim, hidden or output_dim), nn.GELU(), nn.Linear(hidden or output_dim, output_dim))


def split_heads(x, heads):
    return x.reshape(x.shape[0], x.shape[1], heads, -1).transpose(1, 2)


def join_heads(x):
    return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], -1)


class _Readout(nn.Module):
    historical_mechanism_baseline = True

    def __init__(self, dim=64, flatten=True):
        super().__init__()
        self.dim = dim
        self.output_dim = dim
        if flatten:
            self.readout = nn.Linear(5 * dim, dim)

    def pool(self, x, mask):
        return self.readout(safe_mask(x, mask).flatten(1))

    def core(self, tokens, mask):
        if tokens.ndim != 3 or tokens.shape[1:] != (5, self.dim) or mask.shape != tokens.shape[:2]:
            raise ValueError("expected tokens [N,5,dim] and mask [N,5]")
        mask = mask.bool()
        result = self.encode(safe_mask(tokens, mask), mask)
        return safe_mask(result, mask.any(-1))

    def forward(self, tokens, mask):
        return self.core(tokens, mask)


class SAB(_Readout):
    """One four-head SAB + row-wise FFN; M37 changes ONLY the attention simplex."""
    def __init__(self, dim=64, sparse=False):
        super().__init__(dim)
        self.sparse = sparse
        self.qkv = nn.Linear(dim, 3 * dim)
        self.output = nn.Linear(dim, dim)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ffn = mlp(dim, dim, 2 * dim)

    def encode(self, x, mask):
        q, k, v = [split_heads(part, 4) for part in self.qkv(x).chunk(3, -1)]
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.dim // 4)
        allowed = mask[:, None, None, :]
        attention = masked_entmax15(scores, allowed) if self.sparse else masked_softmax(scores, allowed)
        x = safe_mask(self.norm1(x + self.output(join_heads(attention @ v))), mask)
        x = safe_mask(self.norm2(x + self.ffn(x)), mask)
        return self.pool(x, mask)


class DAT(_Readout):
    """Two sensory + two relational heads; selection and four relation values separate."""
    def __init__(self, dim=64):
        super().__init__(dim)
        self.sensory_qkv = nn.Linear(dim, 3 * dim // 2, bias=False)
        self.selection_qk = nn.Linear(dim, dim, bias=False)
        self.relation_qk = nn.Linear(dim, dim, bias=False)
        self.symbols = nn.Parameter(torch.randn(5, dim) * 0.1)
        self.symbol_value = nn.Linear(dim, dim // 2, bias=False)
        self.relation_value = nn.Parameter(torch.randn(2, dim // 4, 4) / math.sqrt(4))
        self.output = nn.Linear(dim, dim)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ffn = mlp(dim, dim, 2 * dim)

    def encode(self, x, mask):
        q, k, v = [split_heads(part, 2) for part in self.sensory_qkv(x).chunk(3, -1)]
        allowed = mask[:, None, None, :]
        attention = masked_softmax(q @ k.transpose(-1, -2) / math.sqrt(self.dim // 4), allowed)
        sensory = join_heads(attention @ v)
        aq, ak = [split_heads(part, 2) for part in self.selection_qk(x).chunk(2, -1)]
        selection = masked_softmax(aq @ ak.transpose(-1, -2) / math.sqrt(self.dim // 4), allowed)
        rq, rk = [split_heads(part, 4) for part in self.relation_qk(x).chunk(2, -1)]
        relations = (rq @ rk.transpose(-1, -2) / math.sqrt(self.dim // 8)).permute(0, 2, 3, 1)
        relational = torch.einsum("bhij,bijr,hdr->bhid", selection, relations, self.relation_value)
        # Positional role symbols are distinct from feature-derived values.
        symbols = self.symbol_value(self.symbols).reshape(5, 2, self.dim // 4).permute(1, 0, 2)
        relational = relational + torch.einsum("bhij,hjd->bhid", selection, symbols)
        combined = torch.cat((sensory, join_heads(relational)), dim=-1)
        x = safe_mask(self.norm1(x + self.output(combined)), mask)
        x = safe_mask(self.norm2(x + self.ffn(x)), mask)
        return self.pool(x, mask)


class GATv2(_Readout):
    """Complete available graph including self-loops; nonlinear-before-score ordering."""
    def __init__(self, dim=64):
        super().__init__(dim)
        self.source, self.target = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.attention = nn.Parameter(torch.randn(4, dim // 4) / math.sqrt(dim // 4))
        self.bias = nn.Parameter(torch.zeros(dim))

    def encode_roles(self, x, mask):
        mask = mask.bool()
        x = safe_mask(x, mask)
        source, target = split_heads(self.source(x), 4), split_heads(self.target(x), 4)
        pairs = F.leaky_relu(target[:, :, :, None] + source[:, :, None, :], 0.2)
        scores = torch.einsum("bhijd,hd->bhij", pairs, self.attention)
        weights = masked_softmax(scores, mask[:, None, None, :])
        updated = safe_mask(join_heads(weights @ source) + self.bias, mask)
        return updated

    def encode(self, x, mask):
        return self.pool(self.encode_roles(x, mask), mask)


class EdgeConv(_Readout):
    """Single complete-role EdgeConv; deliberately no kNN recomputation."""
    def __init__(self, dim=64):
        super().__init__(dim)
        self.edge = mlp(2 * dim, dim)

    def encode(self, x, mask):
        target, source = x[:, :, None].expand(-1, -1, 5, -1), x[:, None, :].expand(-1, 5, -1, -1)
        edge = self.edge(torch.cat((target, source - target), dim=-1))
        edge_mask = mask[:, :, None] & mask[:, None, :]
        updated = edge.masked_fill(~edge_mask[..., None], torch.finfo(edge.dtype).min).max(2).values
        return self.pool(safe_mask(updated, mask), mask)


class PNA(_Readout):
    """One layer, mean/max/min/std × identity/amplification/attenuation.

    log(6) is the fixed full-five-node self-loop degree reference, not batch
    statistics. Neighborhood size is the active-node count for each row.
    """
    def __init__(self, dim=64):
        super().__init__(dim)
        self.message = mlp(2 * dim, dim)
        self.update = mlp(13 * dim, dim)
        self.register_buffer("degree_reference", torch.tensor(math.log(6.0)))

    def encode_roles(self, x, mask):
        mask = mask.bool()
        x = safe_mask(x, mask)
        target, source = x[:, :, None].expand(-1, -1, 5, -1), x[:, None, :].expand(-1, 5, -1, -1)
        edges = mask[:, :, None] & mask[:, None, :]
        messages = safe_mask(self.message(torch.cat((target, source), dim=-1)), edges)
        degree = edges.sum(2).clamp_min(1).to(x.dtype)[..., None]
        mean = messages.sum(2) / degree
        std = (messages.square().sum(2) / degree - mean.square()).clamp_min(0).add(1e-5).sqrt()
        maximum = messages.masked_fill(~edges[..., None], torch.finfo(x.dtype).min).max(2).values
        minimum = messages.masked_fill(~edges[..., None], torch.finfo(x.dtype).max).min(2).values
        stats = safe_mask(torch.cat((mean, maximum, minimum, std), dim=-1), mask)
        scale = degree.log1p() / self.degree_reference.to(x.dtype)
        scaled = torch.cat((stats, stats * scale, stats / scale), dim=-1)
        return safe_mask(self.update(torch.cat((x, scaled), dim=-1)), mask)

    def encode(self, x, mask):
        return self.pool(self.encode_roles(x, mask), mask)


class Hopfield(_Readout):
    """16 trainable STATIC patterns, one association step; attention-equivalent baseline."""
    def __init__(self, dim=64):
        super().__init__(dim)
        self.query = nn.Linear(dim, dim, bias=False)
        self.patterns = nn.Parameter(torch.randn(16, dim) / math.sqrt(dim))

    def encode(self, x, mask):
        weights = (self.query(x) @ self.patterns.t() / math.sqrt(self.dim)).softmax(-1)
        return self.pool(safe_mask(weights @ self.patterns, mask), mask)


class SoftMoE(_Readout):
    """Four independent experts, one slot each, shared-logit dual-axis routing."""
    def __init__(self, dim=64):
        super().__init__(dim)
        self.slots = nn.Parameter(torch.randn(4, dim) * 0.02)
        self.logit_scale = nn.Parameter(torch.ones(()))
        self.experts = nn.ModuleList(mlp(dim, dim, 2 * dim) for _ in range(4))

    def routing(self, x, mask):
        x = safe_mask(x, mask)
        logits = self.logit_scale * torch.einsum("nrd,ed->nre", F.normalize(x, dim=-1), F.normalize(self.slots, dim=-1))
        return masked_softmax(logits, mask[:, :, None], 1), safe_mask(logits.softmax(-1), mask)

    def encode(self, x, mask):
        dispatch, combine = self.routing(x, mask)
        inputs = torch.einsum("nre,nrd->ned", dispatch, x)
        outputs = torch.stack([expert(inputs[:, i]) for i, expert in enumerate(self.experts)], dim=1)
        return self.pool(safe_mask(torch.einsum("nre,ned->nrd", combine, outputs), mask), mask)


class NODE(_Readout):
    """Eight depth-three ODSTs, eight-dimensional leaves at dim=64.

    First nonempty TRAIN forward initializes thresholds and scales. Evaluation
    before training uses finite neutral defaults and NEVER changes the state.
    Masked feature coordinates are excluded from sparse feature selection.
    Use the entmax1.5/entmoid1.5 variant already used by this repository's NODE,
    rather than silently claiming the author's older sparsemax defaults.
    """
    def __init__(self, dim=64):
        super().__init__(dim, flatten=False)
        self.choices = nn.Parameter(torch.empty(8, 3, 5 * dim).uniform_(0, 1))
        self.threshold = nn.Parameter(torch.zeros(8, 3))
        self.log_temperature = nn.Parameter(torch.zeros(8, 3))
        self.response = nn.Parameter(torch.randn(8, 8, dim // 8) * 0.1)
        self.register_buffer("initialized", torch.tensor(False))
        self.register_buffer("init_quantiles", torch.rand(8, 3))
        self.register_buffer("leaf_bits", ((torch.arange(8)[None] >> torch.arange(3)[:, None]) & 1).bool())

    @torch.no_grad()
    def initialize(self, values):
        ordered = values.detach().sort(0).values.flatten(1)
        positions = self.init_quantiles * (len(values) - 1)
        lower, upper = positions.floor().long(), positions.ceil().long()
        columns = torch.arange(24, device=values.device)
        low = ordered[lower.flatten(), columns].reshape(8, 3)
        high = ordered[upper.flatten(), columns].reshape(8, 3)
        self.threshold.copy_(low + (high - low) * (positions - lower))
        self.log_temperature.copy_((values.detach() - self.threshold).abs().amax(0).add(1e-6).log())
        self.initialized.fill_(True)

    def encode(self, x, mask):
        flat = x.flatten(1)
        eligible = mask[..., None].expand(-1, -1, self.dim).flatten(1)
        selectors = masked_entmax15(self.choices[None].expand(len(x), -1, -1, -1), eligible[:, None, None])
        values = torch.einsum("nf,ntdf->ntd", flat, selectors)
        valid = mask.any(-1)
        if self.training and not bool(self.initialized) and bool(valid.any()):
            self.initialize(values[valid])
        logits = (values - self.threshold) * (-self.log_temperature).exp()
        # entmoid15(t) = entmax15([t, 0])[0], not entmax15([-t, t])[1]
        # (the latter would silently double the inverse temperature).
        bins = torch.stack((logits, torch.zeros_like(logits)), dim=-1)
        positive = masked_entmax15(bins, torch.ones_like(bins, dtype=torch.bool))[..., 0]
        probabilities = torch.stack((1 - positive, positive), dim=-1)
        paths = torch.where(self.leaf_bits[None, None], probabilities[..., 1, None], probabilities[..., 0, None]).prod(2)
        return torch.einsum("ntl,tlc->ntc", paths, self.response).flatten(1)


class Capsule(_Readout):
    """Five child capsules -> four 16-D parents, three routing-by-agreement rounds."""
    def __init__(self, dim=64):
        super().__init__(dim, flatten=False)
        self.votes = nn.Parameter(torch.randn(5, 4, dim, dim // 4) / math.sqrt(dim))

    def encode(self, x, mask):
        votes = torch.einsum("nid,ipdc->nipc", x, self.votes)
        votes = torch.where(mask[:, :, None, None], votes, torch.zeros_like(votes))
        logits = x.new_zeros(x.shape[0], 5, 4)
        for iteration in range(3):
            coupling = safe_mask(logits.softmax(-1), mask)
            total = torch.einsum("nip,nipc->npc", coupling, votes)
            squared = total.square().sum(-1, keepdim=True)
            output = squared / (1 + squared) * total / squared.clamp_min(1e-8).sqrt()
            if iteration < 2:
                logits = logits + torch.einsum("nipc,npc->nip", votes, output)
        return output.flatten(1)


METHODS = {
    "m01_sab": SAB, "m02_dat": DAT, "m03_gatv2": GATv2,
    "m04_edgeconv": EdgeConv, "m05_pna": PNA, "m36_hopfield": Hopfield,
    "m37_entmax_sab": SAB, "m38_soft_moe": SoftMoE, "m39_node": NODE,
    "m40_capsule": Capsule,
}


def build(method, dim=64):
    if dim < 8 or dim % 8:
        raise ValueError("dim must be a positive multiple of eight")
    if method not in METHODS:
        raise ValueError(f"Unknown priority relation method: {method}")
    return SAB(dim, sparse=True) if method == "m37_entmax_sab" else METHODS[method](dim)
