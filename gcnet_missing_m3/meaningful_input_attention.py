"""Complete attention operators adapting already-read Flat input evidence.

Independent equation implementations; source pins and declared adaptations are
in experiments/osram_meaningful20_round2_20261004/attention_candidates.json.
There is no temporal state, OSRAM operation, auxiliary loss, or LocalSkip change.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .meaningful_blocks_common import HeadTokenizer, active_groups, safe_mask


def _heads(x, count):
    return x.reshape(x.shape[0], x.shape[1], count, -1).transpose(1, 2)


def _join(x):
    return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], -1)


def _rms(x, eps):
    stable = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    return (stable * torch.rsqrt(stable.square().mean(-1, keepdim=True) + eps)).to(x.dtype)


class DifferentialAttention(nn.Module):
    """Paired softmax subtraction, doubled values, headwise RMS and scaling."""

    def __init__(self):
        super().__init__()
        self.query = nn.Linear(128, 128, bias=False)
        self.key = nn.Linear(128, 128, bias=False)
        self.value = nn.Linear(128, 128, bias=False)
        self.output = nn.Linear(128, 128, bias=False)
        self.lambda_query1 = nn.Parameter(torch.randn(16) * .1)
        self.lambda_key1 = nn.Parameter(torch.randn(16) * .1)
        self.lambda_query2 = nn.Parameter(torch.randn(16) * .1)
        self.lambda_key2 = nn.Parameter(torch.randn(16) * .1)
        self.rms_weight = nn.Parameter(torch.ones(32))
        self.lambda_initial = .2

    def forward(self, x):
        batch, count, _ = x.shape
        q = self.query(x).reshape(batch, count, 4, 2, 16).permute(0, 2, 3, 1, 4)
        k = self.key(x).reshape(batch, count, 4, 2, 16).permute(0, 2, 3, 1, 4)
        maps = (q @ k.transpose(-1, -2) / 4.).softmax(-1)
        lam = ((self.lambda_query1 * self.lambda_key1).sum().exp()
               - (self.lambda_query2 * self.lambda_key2).sum().exp()
               + self.lambda_initial)
        weights = maps[:, :, 0] - lam * maps[:, :, 1]
        values = weights @ _heads(self.value(x), 4)
        values = _rms(values, 1e-5) * self.rms_weight * (1 - self.lambda_initial)
        return self.output(_join(values))


class FlowAttention(nn.Module):
    """Conserved source/sink capacities followed by competition/allocation."""

    def __init__(self):
        super().__init__()
        self.query = nn.Linear(128, 128)
        self.key = nn.Linear(128, 128)
        self.value = nn.Linear(128, 128)
        self.output = nn.Linear(128, 128)

    def forward(self, x):
        q = _heads(self.query(x), 4).sigmoid()
        k = _heads(self.key(x), 4).sigmoid()
        v = _heads(self.value(x), 4)
        eps = 1e-6
        incoming = ((q + eps) * (k.sum(-2, keepdim=True) + eps)).sum(-1).reciprocal()
        outgoing = ((k + eps) * (q.sum(-2, keepdim=True) + eps)).sum(-1).reciprocal()
        conserved_in = ((q + eps) * ((k * outgoing[..., None]).sum(-2, keepdim=True) + eps)).sum(-1)
        conserved_out = ((k + eps) * ((q * incoming[..., None]).sum(-2, keepdim=True) + eps)).sum(-1)
        allocation = (conserved_in * (q.shape[-2] / k.shape[-2])).sigmoid()
        competition = conserved_out.clamp(-1, 1).softmax(-1) * k.shape[-2]
        kv = k.transpose(-1, -2) @ (v * competition[..., None])
        y = ((q * incoming[..., None]) @ kv) * allocation[..., None]
        return self.output(_join(y))


class CompositionalAttention(nn.Module):
    """All search/retrieval pairs, then content-conditioned retrieval choice."""

    def __init__(self):
        super().__init__()
        self.query = nn.Linear(128, 128)
        self.key = nn.Linear(128, 128)
        self.value = nn.Linear(128, 128)
        self.retrieval_query = nn.Linear(128, 128)
        self.retrieval_key = nn.Linear(32, 32)
        self.output = nn.Linear(128, 128)

    def forward(self, x):
        q, k = _heads(self.query(x), 4), _heads(self.key(x), 4)
        score = q @ k.transpose(-1, -2) / math.sqrt(32)
        # Packed legal inputs contain Local and at least one real memory head.
        if x.shape[1] < 2:
            raise ValueError('Compositional retrieval needs at least two active tokens')
        diagonal = torch.eye(x.shape[1], device=x.device, dtype=torch.bool)
        search = score.masked_fill(diagonal, -torch.inf).softmax(-1)
        values = _heads(self.value(x), 4)
        # B x token x search x retrieval x value-width; never force search= retrieval.
        candidates = torch.einsum('bsij,brjd->bisrd', search, values)
        rq = _heads(self.retrieval_query(x), 4).transpose(1, 2)
        choice = (rq[:, :, :, None] * self.retrieval_key(candidates)).sum(-1)
        choice = (choice / math.sqrt(32)).softmax(-1)
        selected = (choice[..., None] * candidates).sum(-2)
        return self.output(selected.flatten(-2))


class DualSymbolicAttention(nn.Module):
    """Parallel sensory and symbol-tagged relation-valued attention paths."""

    def __init__(self):
        super().__init__()
        self.symbol_query = nn.Linear(128, 128)
        self.symbol_templates = nn.Parameter(torch.randn(8, 128))
        self.symbol_library = nn.Parameter(torch.randn(8, 128))
        self.sensory_query = nn.Linear(128, 64, bias=False)
        self.sensory_key = nn.Linear(128, 64, bias=False)
        self.sensory_value = nn.Linear(128, 64, bias=False)
        self.sensory_output = nn.Linear(64, 64, bias=False)
        self.attention_query = nn.Linear(128, 64, bias=False)
        self.attention_key = nn.Linear(128, 64, bias=False)
        self.relation_query = nn.Linear(128, 64, bias=False)
        self.relation_key = nn.Linear(128, 64, bias=False)
        self.symbol_value = nn.Linear(128, 64, bias=False)
        self.relation_weight = nn.Parameter(torch.empty(2, 32, 4))
        nn.init.kaiming_uniform_(self.relation_weight, a=math.sqrt(5))
        self.relation_output = nn.Linear(64, 64, bias=False)

    def forward(self, x):
        symbol_q = _heads(self.symbol_query(x), 4)
        templates = self.symbol_templates.reshape(8, 4, 32).transpose(0, 1)
        library = self.symbol_library.reshape(8, 4, 32).transpose(0, 1)
        symbols = _join((symbol_q @ templates.transpose(-1, -2) / math.sqrt(32)).softmax(-1) @ library)
        sq, sk = _heads(self.sensory_query(x), 2), _heads(self.sensory_key(x), 2)
        sensory_map = (sq @ sk.transpose(-1, -2) / math.sqrt(32)).softmax(-1)
        sensory = self.sensory_output(_join(sensory_map @ _heads(self.sensory_value(x), 2)))
        aq, ak = _heads(self.attention_query(x), 2), _heads(self.attention_key(x), 2)
        selection = (aq @ ak.transpose(-1, -2) / math.sqrt(32)).softmax(-1)
        rq, rk = _heads(self.relation_query(x), 4), _heads(self.relation_key(x), 4)
        relations = (rq @ rk.transpose(-1, -2) / 4.).permute(0, 2, 3, 1)
        selected_relations = torch.einsum('bhij,bijr->bhir', selection, relations)
        related = torch.einsum('bhir,hdr->bhid', selected_relations, self.relation_weight)
        related = related + selection @ _heads(self.symbol_value(symbols), 2)
        relational = self.relation_output(_join(related))
        return torch.cat((sensory, relational), -1)


class _DynamicFactors(nn.Module):
    """One side of one stage's input-conditioned rank-two head transform."""

    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(128, 16, bias=False)
        self.factors = nn.Linear(16, 16, bias=False)
        self.diagonal = nn.Linear(128, 4, bias=False)
        nn.init.normal_(self.hidden.weight, std=math.sqrt(2 / (128 + 16)))
        nn.init.normal_(self.factors.weight, std=.02 / (math.sqrt(2 * 4 * 2) * (4 + 2)))
        nn.init.normal_(self.diagonal.weight, std=.05 * math.sqrt(2 / (4 + 128)))

    def forward(self, x):
        u, v = self.factors(F.gelu(self.hidden(x))).reshape(*x.shape[:2], 2, 2, 4).unbind(-3)
        return _rms(u, 1e-6), v, self.diagonal(x).tanh()


class _HeadComposition(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = _DynamicFactors()
        self.key = _DynamicFactors()

    def forward(self, weights, x):
        uq, vq, gq = self.query(x)
        uk, vk, gk = self.key(x)
        query_hidden = torch.einsum('bhij,birh->brij', weights, uq)
        query_mix = torch.einsum('brij,birh->bhij', query_hidden, vq)
        key_hidden = torch.einsum('bhij,bjrh->brij', weights, uk)
        key_mix = torch.einsum('brij,bjrh->bhij', key_hidden, vk)
        return (weights + query_mix + key_mix
                + weights * gq.transpose(1, 2).unsqueeze(-1)
                + weights * gk.transpose(1, 2).unsqueeze(-2))


class DynamicallyComposedAttention(nn.Module):
    """Independent pre/post-softmax, query/key off-diagonal head composition."""

    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(128, 384, bias=False)
        self.pre = _HeadComposition()
        self.post = _HeadComposition()
        self.output = nn.Linear(128, 128, bias=False)

    def forward(self, x):
        q, k, v = (_heads(tensor, 4) for tensor in self.qkv(x).chunk(3, -1))
        scores = q @ k.transpose(-1, -2) / math.sqrt(32)
        weights = self.pre(scores, x).softmax(-1)
        # The source post-compose weights may be signed: no extra softmax here.
        return self.output(_join(self.post(weights, x) @ v))


_CORES = {
    'differential_attention_v1': DifferentialAttention,
    'flowformer_conservation': FlowAttention,
    'compositional_search_retrieval': CompositionalAttention,
    'dual_attention_symbolic_relations': DualSymbolicAttention,
    'dcformer_dynamic_head_composition': DynamicallyComposedAttention,
}
METHODS = tuple(_CORES)


class AttentionInputAdapter(nn.Module):
    """Zero bridges preserve the original Flat input; LocalSkip is external.

    Evidence roles are Base, Gap-A, Gap-T, Gap-V. The caller owns availability
    semantics and excludes padding/no-history rows; ``active`` is authoritative.
    An all-inactive row is nevertheless safely returned as all zeros.
    """

    def __init__(self, method, latent_dim=256, num_heads=8, value_dim=64):
        super().__init__()
        if min(latent_dim, num_heads, value_dim) < 1:
            raise ValueError('Input dimensions must be positive')
        self.latent_dim, self.num_heads, self.value_dim = latent_dim, num_heads, value_dim
        self.tokenizer = HeadTokenizer(latent_dim, num_heads, value_dim, dim=128,
                                       shared_projection=False, normalize=True)
        self.core = _CORES[method]()
        self.local_output = nn.Linear(128, latent_dim)
        self.memory_outputs = nn.ModuleList(nn.Linear(128, value_dim) for _ in range(num_heads))
        for projection in (self.local_output, *self.memory_outputs):
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(self, local, evidence, active, availability):
        batch = local.shape[0]
        if local.shape != (batch, self.latent_dim) or availability.shape != (batch, 3):
            raise ValueError('Expected per-utterance Local and three modality availability flags')
        tokens, mask = self.tokenizer(local, evidence, active)
        transformed = torch.zeros_like(tokens)
        for rows, columns, packed in active_groups(tokens, mask):
            transformed[rows[:, None], columns[None, :]] = self.core(packed)
        transformed = safe_mask(transformed, mask)
        valid = active.bool().any(-1)
        local_new = safe_mask(local, valid) + safe_mask(self.local_output(transformed[:, 0]), valid)
        memory = transformed[:, 1:].reshape(batch, 4, self.num_heads, 128)
        correction = torch.stack([projection(memory[:, :, h])
                                  for h, projection in enumerate(self.memory_outputs)], 2)
        correction = correction.flatten(2)
        evidence_new = safe_mask(evidence, active) + safe_mask(correction, active)
        return local_new, evidence_new


def build_attention(method, latent_dim=256, num_heads=8, value_dim=64):
    if method not in _CORES:
        raise ValueError(f'Unknown attention method: {method}')
    return AttentionInputAdapter(method, latent_dim, num_heads, value_dim)
