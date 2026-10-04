"""Structured, identity-initialized adapters of already-read OSRAM evidence.

Independent mathematical implementations, not copied upstream source:
* TokenLearner V1.1 + Transformer + TokenFuser: arxiv.org/abs/2106.11297;
  google-research/scenic 8c113c501c9f700b69899c55a69e65bb46727da6.
* MBT synchronous bottleneck exchange: arxiv.org/abs/2107.00135; same
  Scenic pin. Both Scenic reference implementations are Apache-2.0.
* Complete NetVLAD residual encoding: arxiv.org/abs/1511.07247;
  Relja/netvlad 652dbe71aa45c691961ddd9f6cf902574e6bdc2f (MIT).
* ToMe key-space matching, size-weighted coarsening and proportional
  attention: arxiv.org/abs/2210.09461; facebookresearch/ToMe
  af95e4b1befa172dadccd8c81e223b10090f9579 (CC-BY-NC-4.0).

The adapters operate within one utterance, never read/write memory, and
introduce no loss, temporal geometry, running state, or stochastic forward.
NetVLAD redistribution and ToMe membership restoration are readout adapters,
not claims of invertible reconstruction. GroupViT is deliberately excluded.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .meaningful_blocks_common import HeadTokenizer, active_groups, safe_mask


METHODS = (
    "tokenlearner_fuser_v11",
    "mbt_role_bottleneck",
    "netvlad_residual_encoding",
    "tome_merge_reconstruct",
)


class _Attention(nn.Module):
    """Ordinary full attention, optionally corrected for merged-token mass."""

    def __init__(self, dim=128, heads=4):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.output = nn.Linear(dim, dim)

    def forward(self, x, sizes=None):
        batch, count, dim = x.shape
        projected = self.qkv(x).reshape(batch, count, 3, self.heads, self.head_dim)
        query, key, value = projected.permute(2, 0, 3, 1, 4).unbind(0)
        logits = (query @ key.transpose(-1, -2)) * self.head_dim ** -0.5
        if sizes is not None:
            logits = logits + sizes.log()[:, None, None, :, 0]
        context = logits.softmax(-1) @ value
        context = context.transpose(1, 2).reshape(batch, count, dim)
        return self.output(context), key.mean(1)


class _TransformerBlock(nn.Module):
    def __init__(self, dim=128, hidden=256):
        super().__init__()
        self.norm_attention = nn.LayerNorm(dim, eps=1e-6)
        self.attention = _Attention(dim)
        self.norm_ffn = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        message, _ = self.attention(self.norm_attention(x))
        x = x + message
        return x + self.ffn(self.norm_ffn(x))


class _TokenLearnerFuser(nn.Module):
    """Separate learned analysis/synthesis maps around latent interaction."""

    def __init__(self, dim=128, compressed=4):
        super().__init__()
        self.analysis_norm = nn.LayerNorm(dim, eps=1e-6)
        self.analysis = nn.Sequential(nn.Linear(dim, 64), nn.GELU(), nn.Linear(64, compressed))
        self.latent_block = _TransformerBlock(dim)
        self.mix_norm_in = nn.LayerNorm(dim, eps=1e-6)
        self.token_mix = nn.Linear(compressed, compressed)
        self.mix_norm_out = nn.LayerNorm(dim, eps=1e-6)
        self.synthesis_norm = nn.LayerNorm(dim, eps=1e-6)
        self.synthesis = nn.Sequential(nn.Linear(dim, 64), nn.GELU(), nn.Linear(64, compressed))
        # Source zero-initializes this map. The raw decoder is already zero:
        # two consecutive zero maps would prevent either weight from learning.
        nn.init.xavier_uniform_(self.token_mix.weight)
        nn.init.zeros_(self.token_mix.bias)

    def forward(self, x, columns, num_heads):
        analysis = self.analysis(self.analysis_norm(x)).transpose(1, 2).softmax(-1)
        compressed = self.latent_block(analysis @ x)
        mixed = self.token_mix(self.mix_norm_in(compressed).transpose(1, 2)).transpose(1, 2)
        mixed = self.mix_norm_out(mixed)
        synthesis = self.synthesis(self.synthesis_norm(x)).sigmoid()
        # This is the Fuser correction; the wrapper supplies the original
        # raw-slot residual instead of replacing it by projected tokens.
        return synthesis @ mixed


class _MultimodalBottleneck(nn.Module):
    """Independent role streams with two synchronous consensus exchanges."""

    def __init__(self, dim=128, bottlenecks=4):
        super().__init__()
        self.initial_bottleneck = nn.Parameter(torch.empty(1, bottlenecks, dim))
        nn.init.normal_(self.initial_bottleneck, std=0.02)
        self.private = nn.ModuleList([_TransformerBlock(dim) for _ in range(5)])
        self.exchange = nn.ModuleList([
            nn.ModuleList([_TransformerBlock(dim) for _ in range(5)]) for _ in range(2)
        ])
        self.final_norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x, columns, num_heads):
        roles = torch.where(columns == 0, 0, (columns - 1) // num_heads + 1)
        indices = {role: (roles == role).nonzero(as_tuple=True)[0] for role in range(5)}
        indices = {role: index for role, index in indices.items() if index.numel()}
        streams = {role: self.private[role](x[:, index]) for role, index in indices.items()}
        bottleneck = self.initial_bottleneck.expand(x.shape[0], -1, -1)
        for encoders in self.exchange:
            proposals, updated = [], {}
            for role, stream in streams.items():
                length = stream.shape[1]
                combined = encoders[role](torch.cat((stream, bottleneck), dim=1))
                updated[role] = combined[:, :length]
                proposals.append(combined[:, length:])
            # Every encoder above consumed the same old bottleneck. Absent
            # roles contribute neither a proposal nor a denominator term.
            bottleneck = torch.stack(proposals, dim=0).mean(0)
            streams = updated
        output = torch.zeros_like(x)
        for role, index in indices.items():
            output[:, index] = streams[role].to(output.dtype)
        return self.final_norm(output)


class _NetVLAD(nn.Module):
    """Complete center-relative residual encoder, then typed redistribution."""

    def __init__(self, dim=128, codewords=8):
        super().__init__()
        self.assignment = nn.Linear(dim, codewords)
        self.centers = nn.Parameter(torch.empty(codewords, dim))
        nn.init.normal_(self.centers, std=0.02)

    def forward(self, x, columns, num_heads):
        x = F.normalize(x, p=2, dim=-1, eps=1e-6)
        assignment = self.assignment(x).softmax(-1)
        first_moments = assignment.transpose(1, 2) @ x
        mass = assignment.sum(1).unsqueeze(-1)
        residuals = first_moments - mass * self.centers[None]
        intra_normalized = F.normalize(residuals, p=2, dim=-1, eps=1e-6)
        descriptor = F.normalize(intra_normalized.flatten(1), p=2, dim=-1, eps=1e-6)
        code_features = descriptor.reshape_as(residuals)
        # The paper's full normalized descriptor is retained. Mapping its
        # blocks to existing input slots is our explicit readout adaptation.
        return assignment @ code_features


def _merge_evidence(x, metric, sizes, max_merges=4):
    """Coarsen a bipartite token graph; return original-to-coarse indices.

    Local is the protected token at index zero. Alternating indices define
    a graph partition only, not distances, chronology, or a pixel layout.
    Hard topology selection is detached; weighted values remain attached.
    """
    batch, count, dim = x.shape
    merges = min(max_merges, (count - 1) // 2)
    if merges == 0:
        identity = torch.arange(count, device=x.device).expand(batch, -1)
        return x, sizes, identity

    with torch.no_grad():
        normalized = F.normalize(metric, p=2, dim=-1, eps=1e-6)
        similarities = normalized[:, ::2] @ normalized[:, 1::2].transpose(1, 2)
        similarities[:, 0] = -torch.inf
        best_score, destinations = similarities.max(-1)
        ranking = best_score.argsort(dim=-1, descending=True, stable=True)
        merged_sources = ranking[:, :merges]
        kept_sources = ranking[:, merges:].sort(-1).values
        chosen_destinations = destinations.gather(1, merged_sources)
        kept_count = kept_sources.shape[1]
        destination_count = count // 2
        inverse = torch.empty(batch, count, device=x.device, dtype=torch.long)
        inverse[:, 1::2] = kept_count + torch.arange(destination_count, device=x.device)
        inverse.scatter_(1, 2 * kept_sources,
                         torch.arange(kept_count, device=x.device).expand(batch, -1))
        inverse.scatter_(1, 2 * merged_sources, kept_count + chosen_destinations)

    def group_sum(values):
        width = values.shape[-1]
        source, destination = values[:, ::2], values[:, 1::2]
        kept = source.gather(1, kept_sources[..., None].expand(-1, -1, width))
        selected = source.gather(1, merged_sources[..., None].expand(-1, -1, width))
        summed = destination.scatter_add(
            1, chosen_destinations[..., None].expand(-1, -1, width), selected)
        return torch.cat((kept, summed), dim=1)

    total_mass = group_sum(sizes)
    return group_sum(x * sizes) / total_mass, total_mass, inverse


class _TokenMerging(nn.Module):
    """Two full ToMe blocks and reverse within-forward membership mapping."""

    def __init__(self, dim=128):
        super().__init__()
        self.blocks = nn.ModuleList([_TransformerBlock(dim) for _ in range(2)])

    def forward(self, x, columns, num_heads):
        sizes = torch.ones_like(x[..., :1])
        inverse_maps = []
        for block in self.blocks:
            message, metric = block.attention(block.norm_attention(x), sizes)
            x = x + message
            x, sizes, inverse = _merge_evidence(x, metric, sizes)
            inverse_maps.append(inverse)
            x = x + block.ffn(block.norm_ffn(x))
        # Restoration broadcasts a group's result to its members; information
        # destroyed by merging is not recovered. Raw evidence has its own skip.
        for inverse in reversed(inverse_maps):
            x = x.gather(1, inverse[..., None].expand(-1, -1, x.shape[-1]))
        return x


class RepresentationInputAdapter(nn.Module):
    """Return corrected raw inputs; the existing Flat skip/head stay outside."""

    def __init__(self, method, latent_dim=256, num_heads=8, value_dim=64):
        super().__init__()
        cores = {
            "tokenlearner_fuser_v11": _TokenLearnerFuser,
            "mbt_role_bottleneck": _MultimodalBottleneck,
            "netvlad_residual_encoding": _NetVLAD,
            "tome_merge_reconstruct": _TokenMerging,
        }
        if method not in cores:
            raise ValueError(f"Unknown representation method: {method}")
        self.method = method
        self.num_heads = num_heads
        self.value_dim = value_dim
        self.tokenizer = HeadTokenizer(latent_dim, num_heads, value_dim, dim=128,
                                       shared_projection=False, normalize=True)
        self.core = cores[method]()
        self.local_decoder = nn.Linear(128, latent_dim)
        self.memory_decoders = nn.ModuleList([nn.Linear(128, value_dim) for _ in range(num_heads)])
        for decoder in [self.local_decoder, *self.memory_decoders]:
            nn.init.zeros_(decoder.weight)
            nn.init.zeros_(decoder.bias)

    def forward(self, local, evidence, active, availability):
        active = active.bool()
        tokens, token_mask = self.tokenizer(local, evidence, active)
        contextual = torch.zeros_like(tokens)
        for rows, columns, packed in active_groups(tokens, token_mask):
            updated = self.core(packed, columns, self.num_heads)
            contextual[rows[:, None], columns[None, :]] = updated.to(contextual.dtype)
        local_delta = safe_mask(self.local_decoder(contextual[:, 0]), active.any(-1))
        memory_tokens = contextual[:, 1:].reshape(local.shape[0], 4, self.num_heads, 128)
        memory_delta = torch.stack([
            decoder(memory_tokens[:, :, head])
            for head, decoder in enumerate(self.memory_decoders)
        ], dim=2).flatten(2)
        memory_delta = safe_mask(memory_delta, active)
        return (local + local_delta.to(local.dtype),
                safe_mask(evidence, active) + memory_delta.to(evidence.dtype))


def build_representation(method, latent_dim=256, num_heads=8, value_dim=64):
    return RepresentationInputAdapter(method, latent_dim, num_heads, value_dim)
