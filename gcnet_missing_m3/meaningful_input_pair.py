"""PPGN pair-composition adapter of already-read, current-utterance evidence.

Independent mathematical implementation of the complete three-block core in
Maron et al., Provably Powerful Graph Networks (NeurIPS 2019):
https://arxiv.org/abs/1905.11136, sections 6-7.
Author reference: hadarser/ProvablyPowerfulGraphNetworks_torch,
4576eff7dc9137c70bdcef724392b82a1642c562, layers/modules.py (Apache-2.0).

The pair axes enumerate evidence identities, not image pixels or time. The
endpoint lift and per-token diagonal/global readback adapt the source graph
classifier; the source pair transforms, matrix product and concatenation
reducer are retained. No claim of a 3-WL guarantee for this finite adapter.
"""

import torch
from torch import nn

from .meaningful_blocks_common import HeadTokenizer, active_groups, safe_mask


METHODS = ("ppgn_pair_composition",)


def _pair_compose(left, right):
    """P[b,i,j,c] = sum_k left[b,i,k,c] * right[b,k,j,c]."""
    # Channels are independent batches of square vertex-pair matrices. In
    # particular, this is NOT an endpoint feature/channel dot product.
    product = left.permute(0, 3, 1, 2) @ right.permute(0, 3, 1, 2)
    return product.permute(0, 2, 3, 1)


def _initialize_affine(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class _PairBlock(nn.Module):
    """Two nonlinear pair maps, channelwise composition, concat/reduction."""

    def __init__(self, channels=64):
        super().__init__()
        self.left = nn.Sequential(
            nn.Linear(channels, channels), nn.ReLU(),
            nn.Linear(channels, channels), nn.ReLU())
        self.right = nn.Sequential(
            nn.Linear(channels, channels), nn.ReLU(),
            nn.Linear(channels, channels), nn.ReLU())
        self.reduce = nn.Linear(2 * channels, channels)
        self.apply(_initialize_affine)

    def forward(self, pairs):
        composed = _pair_compose(self.left(pairs), self.right(pairs))
        # The source skip is concatenation plus an affine map, not addition;
        # there is deliberately no extra activation or size normalization.
        return self.reduce(torch.cat((pairs, composed), dim=-1))


def _pair_readback(pairs):
    """Per-token diagonal plus within-sample diagonal/off-diagonal maxima."""
    count = pairs.shape[1]
    indices = torch.arange(count, device=pairs.device)
    diagonal = pairs[:, indices, indices]
    diagonal_max = diagonal.amax(dim=1, keepdim=True).expand(-1, count, -1)
    if count > 1:
        is_diagonal = torch.eye(count, device=pairs.device, dtype=torch.bool)
        off_diagonal = pairs.masked_fill(is_diagonal[None, :, :, None], -torch.inf)
        off_diagonal_max = off_diagonal.amax(dim=(1, 2))[:, None].expand(-1, count, -1)
    else:
        off_diagonal_max = torch.zeros_like(diagonal)
    return torch.cat((diagonal, diagonal_max, off_diagonal_max), dim=-1)


class PairInputAdapter(nn.Module):
    """Return raw Local/evidence inputs for the unchanged Flat adapter."""

    def __init__(self, latent_dim=256, num_heads=8, value_dim=64):
        super().__init__()
        self.num_heads = num_heads
        self.value_dim = value_dim
        self.tokenizer = HeadTokenizer(latent_dim, num_heads, value_dim, dim=128,
                                       shared_projection=False, normalize=True)
        self.pair_lift = nn.Linear(257, 64)
        _initialize_affine(self.pair_lift)
        self.blocks = nn.ModuleList([_PairBlock(64) for _ in range(3)])
        self.local_decoder = nn.Linear(576, latent_dim)
        self.memory_decoders = nn.ModuleList([
            nn.Linear(576, value_dim) for _ in range(num_heads)
        ])
        for decoder in [self.local_decoder, *self.memory_decoders]:
            nn.init.zeros_(decoder.weight)
            nn.init.zeros_(decoder.bias)

    def _encode_pairs(self, tokens):
        batch, count, dim = tokens.shape
        left = tokens[:, :, None].expand(-1, -1, count, -1)
        right = tokens[:, None, :].expand(-1, count, -1, -1)
        equal = torch.eye(count, device=tokens.device, dtype=tokens.dtype)
        equal = equal[None, :, :, None].expand(batch, -1, -1, -1)
        pairs = self.pair_lift(torch.cat((left, right, equal), dim=-1))
        readbacks = []
        for block in self.blocks:
            pairs = block(pairs)
            readbacks.append(_pair_readback(pairs))
        return torch.cat(readbacks, dim=-1)

    def forward(self, local, evidence, active, availability):
        active = active.bool()
        compute_dtype = (torch.float64 if self.tokenizer.local.weight.dtype == torch.float64
                         else torch.float32)
        # Pair products intentionally remain unnormalized; keep them FP32
        # under autocast without downcasting a deliberately double model.
        with torch.autocast(device_type=local.device.type, enabled=False):
            tokens, mask = self.tokenizer(local.to(compute_dtype),
                                          evidence.to(compute_dtype), active)
            contexts = tokens.new_zeros(tokens.shape[0], tokens.shape[1], 576)
            for rows, columns, packed in active_groups(tokens, mask):
                contexts[rows[:, None], columns[None, :]] = self._encode_pairs(packed)
            local_delta = safe_mask(self.local_decoder(contexts[:, 0]), active.any(-1))
            memory = contexts[:, 1:].reshape(local.shape[0], 4, self.num_heads, 576)
            memory_delta = torch.stack([
                decoder(memory[:, :, head])
                for head, decoder in enumerate(self.memory_decoders)
            ], dim=2).flatten(2)
            memory_delta = safe_mask(memory_delta, active)
        return (local + local_delta.to(local.dtype),
                safe_mask(evidence, active) + memory_delta.to(evidence.dtype))


def build_pair(method, latent_dim=256, num_heads=8, value_dim=64):
    if method not in METHODS:
        raise ValueError(f"Unknown pair representation method: {method}")
    return PairInputAdapter(latent_dim, num_heads, value_dim)
