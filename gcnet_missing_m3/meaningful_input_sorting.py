"""FSPool/FSUnpool transforms of existing evidence, not missing-feature recovery.

Independent equations follow the author pin a9f93cc774610c6d96c2c3095a1ab16f53abbefb
(Cyanogenoid/fspool), fspool.py and autoencoder/model.py. The complete transferred
chain includes BOTH latent MLPs: FSEncoder.lin after pooling and FSDecoder.lin
before unpooling, each Linear-ReLU-Linear. The saved soft sorting map is reused
by transpose, not inverted. Source reconstruction loss is not transferred.

Local conditions the set but its original Flat/Skip value is never rewritten.
Only zero-initialized corrections to existing, active memory heads are added.
"""
from __future__ import annotations

import torch
from torch import nn

from .meaningful_blocks_common import HeadTokenizer, active_groups, safe_mask


METHODS = ('fspool_fsunpool_evidence',)


class FeaturewiseSortPool(nn.Module):
    """Deterministic NeuralSort and a learned piecewise-linear rank functional."""

    def __init__(self, channels=64, pieces=20, temperature=1.):
        super().__init__()
        if channels < 1 or pieces < 1 or temperature <= 0:
            raise ValueError('Channels, pieces and temperature must be positive')
        self.pieces = pieces
        self.temperature = temperature
        self.weight = nn.Parameter(torch.randn(channels, pieces + 1))

    def rank_weights(self, count):
        if count < 1:
            raise ValueError('Sort pooling needs a nonempty packed set')
        position = torch.arange(count, dtype=self.weight.dtype, device=self.weight.device)
        position = position * (self.pieces / max(count - 1, 1))
        # Clamp protects the last knot against floating-point endpoint rounding.
        position = position.clamp(max=self.pieces)
        left = position.floor().long()
        fraction = position - left
        right = (left + 1).clamp(max=self.pieces)
        return self.weight[:, left] * (1 - fraction) + self.weight[:, right] * fraction

    def forward(self, x):
        """B x elements x channels -> (B x channels, B x channels x ranks x elements)."""
        scores = x.transpose(1, 2)
        count = scores.shape[-1]
        if count < 1 or scores.shape[1] != self.weight.shape[0]:
            raise ValueError('Expected nonempty packed tokens with matching channels')
        distances = (scores[..., :, None] - scores[..., None, :]).abs().sum(-1)
        rank = torch.arange(count, device=x.device, dtype=x.dtype)
        slope = count - 1 - 2 * rank
        logits = slope[:, None] * scores[..., None, :] - distances[..., None, :]
        permutation = (logits / self.temperature).softmax(-1)
        sorted_values = torch.einsum('bcij,bcj->bci', permutation, scores)
        pooled = (sorted_values * self.rank_weights(count)).sum(-1)
        return pooled, permutation

    def forward_transpose(self, latent, permutation):
        """Separate decoder weights expand latent ranks, then apply saved P^T."""
        count = permutation.shape[-1]
        ranked = latent[..., None] * self.rank_weights(count)
        return torch.einsum('bcij,bci->bjc', permutation, ranked)


def _pointwise_mlp():
    # A tokenwise Linear is exactly the 1x1 Conv1d operation in the source.
    return nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64))


class SortPoolUnpoolCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _pointwise_mlp()
        self.pool = FeaturewiseSortPool()
        self.encoder_latent = _pointwise_mlp()
        self.decoder_latent = _pointwise_mlp()
        self.unpool = FeaturewiseSortPool()
        self.decoder = _pointwise_mlp()
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, tokens):
        pooled, permutation = self.pool(self.encoder(tokens))
        latent = self.decoder_latent(self.encoder_latent(pooled))
        restored = self.unpool.forward_transpose(latent, permutation)
        return self.decoder(restored)


class SortingInputAdapter(nn.Module):
    """Pack real heads, preserve their identities through P^T, and add zero bridges.

    ``active`` is authoritative; the caller supplies already ablation-masked reads
    and bypasses no-history/padding rows. Availability is accepted without deriving
    any additional memory reads or imputations from it.
    """

    def __init__(self, latent_dim=256, num_heads=8, value_dim=64):
        super().__init__()
        if min(latent_dim, num_heads, value_dim) < 1:
            raise ValueError('Input dimensions must be positive')
        self.latent_dim, self.num_heads, self.value_dim = latent_dim, num_heads, value_dim
        self.tokenizer = HeadTokenizer(latent_dim, num_heads, value_dim, dim=64,
                                       shared_projection=False, normalize=False)
        # The approved 64-dimensional heads are already in the core input space.
        # Only nonstandard value widths need a tokenwise dimensional projection.
        self.tokenizer.memory = nn.ModuleList(
            nn.Identity() if value_dim == 64 else nn.Linear(value_dim, 64)
            for _ in range(num_heads)
        )
        self.core = SortPoolUnpoolCore()
        self.memory_outputs = nn.ModuleList(nn.Linear(64, value_dim) for _ in range(num_heads))
        for projection in self.memory_outputs:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(self, local, evidence, active, availability):
        batch = local.shape[0]
        if local.shape != (batch, self.latent_dim) or availability.shape != (batch, 3):
            raise ValueError('Expected per-utterance Local and three availability flags')
        tokens, mask = self.tokenizer(local, evidence, active)
        transformed = torch.zeros_like(tokens)
        for rows, columns, packed in active_groups(tokens, mask):
            transformed[rows[:, None], columns[None, :]] = self.core(packed)
        memory = safe_mask(transformed, mask)[:, 1:].reshape(batch, 4, self.num_heads, 64)
        correction = torch.stack([projection(memory[:, :, head])
                                  for head, projection in enumerate(self.memory_outputs)], 2)
        evidence_new = safe_mask(evidence, active) + safe_mask(correction.flatten(2), active)
        return local, evidence_new


def build_sorting(method, latent_dim=256, num_heads=8, value_dim=64):
    if method not in METHODS:
        raise ValueError(f'Unknown sorting method: {method}')
    # New initialization must not advance the legacy model/dropout RNG stream.
    with torch.random.fork_rng(devices=[]):
        return SortingInputAdapter(latent_dim, num_heads, value_dim)
