"""Eight contiguous Local groups on the original rooted Nested GNN.

Only the graph's Local tokenization and matching residual decoders change;
memory tokens still contain the actual cfg84 forward value heads.
"""
import torch
from torch import nn

from .meaningful_blocks_common import active_groups, safe_mask
from .meaningful_input_new40 import zero_linear
from .nested_sweep import NestedSweepGNN


class Local8Tokenizer(nn.Module):
    """Local0..7, Base0..7, GapA0..7, GapT0..7, GapV0..7."""

    def __init__(self):
        super().__init__()
        # Consume the original Local projection's initialization draws without
        # registering an unused layer. New group projections use isolated draws
        # so the existing memory projections and embeddings retain seed parity.
        nn.Linear(256, 64)
        with torch.random.fork_rng(devices=[]):
            self.local = nn.ModuleList([nn.Linear(32, 64) for _ in range(8)])
        # A head projection is shared across the four memory roles.
        self.memory = nn.ModuleList([nn.Linear(64, 64) for _ in range(8)])
        self.role_embedding = nn.Embedding(5, 64)
        self.head_embedding = nn.Embedding(8, 64)
        self.norm = nn.LayerNorm(64)
        self.register_buffer('role_ids', torch.arange(5).repeat_interleave(8))
        self.register_buffer('head_ids', torch.arange(8).repeat(5))

    def forward(self, local, evidence, active):
        batch = local.shape[0]
        if (local.shape != (batch, 256) or evidence.shape != (batch, 4, 512)
                or active.shape != (batch, 4)):
            raise ValueError('Local8 requires Local256 and four actual 8x64 evidence slots')
        active = active.bool()
        valid = active.any(-1)
        groups = safe_mask(local, valid).reshape(batch, 8, 32)
        local_tokens = torch.stack([layer(groups[:, head])
                                    for head, layer in enumerate(self.local)], 1)
        values = safe_mask(evidence, active).reshape(batch, 4, 8, 64)
        memory_tokens = torch.stack([layer(values[:, :, head])
                                     for head, layer in enumerate(self.memory)], 2)
        tokens = torch.cat((local_tokens, memory_tokens.flatten(1, 2)), 1)
        tokens = self.norm(tokens + self.role_embedding(self.role_ids)[None]
                           + self.head_embedding(self.head_ids)[None])
        mask = torch.cat((valid[:, None].expand(-1, 8),
                          active[..., None].expand(-1, -1, 8).flatten(1)), 1)
        return safe_mask(tokens, mask), mask


class Local8NestedGNN(NestedSweepGNN):
    """Reuse the old 64d/depth3 rooted mean-pool algorithm verbatim."""

    def adjacency(self, columns, heads, dtype):
        role, head = columns // heads, columns % heads
        adjacency = ((role[:, None] == role[None, :]) |
                     (head[:, None] == head[None, :]) |
                     (role[:, None] == 0) | (role[None, :] == 0))
        adjacency.fill_diagonal_(False)
        return adjacency.to(dtype)


class Local8Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads, self.value_dim = 8, 64
        self.core = Local8NestedGNN()
        self.tokenizer = Local8Tokenizer()
        self.local_decoders = nn.ModuleList([zero_linear(64, 32) for _ in range(8)])
        self.memory_decoders = nn.ModuleList([zero_linear(64, 64) for _ in range(8)])

    @property
    def local_projections(self):
        return self.tokenizer.local

    def forward(self, local, evidence, active, availability):
        tokens, mask = self.tokenizer(local, evidence, active)
        result = torch.zeros_like(tokens)
        for rows, columns, packed in active_groups(tokens, mask):
            value = self.core(packed, columns, self.num_heads)
            if value.shape != packed.shape:
                raise RuntimeError('Local8 Nested changed the packed token interface')
            result[rows[:, None], columns[None, :]] = value.to(result.dtype)
        delta_local = torch.cat([decoder(result[:, head])
                                 for head, decoder in enumerate(self.local_decoders)], -1)
        delta_local = safe_mask(delta_local, active.any(-1))
        memory = result[:, 8:].reshape(-1, 4, 8, 64)
        delta_memory = torch.stack([decoder(memory[:, :, head])
                                   for head, decoder in enumerate(self.memory_decoders)], 2).flatten(2)
        return local + delta_local.to(local.dtype), safe_mask(
            evidence + delta_memory.to(evidence.dtype), active)


def build(method, latent_dim=256, num_heads=8, value_dim=64):
    if method != 'nested_local8_evidence':
        raise ValueError('Unknown Local8 Nested method: ' + method)
    if (latent_dim, num_heads, value_dim) != (256, 8, 64):
        raise ValueError('Local8 requires cfg84 Local256 and actual 8x64 memory heads')
    return Local8Adapter()
