"""Typed real-head evidence helpers; no OSRAM reads, labels or temporal state."""
import torch
from torch import nn


def safe_mask(tensor, mask):
    while mask.ndim < tensor.ndim:
        mask = mask.unsqueeze(-1)
    return torch.where(mask.bool(), tensor, torch.zeros_like(tensor))


class HeadTokenizer(nn.Module):
    """Local followed by four fixed evidence roles, each with actual value heads."""

    def __init__(self, latent_dim, num_heads, value_dim, dim=128,
                 shared_projection=False, normalize=False):
        super().__init__()
        self.num_heads, self.value_dim = num_heads, value_dim
        self.local = nn.Linear(latent_dim, dim)
        self.shared_projection = shared_projection
        self.memory = nn.ModuleList([
            nn.Linear(value_dim, dim) for _ in range(1 if shared_projection else num_heads)
        ])
        self.role_embedding = nn.Embedding(5, dim)
        self.head_embedding = nn.Embedding(num_heads, dim)
        self.norm = nn.LayerNorm(dim) if normalize else nn.Identity()
        self.register_buffer('role_ids', torch.tensor([0] + sum(([i]*num_heads for i in range(1, 5)), [])))
        self.register_buffer('head_ids', torch.tensor([-1] + list(range(num_heads))*4))

    def forward(self, local, evidence, active):
        batch = local.shape[0]
        active = active.bool()
        if evidence.shape != (batch, 4, self.num_heads*self.value_dim) or active.shape != (batch, 4):
            raise ValueError('Expected four fixed evidence slots of actual flattened heads')
        valid = active.any(-1)
        local_token = self.local(safe_mask(local, valid))[:, None]
        values = safe_mask(evidence, active).reshape(batch, 4, self.num_heads, self.value_dim)
        if self.shared_projection:
            memory_tokens = self.memory[0](values)
        else:
            memory_tokens = torch.stack([layer(values[:, :, h]) for h, layer in enumerate(self.memory)], 2)
        memory_tokens = memory_tokens + self.head_embedding.weight[None, None]
        tokens = torch.cat((local_token, memory_tokens.flatten(1, 2)), 1)
        tokens = self.norm(tokens + self.role_embedding(self.role_ids)[None])
        mask = torch.cat((valid[:, None], active[..., None].expand(-1, -1, self.num_heads).flatten(1)), 1)
        return safe_mask(tokens, mask), mask


def typed_means(tokens, mask, num_heads):
    safe = safe_mask(tokens, mask)
    memory = safe[:, 1:].reshape(tokens.shape[0], 4, num_heads, tokens.shape[-1])
    counts = mask[:, 1:].reshape(tokens.shape[0], 4, num_heads).sum(-1).clamp_min(1)
    return torch.cat((safe[:, :1], memory.sum(2)/counts[..., None]), 1)


def active_groups(tokens, mask):
    """Pack equal-availability rows without changing canonical role/head order."""
    patterns, inverse = torch.unique(mask.bool(), dim=0, return_inverse=True)
    for index, pattern in enumerate(patterns):
        rows = (inverse == index).nonzero(as_tuple=True)[0]
        columns = pattern.nonzero(as_tuple=True)[0]
        if columns.numel():
            yield rows, columns, tokens[rows][:, columns]
