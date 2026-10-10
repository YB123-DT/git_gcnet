"""Direct evidence decoding through existing Perceiver latent processors.

No external raw-evidence residual. Only current Local/OSRAM reads are tokens;
latent slots are parameters, not persistent history or cached observations.
"""
import torch
from torch import nn
from .meaningful_blocks_common import active_groups, safe_mask
from .meaningful_blocks_set_context import PerceiverIOReadout


class DirectPerceiverInput(nn.Module):
    def __init__(self, latent_dim=256, num_heads=8, value_dim=64):
        super().__init__()
        self.num_heads = num_heads
        self.core = PerceiverIOReadout(latent_dim, num_heads, value_dim)
        self.local_decoder = nn.Linear(128, latent_dim)
        self.memory_decoders = nn.ModuleList([nn.Linear(128, value_dim) for _ in range(num_heads)])

    def forward(self, local, evidence, active, availability):
        tokens, mask = self.core.tokenizer(local, evidence, active)
        result = torch.zeros_like(tokens)
        for rows, columns, packed in active_groups(tokens, mask):
            latents = self.core.encode(self.core.latents[None].expand(len(rows), -1, -1), packed)
            for layer in self.core.process:
                latents = layer(latents)
            # Local retains the original Local-derived output query. Memory
            # queries keep their real role/head identity from the same tokenizer.
            query_local = (self.core.query(local[rows]) + self.core.query_bias)[:, None]
            queries = torch.cat((query_local, packed[:, 1:]), 1)
            decoded = self.core.decode(queries, latents)
            result[rows[:, None], columns[None, :]] = decoded.to(result.dtype)
        values = result[:, 1:].reshape(-1, 4, self.num_heads, 128)
        memory = torch.stack([decoder(values[:, :, h]) for h,decoder in enumerate(self.memory_decoders)],2).flatten(2)
        return (safe_mask(self.local_decoder(result[:,0]), active.any(-1)).to(local.dtype),
                safe_mask(memory, active).to(evidence.dtype))
