"""Direct pre-decoder Nested graph features for the original Flat task path.

No reconstructed evidence, extra memory query, task loss or residual decoder.
The original Nested tokenizer, topology, rooted GIN and graph readout are reused.
"""
import torch
from torch import nn

from .meaningful_blocks_common import HeadTokenizer, active_groups, safe_mask, typed_means
from .meaningful_new40_structure import NestedGNN


class NestedRootedConcat(nn.Module):
    output_dim = 320

    def __init__(self, latent_dim=256, num_heads=8, value_dim=64):
        super().__init__()
        self.num_heads = num_heads
        self.forward_dim = num_heads * value_dim
        # Original TokenAdapter constructs its core before its tokenizer.
        self.core = NestedGNN()
        self.tokenizer = HeadTokenizer(latent_dim, num_heads, value_dim, dim=64,
                                       shared_projection=False, normalize=True)
        self.last_diagnostics = {}

    def forward(self, local, base, gap, availability, umask):
        valid = umask.T.bool()
        active = torch.cat((valid[..., None], valid[..., None] & ~availability.bool()), -1)
        evidence = torch.cat((base[..., :self.forward_dim].unsqueeze(2),
                              gap[..., :self.forward_dim]), 2)
        evidence = safe_mask(evidence, active)
        # Match original MeaningfulInputAdapter: no graph on first valid turn
        # or on an entirely zero historical readout.
        history = valid & (valid.long().cumsum(0) > 1) & evidence.ne(0).any(-1).any(-1)
        output = local.new_zeros(*local.shape[:2], self.output_dim)
        if history.any():
            tokens, mask = self.tokenizer(local[history], evidence[history], active[history])
            graph = torch.zeros_like(tokens)
            for rows, columns, packed in active_groups(tokens, mask):
                graph[rows[:, None], columns[None, :]] = self.core(
                    packed, columns, self.num_heads).to(graph.dtype)
            pooled = typed_means(graph, mask, self.num_heads)
            role_mask = torch.cat((active[history].any(-1, keepdim=True), active[history]), -1)
            output[history] = safe_mask(pooled, role_mask).flatten(1)
        output = safe_mask(output, valid)
        self.last_diagnostics = {'valid_count': int(valid.sum()), 'history_count': int(history.sum()),
                                 'concat_dim': self.output_dim}
        return output


def capacity_control_width(latent_dim=256, context_dim=1024, output_dim=1600):
    """Nearest Flat width matching graph+expanded adapter total parameter count."""
    with torch.random.fork_rng(devices=[]):
        graph_params = sum(p.numel() for p in NestedRootedConcat(latent_dim).parameters())
    raw_dim = latent_dim + 4 * context_dim
    extra = 320 * output_dim + 2 * 320 + graph_params
    return output_dim + int(extra / (raw_dim + output_dim + 1) + 0.5)


class ConcatCapacityControl(nn.Module):
    """Diagnostic metadata only; no graph and no added trainable parameters."""
    def __init__(self, width):
        super().__init__()
        self.last_diagnostics = {'capacity_control_width': width}
