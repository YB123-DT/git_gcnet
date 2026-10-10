"""Shared interface, not shared algorithms, for the additional forty designs."""
import importlib

import torch
from torch import nn

from .meaningful_blocks_common import HeadTokenizer, active_groups, safe_mask
from .meaningful_new40_registry import NEW40_FAMILIES, NEW40_VARIANTS, NESTED_SWEEP


def zero_linear(in_features, out_features):
    layer = nn.Linear(in_features, out_features)
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class TokenAdapter(nn.Module):
    """Pack real active heads, call one distinct core, decode raw-slot deltas."""
    def __init__(self, core, latent_dim=256, num_heads=8, value_dim=64, dim=64, residual=True,
                 zero_decoder=True):
        super().__init__()
        self.residual = residual
        self.num_heads, self.value_dim = num_heads, value_dim
        self.tokenizer = HeadTokenizer(latent_dim, num_heads, value_dim, dim=dim,
                                       shared_projection=False, normalize=True)
        self.core = core
        decoder = zero_linear if zero_decoder else nn.Linear
        self.local_decoder = decoder(dim, latent_dim)
        self.memory_decoders = nn.ModuleList([decoder(dim, value_dim)
                                             for _ in range(num_heads)])

    def forward(self, local, evidence, active, availability):
        tokens, mask = self.tokenizer(local, evidence, active)
        result = torch.zeros_like(tokens)
        for rows, columns, packed in active_groups(tokens, mask):
            value = self.core(packed, columns, self.num_heads)
            if value.shape != packed.shape:
                raise RuntimeError('A token mechanism changed the packed interface')
            result[rows[:, None], columns[None, :]] = value.to(result.dtype)
        delta_local = safe_mask(self.local_decoder(result[:, 0]), active.any(-1))
        memory = result[:, 1:].reshape(-1, 4, self.num_heads, result.shape[-1])
        delta_memory = torch.stack([decoder(memory[:, :, h])
                                   for h, decoder in enumerate(self.memory_decoders)], 2).flatten(2)
        gap_gate = getattr(self, 'gap_residual_gate', None)
        if gap_gate is not None:
            gated_gap = gap_gate(local, evidence[:, 1:], delta_memory[:, 1:],
                                 active[:, 1:], availability)
            delta_memory = torch.cat((delta_memory[:, :1], gated_gap), 1)
        if not self.residual:
            return delta_local.to(local.dtype), safe_mask(delta_memory.to(evidence.dtype), active)
        return local + delta_local.to(local.dtype), safe_mask(
            evidence + delta_memory.to(evidence.dtype), active)


def build_new40(method, latent_dim=256, num_heads=8, value_dim=64):
    if method == 'nested_gnn_gap_residual_gate':
        from .meaningful_new40_structure import NestedGNN
        from .nested_gap_gate import NestedGapResidualGate
        adapter = TokenAdapter(NestedGNN(), latent_dim, num_heads, value_dim, dim=64)
        # All original parameters and downstream random draws remain unchanged.
        with torch.random.fork_rng(devices=[]):
            adapter.gap_residual_gate = NestedGapResidualGate(latent_dim, num_heads * value_dim)
        return adapter
    if method in ('neural_production_local', 'neural_production_local_w256'):
        from .meaningful_new40_conditional import NeuralProduction, TokenReadout
        width = 256 if method.endswith('_w256') else 96
        return TokenReadout(NeuralProduction(rule_hidden=width), latent_dim, num_heads, value_dim,
                            local_correction=True)
    if method == 'nested_local8_evidence':
        module = importlib.import_module('.nested_local8', __package__)
        return module.build(method, latent_dim, num_heads, value_dim)
    if method in NESTED_SWEEP:
        module = importlib.import_module('.nested_sweep', __package__)
        return module.build(method, latent_dim, num_heads, value_dim)
    if method in NEW40_VARIANTS:
        module = importlib.import_module('.meaningful_new40_structure', __package__)
        return module.build(method, latent_dim, num_heads, value_dim)
    for family, methods in NEW40_FAMILIES.items():
        if method in methods:
            module = importlib.import_module('.meaningful_new40_' + family, __package__)
            return module.build(method, latent_dim, num_heads, value_dim)
    raise ValueError('Unknown additional-forty method: ' + method)
