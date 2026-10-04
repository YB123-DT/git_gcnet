"""R: supplementary residual, I: adapter-input modulation, N: first-LN replacement.

The five slots are Local/Base/Gap-A/Gap-T/Gap-V. No time-axis tokenization,
cross-example statistics, Memory changes, replacement task head or extra loss.
"""
import importlib

import torch
from torch import nn

from .meaningful_blocks_common import safe_mask
from .priority40_registry import PRIORITY_FAMILIES


def build_operator(method):
    for family, names in PRIORITY_FAMILIES.items():
        if method in names:
            return importlib.import_module('.priority40_' + family, __package__).build(method, dim=64)
    raise ValueError('Unknown priority readout: ' + method)


class PriorityFeatureCore(nn.Module):
    output_dim = 64

    def __init__(self, method, latent_dim=256, num_heads=8, value_dim=64):
        super().__init__()
        self.local = nn.Linear(latent_dim, 64)
        self.memory = nn.Linear(num_heads * value_dim, 64)
        self.role = nn.Parameter(torch.zeros(5, 64))
        self.operator = build_operator(method)

    def forward(self, local, evidence, active, availability):
        valid = active.any(-1)
        mask = torch.cat((valid[:, None], active.bool()), -1)
        q = self.local(safe_mask(local, valid))[:, None]
        k = self.memory(safe_mask(evidence, active))
        tokens = safe_mask(torch.cat((q, k), 1) + self.role, mask)
        result = self.operator(tokens, mask)
        if result.shape != (local.shape[0], 64):
            raise RuntimeError('Priority operators return [N,64]')
        return safe_mask(result, valid)


class DynamicTanh(nn.Module):
    """DyT, Zhu et al., CVPR 2025, https://arxiv.org/abs/2503.10622.

    gamma*tanh(alpha*x)+beta. Replaces only emotion_adapter[0], not emotion_norm.
    It is intentionally not function-equivalent to the previous LayerNorm.
    """
    def __init__(self, width):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.weight = nn.Parameter(torch.ones(width))
        self.bias = nn.Parameter(torch.zeros(width))

    def forward(self, x):
        return self.weight * torch.tanh(self.alpha * x) + self.bias


class NormalizationOnlyReadout(nn.Module):
    """No residual parameters: M30 changes exactly one normalization operator."""
    def __init__(self):
        super().__init__()
        self.last_diagnostics = {'method': 'm30_dyt', 'placement': 'emotion_adapter[0]',
                                 'zero_start_equivalence_claimed': False}

    def forward(self, local, base, gap, availability, umask, flat_anchor):
        return torch.zeros_like(flat_anchor)
