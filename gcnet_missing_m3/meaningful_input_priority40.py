"""Opt-in priority operators at Flat adapter inputs; Local skip is unchanged."""
import torch
from torch import nn
from .meaningful_blocks_common import safe_mask
from .priority40_common import PriorityFeatureCore


class XCADirectInput(nn.Module):
    """Role outputs replace evidence, without original-input additions."""
    def __init__(self, latent_dim=256, num_heads=8, value_dim=64,
                 operator_method='m28_xcit_xca'):
        super().__init__()
        self.features = PriorityFeatureCore(operator_method, latent_dim, num_heads, value_dim)
        # Direct decoding replaces pooled readout; do not leave unused parameters.
        if hasattr(self.features.operator, 'readout'):
            del self.features.operator.readout
        self.local_decoder = nn.Linear(64, latent_dim)
        self.memory_decoder = nn.Linear(64, num_heads * value_dim)

    def forward(self, local, evidence, active, availability):
        valid = active.any(-1)
        mask = torch.cat((valid[:, None], active.bool()), -1)
        q = self.features.local(safe_mask(local, valid))[:, None]
        k = self.features.memory(safe_mask(evidence, active))
        tokens = safe_mask(torch.cat((q,k),1) + self.features.role, mask)
        updated = self.features.operator.encode_roles(tokens, mask)
        return (safe_mask(self.local_decoder(updated[:,0]), valid).to(local.dtype),
                safe_mask(self.memory_decoder(updated[:,1:]),active).to(evidence.dtype))


def build_priority40(method, latent_dim=256, num_heads=8, value_dim=64):
    if method == 'm28_xcit_xca_direct':
        return XCADirectInput(latent_dim, num_heads, value_dim)
    if method in ('m03_gatv2_direct', 'm05_pna_direct'):
        return XCADirectInput(latent_dim, num_heads, value_dim,
                              operator_method=method.removesuffix('_direct'))
    from .priority40_conditioning import build_input
    return build_input(method, latent_dim=latent_dim, forward_dim=num_heads * value_dim)
