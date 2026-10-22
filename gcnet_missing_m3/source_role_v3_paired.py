"""V3 pairs isolating source topology, Local Skip routing and task normalization.

The original V3 core remains the implementation of pair encoding and role
queries. The caller chooses how a Local/task correction enters the original
readout; those paired variants intentionally have identical module graphs.
"""
import torch
from torch import nn

from .source_role_v3 import HierarchicalSourceRoleV3


VARIANTS = ('topology-full', 'topology-diagonal', 'local-adapter',
            'local-shared', 'task-pre-norm', 'task-post-norm')


class PairedV3Block(HierarchicalSourceRoleV3):
    def __init__(self, local_dim=256, forward_dim=512, width=64,
                 pair_hidden=128, output_dim=1600, variant='topology-full'):
        if variant not in VARIANTS:
            raise ValueError(f'unknown paired V3 variant: {variant}')
        if not isinstance(output_dim, int) or isinstance(output_dim, bool) or output_dim <= 0:
            raise ValueError('output_dim must be a positive integer')
        super().__init__(local_dim=local_dim, forward_dim=forward_dim,
                         width=width, pair_hidden=pair_hidden, mode='residual')
        self.variant = variant
        self.output_dim = output_dim
        if not variant.startswith('topology-'):
            # Preserve the parent's construction/RNG order, then expose its
            # four role contexts without altering their attention or norms.
            self.decoders = nn.ModuleList(nn.Identity() for _ in range(4))
            target_dim = local_dim if variant.startswith('local-') else output_dim
            self.output_projection = nn.Linear(4 * width, target_dim)
            nn.init.zeros_(self.output_projection.weight)
            nn.init.zeros_(self.output_projection.bias)

    def _mix_sources(self, tokens, active):
        if self.variant != 'topology-diagonal':
            return super()._mix_sources(tokens, active)
        t, b = tokens.shape[:2]
        safe_tokens = torch.where(active[..., None], tokens, torch.zeros_like(tokens))
        flat_tokens = safe_tokens.reshape(t * b * 4, 3, self.width)
        flat_active = active.reshape(t * b * 4, 3)
        multiple = flat_active.sum(-1) >= 2
        result = flat_tokens.clone()
        if multiple.any():
            identities = self._identities()[None].expand(t * b, 4, 3, self.width)
            keys = (safe_tokens.reshape(t * b, 4, 3, self.width) + identities)
            keys = keys.reshape(t * b * 4, 3, self.width)[multiple]
            values = flat_tokens[multiple]
            selected_active = flat_active[multiple]
            # Valid queries have exactly one legal key (themselves). Inactive
            # queries are later zeroed, but must still have one legal key to
            # avoid all-masked softmax rows and NaN gradients on older Torch.
            allowed = torch.eye(3, dtype=torch.bool, device=tokens.device)[None]
            allowed = allowed & selected_active[:, None, :]
            fallback = selected_active.to(torch.int64).argmax(-1)
            fallback_key = torch.nn.functional.one_hot(fallback, 3).bool()
            allowed = allowed | ((~selected_active)[..., None] & fallback_key[:, None, :])
            blocked = (~allowed).repeat_interleave(self.source_attention.num_heads, dim=0)
            related, _ = self.source_attention(
                keys, keys, values, attn_mask=blocked,
                key_padding_mask=~selected_active, need_weights=False)
            result[multiple] = self.source_norm(values + related)
        result = torch.where(flat_active[..., None], result, torch.zeros_like(result))
        return result.reshape(t, b, 4, 3, self.width)

    def forward(self, local, source_base, source_gap, trajectory, availability, umask):
        role_outputs = super().forward(
            local, source_base, source_gap, trajectory, availability, umask)
        if self.variant.startswith('topology-'):
            return role_outputs
        enabled = ((umask.transpose(0, 1) > 0)
                   & (trajectory[..., 3] > 0).any(-1)
                   & (~(availability > 0)).any(-1))
        delta = self.output_projection(role_outputs.flatten(-2))
        delta = torch.where(enabled[..., None], delta, torch.zeros_like(delta))
        with torch.no_grad():
            self.last_diagnostics = dict(self.last_diagnostics)
            self.last_diagnostics['context_norm'] = self.last_diagnostics['delta_norm']
            self.last_diagnostics['per_role_context_norm'] = self.last_diagnostics.pop('per_role_delta_norm')
            self.last_diagnostics['delta_norm'] = (
                delta.norm(dim=-1).sum() / enabled.sum().clamp_min(1))
        return delta

    def expected_gradient_groups(self):
        groups = ('memory_projection', 'local_projection', 'pair_mlp',
                  'source_attention', 'source_norm', 'role_queries',
                  'output_attention')
        return groups + (('decoders',) if self.variant.startswith('topology-')
                         else ('output_projection',))

    def gradient_structure_notes(self):
        if self.variant == 'topology-diagonal':
            return ('Layer 2 singleton softmax makes source_attention Q/K '
                    'projection derivatives zero; V/O and source_norm remain '
                    'trainable for rows with at least two valid sources. '
                    'Single-source rows retain the original bypass.')
        return ('The zero-initialized output mapping initially blocks gradients '
                'to earlier V3 layers; test cumulative coverage after updates.')
