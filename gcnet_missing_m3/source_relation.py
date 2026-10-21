"""Source-preserving Base/Gap relations with a zero-initialized Flat residual."""

import torch
from torch import nn


class SourceRelationAttention(nn.Module):
    """Attend to strict-past source relations using the current Local query.

    Base sources and Gap sources use the same audio/text/visual order. The Gap
    axes are target, source. Trajectories contain count, coverage, age and
    ever-seen state from before the current utterance writes to memory.
    """

    def __init__(self, local_dim=256, forward_dim=512, output_dim=1600):
        super().__init__()
        for name, value in (('local_dim', local_dim), ('forward_dim', forward_dim),
                            ('output_dim', output_dim)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        self.local_dim = local_dim
        self.forward_dim = forward_dim
        self.output_dim = output_dim
        self.base_projection = nn.Linear(forward_dim, 64, bias=False)
        self.gap_projections = nn.ModuleList(
            nn.Linear(forward_dim, 64, bias=False) for _ in range(3))
        self.relation_mlp = nn.Sequential(
            nn.LayerNorm(263), nn.Linear(263, 128), nn.GELU(), nn.Linear(128, 128))
        self.source_embedding = nn.Parameter(torch.empty(3, 128))
        nn.init.normal_(self.source_embedding, std=.02)
        self.source_self_attention = nn.MultiheadAttention(
            128, 4, dropout=0., batch_first=True)
        self.local_projection = nn.Linear(local_dim, 128)
        self.local_cross_attention = nn.MultiheadAttention(
            128, 4, dropout=0., batch_first=True)
        self.output_projection = nn.Linear(128, output_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self.last_diagnostics = {}

    def forward(self, local, source_base, source_gap, trajectory, availability, umask):
        if local.ndim != 3 or local.shape[-1] != self.local_dim:
            raise ValueError(f'local must have shape [T,B,{self.local_dim}]')
        t, b, _ = local.shape
        expected = ((source_base, (t, b, 3, self.forward_dim), 'source_base'),
                    (source_gap, (t, b, 3, 3, self.forward_dim), 'source_gap'),
                    (trajectory, (t, b, 3, 4), 'trajectory'),
                    (availability, (t, b, 3), 'availability'),
                    (umask, (b, t), 'umask'))
        for value, shape, name in expected:
            if tuple(value.shape) != shape:
                raise ValueError(f'{name} must have shape {shape}')

        valid = umask.transpose(0, 1) > 0
        seen = (trajectory[..., 3] > 0) & valid.unsqueeze(-1)
        has_history = seen.any(dim=-1)
        gap_active = (~(availability > 0)).unsqueeze(-1) & seen.unsqueeze(2)

        # Mask before projections and nonlinearities: multiplying by zero would
        # preserve NaNs in inactive reads and poison parameter gradients.
        base = torch.where(seen.unsqueeze(-1), source_base, torch.zeros_like(source_base))
        gap = torch.where(gap_active.unsqueeze(-1), source_gap, torch.zeros_like(source_gap))
        base = self.base_projection(base)
        projected_gap = torch.stack([
            projection(gap[:, :, target])
            for target, projection in enumerate(self.gap_projections)], dim=2)
        projected_gap = torch.where(gap_active.unsqueeze(-1), projected_gap,
                                    torch.zeros_like(projected_gap))
        gap_mean = projected_gap.sum(dim=2) / gap_active.sum(dim=2).clamp_min(1).unsqueeze(-1)

        metadata = torch.where(seen.unsqueeze(-1), trajectory, torch.zeros_like(trajectory))
        metadata = torch.stack((metadata[..., 0].clamp_min(0).log1p(),
                                metadata[..., 1],
                                metadata[..., 2].clamp_min(0).log1p(),
                                metadata[..., 3]), dim=-1)
        current = availability.unsqueeze(2).expand(t, b, 3, 3)
        current = torch.where(seen.unsqueeze(-1), current, torch.zeros_like(current))
        features = torch.cat((base, gap_mean, base * gap_mean, (base - gap_mean).abs(),
                              current.to(local.dtype), metadata.to(local.dtype)), dim=-1)
        tokens = self.relation_mlp(features) + self.source_embedding
        tokens = torch.where(seen.unsqueeze(-1), tokens, torch.zeros_like(tokens))
        local = torch.where(has_history.unsqueeze(-1), local, torch.zeros_like(local))

        flat_tokens = tokens.reshape(t * b, 3, 128)
        flat_seen = seen.reshape(t * b, 3)
        active_rows = has_history.reshape(t * b)
        context = tokens.new_zeros(t * b, 128)
        cross_weights = tokens.new_zeros(t * b, 3)
        # Avoid all-masked softmax rows entirely, including first utterances.
        if active_rows.any():
            selected_tokens = flat_tokens[active_rows]
            key_padding_mask = ~flat_seen[active_rows]
            source_context, _ = self.source_self_attention(
                selected_tokens, selected_tokens, selected_tokens,
                key_padding_mask=key_padding_mask, need_weights=False)
            query = self.local_projection(local.reshape(t * b, self.local_dim)[active_rows])
            attended, weights = self.local_cross_attention(
                query.unsqueeze(1), source_context, source_context,
                key_padding_mask=key_padding_mask)
            context[active_rows] = attended.squeeze(1)
            cross_weights[active_rows] = weights.squeeze(1)

        residual = self.output_projection(context).reshape(t, b, self.output_dim)
        residual = torch.where(has_history.unsqueeze(-1), residual, torch.zeros_like(residual))
        with torch.no_grad():
            self.last_diagnostics = {
                'residual_norm': residual.norm(dim=-1).sum() / has_history.sum().clamp_min(1),
                'source_token_norm': tokens.norm(dim=-1).sum() / seen.sum().clamp_min(1),
                'active_source_mean': seen.sum().to(local.dtype) / valid.sum().clamp_min(1),
                'cross_attention_weights': cross_weights.detach().reshape(t, b, 3),
            }
        return residual
