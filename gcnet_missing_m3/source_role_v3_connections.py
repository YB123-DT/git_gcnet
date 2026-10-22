"""Three V3 connections sharing same-source pairs and same-role source mixing.

``source-slots`` decodes the fixed A/T/V slots without a final role query.
``flat-hidden`` returns a nonzero hidden branch for the caller's zero-init Flat
W2. ``head-preserving`` processes each memory head independently and returns
four zero-init forward corrections. None of these modules changes memory writes.
"""
import math

import torch
from torch import nn


class V3ConnectionBlock(nn.Module):
    def __init__(self, local_dim=256, forward_dim=512, width=64, pair_hidden=128,
                 num_memory_heads=8, flat_hidden_dim=1600, variant='head-preserving'):
        super().__init__()
        if variant not in ('head-preserving', 'source-slots', 'flat-hidden'):
            raise ValueError('variant must be head-preserving, source-slots or flat-hidden')
        for name, value in (('local_dim', local_dim), ('forward_dim', forward_dim),
                            ('width', width), ('pair_hidden', pair_hidden),
                            ('num_memory_heads', num_memory_heads),
                            ('flat_hidden_dim', flat_hidden_dim)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if width % 4:
            raise ValueError('width must be divisible by the fixed four attention heads')
        if forward_dim % num_memory_heads:
            raise ValueError('forward_dim must be divisible by num_memory_heads')
        self.variant = variant
        self.local_dim = local_dim
        self.forward_dim = forward_dim
        self.width = width
        self.num_memory_heads = num_memory_heads
        self.flat_hidden_dim = flat_hidden_dim
        self.head_dim = forward_dim // num_memory_heads

        if variant == 'head-preserving':
            # One coordinate system per memory head, shared across roles/sources.
            self.head_projections = nn.ModuleList(
                nn.Linear(self.head_dim, width, bias=False) for _ in range(num_memory_heads))
            self.head_embedding = nn.Parameter(torch.zeros(num_memory_heads, width))
        else:
            self.memory_projection = nn.Linear(forward_dim, width, bias=False)
        self.local_projection = nn.Linear(local_dim, width)
        self.source_embedding = nn.Embedding(3, 8)
        self.gap_role_embedding = nn.Embedding(3, 8)
        self.pair_mlp = nn.Sequential(
            nn.LayerNorm(5 * width + 23), nn.Linear(5 * width + 23, pair_hidden),
            nn.GELU(), nn.Linear(pair_hidden, width))
        self.attention_source_embedding = nn.Embedding(3, width)
        self.attention_role_embedding = nn.Embedding(4, width)
        self.source_attention = nn.MultiheadAttention(width, 4, dropout=0., batch_first=True)
        self.source_norm = nn.LayerNorm(width)
        if variant != 'source-slots':
            self.role_queries = nn.ModuleList(nn.Linear(local_dim, width) for _ in range(4))
            self.output_attention = nn.MultiheadAttention(width, 4, dropout=0., batch_first=True)
        if variant == 'flat-hidden':
            self.flat_norm = nn.LayerNorm(4 * width)
            self.flat_projection = nn.Linear(4 * width, flat_hidden_dim, bias=False)
        else:
            if variant == 'source-slots':
                self.decoders = nn.ModuleList(nn.Linear(3 * width, forward_dim) for _ in range(4))
            else:
                self.decoders = nn.ModuleList(nn.ModuleList(
                    nn.Linear(width, self.head_dim) for _ in range(num_memory_heads))
                    for _ in range(4))
            for decoder in self.decoders.modules():
                if isinstance(decoder, nn.Linear):
                    nn.init.zeros_(decoder.weight)
                    nn.init.zeros_(decoder.bias)
        self.last_diagnostics = {}

    def expected_gradient_groups(self):
        """Named-parameter prefixes for modules present in this connection only."""
        groups = tuple(name + '.' for name, _ in self.named_children())
        return groups + (('head_embedding',) if self.variant == 'head-preserving' else ())

    def _identities(self):
        return (self.attention_role_embedding.weight[:, None, :]
                + self.attention_source_embedding.weight[None, :, :])

    def _pair_tokens(self, projected, current, meta, availability, active):
        """Keep every active Base/Gap pair separate until its message is encoded."""
        prefix = projected.shape[:-3]
        pair_active = active[..., 1:, :]
        base = projected[..., 0, :, :].unsqueeze(-3).expand(*prefix, 3, 3, self.width)
        gap = projected[..., 1:, :, :]
        singleton = (1,) * len(prefix)
        source_identity = self.source_embedding.weight.reshape(*singleton, 1, 3, 8)
        gap_identity = self.gap_role_embedding.weight.reshape(*singleton, 3, 1, 8)
        features = torch.cat((base, gap, base * gap, (base - gap).abs(),
            current[..., None, None, :].expand(*prefix, 3, 3, self.width),
            availability[..., None, None, :].expand(*prefix, 3, 3, 3),
            meta[..., None, :, :].expand(*prefix, 3, 3, 4),
            source_identity.expand(*prefix, 3, 3, 8),
            gap_identity.expand(*prefix, 3, 3, 8)), dim=-1)
        features = torch.where(pair_active[..., None], features, torch.zeros_like(features))
        messages = self.pair_mlp(features)
        messages = torch.where(pair_active[..., None], messages, torch.zeros_like(messages))
        base_message = messages.sum(-3) / pair_active.sum(-2).clamp_min(1)[..., None]
        tokens = projected + torch.cat((base_message.unsqueeze(-3), messages), dim=-3)
        return torch.where(active[..., None], tokens, torch.zeros_like(tokens))

    def _mix_sources(self, tokens, active):
        """Only three sources attend: zero->zero, one->identity, >=2->MHA+LN.

        Leading axes are independent cells (and memory heads for that variant).
        Roles never enter the source sequence. Embeddings affect Q/K only.
        """
        safe_tokens = torch.where(active[..., None], tokens, torch.zeros_like(tokens))
        cells = math.prod(tokens.shape[:-3])
        values = safe_tokens.reshape(cells * 4, 3, self.width)
        flat_active = active.reshape(cells * 4, 3)
        result = values.clone()
        multiple = flat_active.sum(-1) >= 2
        if multiple.any():
            keys = safe_tokens.reshape(cells, 4, 3, self.width) + self._identities()[None]
            keys = keys.reshape(cells * 4, 3, self.width)[multiple]
            selected_values = values[multiple]
            related, _ = self.source_attention(
                keys, keys, selected_values, key_padding_mask=~flat_active[multiple],
                need_weights=False)
            result[multiple] = self.source_norm(selected_values + related)
        result = torch.where(flat_active[..., None], result, torch.zeros_like(result))
        return result.reshape_as(tokens)

    def _role_context(self, mixed, active, safe_local):
        cells = math.prod(mixed.shape[:-3])
        values = mixed.reshape(cells * 4, 3, self.width)
        flat_active = active.reshape(cells * 4, 3)
        role_rows = flat_active.any(-1)
        context = mixed.new_zeros(cells * 4, self.width)
        if role_rows.any():
            query = torch.stack([layer(safe_local) for layer in self.role_queries], dim=2)
            if self.variant == 'head-preserving':
                query = query[:, :, None] + self.head_embedding[None, None, :, None, :]
            query = query.reshape(cells * 4, 1, self.width)[role_rows]
            keys = mixed.reshape(cells, 4, 3, self.width) + self._identities()[None]
            keys = keys.reshape(cells * 4, 3, self.width)[role_rows]
            selected, _ = self.output_attention(
                query, keys, values[role_rows], key_padding_mask=~flat_active[role_rows],
                need_weights=False)
            context[role_rows] = selected.squeeze(1)
        return context.reshape(*mixed.shape[:-2], self.width)

    def forward(self, local, source_base, source_gap, trajectory, availability, umask):
        if local.ndim != 3 or local.shape[-1] != self.local_dim:
            raise ValueError(f'local must have shape [T,B,{self.local_dim}]')
        t, b, _ = local.shape
        for value, shape, name in (
                (source_base, (t, b, 3, self.forward_dim), 'source_base'),
                (source_gap, (t, b, 3, 3, self.forward_dim), 'source_gap'),
                (trajectory, (t, b, 3, 4), 'trajectory'),
                (availability, (t, b, 3), 'availability'),
                (umask, (b, t), 'umask')):
            if tuple(value.shape) != shape:
                raise ValueError(f'{name} must have shape {shape}')

        valid = umask.transpose(0, 1) > 0
        missing = ~(availability > 0)
        seen = (trajectory[..., 3] > 0) & valid[..., None]
        enabled = valid & missing.any(-1) & seen.any(-1)
        roles = torch.cat((enabled[..., None], enabled[..., None] & missing), dim=-1)
        active = roles[..., None] & seen[:, :, None, :]
        pair_active = active[:, :, 1:]
        raw = torch.cat((source_base.unsqueeze(2), source_gap), dim=2)
        raw = torch.where(active[..., None], raw, torch.zeros_like(raw))
        safe_local = torch.where(enabled[..., None], local, torch.zeros_like(local))
        current = self.local_projection(safe_local)
        meta_active = enabled[..., None] & seen
        meta = torch.where(meta_active[..., None], trajectory, torch.zeros_like(trajectory))
        meta = torch.stack((meta[..., 0].clamp_min(0).log1p(), meta[..., 1],
                            meta[..., 2].clamp_min(0).log1p(), meta[..., 3]), dim=-1)
        meta = meta.to(local.dtype)
        safe_availability = torch.where(enabled[..., None], availability,
                                        torch.zeros_like(availability)).to(local.dtype)

        if self.variant == 'head-preserving':
            raw_heads = raw.reshape(t, b, 4, 3, self.num_memory_heads, self.head_dim)
            projected = torch.stack([layer(raw_heads[..., head, :])
                                     for head, layer in enumerate(self.head_projections)], dim=2)
            current = current[:, :, None, :] + self.head_embedding[None, None, :, :]
            current = torch.where(enabled[..., None, None], current, torch.zeros_like(current))
            meta = meta[:, :, None].expand(t, b, self.num_memory_heads, 3, 4)
            safe_availability = safe_availability[:, :, None].expand(t, b, self.num_memory_heads, 3)
            active = active[:, :, None].expand(t, b, self.num_memory_heads, 4, 3)
        else:
            projected = self.memory_projection(raw)
            current = torch.where(enabled[..., None], current, torch.zeros_like(current))
        tokens = self._pair_tokens(projected, current, meta, safe_availability, active)
        mixed = self._mix_sources(tokens, active)

        if self.variant == 'source-slots':
            output = torch.stack([layer(mixed[:, :, role].reshape(t, b, 3 * self.width))
                                  for role, layer in enumerate(self.decoders)], dim=2)
        else:
            context = self._role_context(mixed, active, safe_local)
            if self.variant == 'head-preserving':
                output = torch.stack([torch.cat([
                    layer(context[:, :, head, role]) for head, layer in enumerate(decoders)],
                    dim=-1) for role, decoders in enumerate(self.decoders)], dim=2)
            else:
                flat = self.flat_norm(context.reshape(t, b, 4 * self.width))
                role_context = torch.where(roles[..., None], flat.reshape(t, b, 4, self.width),
                                           torch.zeros_like(context))
                output = self.flat_projection(role_context.reshape(t, b, 4 * self.width))
        if self.variant == 'flat-hidden':
            output = torch.where(enabled[..., None], output, torch.zeros_like(output))
        else:
            output = torch.where(roles[..., None], output, torch.zeros_like(output))

        with torch.no_grad():
            norms = output.norm(dim=-1)
            denominator = enabled.sum() if self.variant == 'flat-hidden' else roles.sum()
            self.last_diagnostics = {
                'delta_norm': norms.sum() / denominator.clamp_min(1),
                'active_source_mean': seen.sum().to(local.dtype) / valid.sum().clamp_min(1),
                'active_pair_mean': pair_active.sum().to(local.dtype) / valid.sum().clamp_min(1),
            }
            if self.variant == 'flat-hidden':
                # The hidden output mixes roles; label its preprojection role
                # norms explicitly instead of implying four output deltas.
                self.last_diagnostics['per_role_context_norm'] = (
                    role_context.norm(dim=-1).sum((0, 1)) / roles.sum((0, 1)).clamp_min(1))
            else:
                self.last_diagnostics['per_role_delta_norm'] = (
                    norms.sum((0, 1)) / roles.sum((0, 1)).clamp_min(1))
        return output
