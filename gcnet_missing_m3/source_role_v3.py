"""Hierarchical same-source Base--Gap and same-role cross-source interactions.

Only forward-half read corrections are returned. The original CHM, Local Skip,
Flat adapter and final normalization remain the caller's unchanged paths.
Sources denote Value-slot contributions of the joint update, not pure semantic
modalities. Trajectories are strict-past count/coverage/age/ever-seen statistics.
"""
import torch
from torch import nn


class HierarchicalSourceRoleV3(nn.Module):
    def __init__(self, local_dim=256, forward_dim=512):
        super().__init__()
        for name, value in (('local_dim', local_dim), ('forward_dim', forward_dim)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        self.local_dim = local_dim
        self.forward_dim = forward_dim
        # A single coordinate system for products/differences across all roles.
        self.memory_projection = nn.Linear(forward_dim, 64, bias=False)
        self.local_projection = nn.Linear(local_dim, 64)
        self.source_embedding = nn.Embedding(3, 8)
        self.gap_role_embedding = nn.Embedding(3, 8)
        self.pair_mlp = nn.Sequential(
            nn.LayerNorm(343), nn.Linear(343, 128), nn.GELU(), nn.Linear(128, 64))
        self.attention_source_embedding = nn.Embedding(3, 64)
        self.attention_role_embedding = nn.Embedding(4, 64)
        self.source_attention = nn.MultiheadAttention(64, 4, dropout=0., batch_first=True)
        self.source_norm = nn.LayerNorm(64)
        self.role_queries = nn.ModuleList(nn.Linear(local_dim, 64) for _ in range(4))
        self.output_attention = nn.MultiheadAttention(64, 4, dropout=0., batch_first=True)
        self.decoders = nn.ModuleList(nn.Linear(64, forward_dim) for _ in range(4))
        for decoder in self.decoders:
            nn.init.zeros_(decoder.weight)
            nn.init.zeros_(decoder.bias)
        self.last_diagnostics = {}

    def _identities(self):
        return (self.attention_role_embedding.weight[:, None, :]
                + self.attention_source_embedding.weight[None, :, :])

    def _mix_sources(self, tokens, active):
        """Independent role rows: zero->zero; one->identity; >=2->MHA+LN.

        Embeddings enter attention Q/K, not values. Single-source rows bypass
        embeddings, attention and LayerNorm entirely, retaining exact tokens.
        """
        t, b = tokens.shape[:2]
        safe_tokens = torch.where(active[..., None], tokens, torch.zeros_like(tokens))
        flat_tokens = safe_tokens.reshape(t * b * 4, 3, 64)
        flat_active = active.reshape(t * b * 4, 3)
        count = flat_active.sum(-1)
        result = flat_tokens.clone()
        multiple = count >= 2
        if multiple.any():
            identities = self._identities()[None].expand(t * b, 4, 3, 64)
            keys = (safe_tokens.reshape(t * b, 4, 3, 64) + identities)
            keys = keys.reshape(t * b * 4, 3, 64)[multiple]
            values = flat_tokens[multiple]
            related, _ = self.source_attention(
                keys, keys, values, key_padding_mask=~flat_active[multiple], need_weights=False)
            result[multiple] = self.source_norm(values + related)
        result = torch.where(flat_active[..., None], result, torch.zeros_like(result))
        return result.reshape(t, b, 4, 3, 64)

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
        projected = self.memory_projection(raw)
        safe_local = torch.where(enabled[..., None], local, torch.zeros_like(local))
        current = self.local_projection(safe_local)
        meta = torch.where(seen[..., None], trajectory, torch.zeros_like(trajectory))
        # Same explicit encoding as prior trajectory experiments; never-seen age
        # stays zero and is distinguished by ever_seen=0, rather than infinity.
        meta = torch.stack((meta[..., 0].clamp_min(0).log1p(), meta[..., 1],
                            meta[..., 2].clamp_min(0).log1p(), meta[..., 3]), dim=-1)
        base = projected[:, :, 0].unsqueeze(2).expand(t, b, 3, 3, 64)
        gap = projected[:, :, 1:]
        features = torch.cat((base, gap, base * gap, (base - gap).abs(),
            current[:, :, None, None, :].expand(t, b, 3, 3, 64),
            availability[:, :, None, None, :].to(local.dtype).expand(t, b, 3, 3, 3),
            meta[:, :, None, :, :].to(local.dtype).expand(t, b, 3, 3, 4),
            self.source_embedding.weight[None, None, None].expand(t, b, 3, 3, 8),
            self.gap_role_embedding.weight[None, None, :, None].expand(t, b, 3, 3, 8)), dim=-1)
        features = torch.where(pair_active[..., None], features, torch.zeros_like(features))
        messages = self.pair_mlp(features)
        messages = torch.where(pair_active[..., None], messages, torch.zeros_like(messages))
        # Only aggregate AFTER independently encoding each Base--Gap pair.
        base_message = messages.sum(2) / pair_active.sum(2).clamp_min(1)[..., None]
        tokens = projected + torch.cat((base_message.unsqueeze(2), messages), dim=2)
        tokens = torch.where(active[..., None], tokens, torch.zeros_like(tokens))
        mixed = self._mix_sources(tokens, active)

        flat_mixed = mixed.reshape(t * b * 4, 3, 64)
        flat_active = active.reshape(t * b * 4, 3)
        role_rows = roles.reshape(t * b * 4)
        context = local.new_zeros(t * b * 4, 64)
        weights = local.new_zeros(t * b * 4, 3)
        if role_rows.any():
            query = torch.stack([p(safe_local) for p in self.role_queries], dim=2)
            query = query.reshape(t * b * 4, 1, 64)[role_rows]
            identities = self._identities()[None].expand(t * b, 4, 3, 64)
            keys = (mixed.reshape(t * b, 4, 3, 64) + identities)
            keys = keys.reshape(t * b * 4, 3, 64)[role_rows]
            selected, attention = self.output_attention(
                query, keys, flat_mixed[role_rows],
                key_padding_mask=~flat_active[role_rows])
            context[role_rows] = selected.squeeze(1)
            weights[role_rows] = attention.squeeze(1)
        context = context.reshape(t, b, 4, 64)
        delta = torch.stack([p(context[:, :, role])
                             for role, p in enumerate(self.decoders)], dim=2)
        delta = torch.where(roles[..., None], delta, torch.zeros_like(delta))
        with torch.no_grad():
            role_norm = delta.norm(dim=-1).sum((0, 1)) / roles.sum((0, 1)).clamp_min(1)
            self.last_diagnostics = {
                'delta_norm': delta.norm(dim=-1).sum() / roles.sum().clamp_min(1),
                'per_role_delta_norm': role_norm,
                'active_source_mean': seen.sum().to(local.dtype) / valid.sum().clamp_min(1),
                'active_pair_mean': pair_active.sum().to(local.dtype) / valid.sum().clamp_min(1),
                'cross_attention_weights': weights.detach().reshape(t, b, 4, 3),
            }
        return delta
