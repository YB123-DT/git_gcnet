"""Complete VQA interaction cores as identity-start Flat-input transforms.

Source equations and pinned author code are recorded in
experiments/osram_meaningful20_round2_20261004/vqa_candidates.json.
Independent PyTorch implementation: no source code copied, no new dependencies.
Only MCAN, DCN and basic MAC are accepted here; historical BAN/MFH are excluded.

Base's real heads form Q; Local followed by active Gap heads form V. There is
no invented spatial order, duplicated Base, persistent memory access or task
head. The caller owns valid-prefix/no-history filtering and the original Local
skip; outputs are replacements for emotion_adapter inputs only. Output affine
bridges start at zero; original inputs therefore survive exactly on active slots.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .meaningful_blocks_common import active_groups, safe_mask


VQA_METHODS = ('mcan_encoder_decoder', 'dense_coattention',
               'mac_control_read_write')
_DIM = 128


def _zero_linear(in_features, out_features):
    layer = nn.Linear(in_features, out_features)
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class _MCANorm(nn.Module):
    """Author net_utils.LayerNorm: sample std, epsilon outside the square root."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(_DIM))
        self.bias = nn.Parameter(torch.zeros(_DIM))

    def forward(self, x):
        return self.weight * (x - x.mean(-1, keepdim=True)) / (
            x.std(-1, keepdim=True, unbiased=True) + 1e-6) + self.bias


class _MCAAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 4
        self.query = nn.Linear(_DIM, _DIM)
        self.key = nn.Linear(_DIM, _DIM)
        self.value = nn.Linear(_DIM, _DIM)
        self.merge = nn.Linear(_DIM, _DIM)

    def forward(self, query, context):
        def split(x):
            return x.reshape(x.shape[0], x.shape[1], self.heads,
                             _DIM // self.heads).transpose(1, 2)
        q, k, v = split(self.query(query)), split(self.key(context)), split(self.value(context))
        attention = (q @ k.transpose(-2, -1) / math.sqrt(_DIM // self.heads)).softmax(-1)
        result = (attention @ v).transpose(1, 2).reshape(query.shape[0], query.shape[1], _DIM)
        return self.merge(result)


class _MCAEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = _MCAAttention()
        self.ffn = nn.Sequential(nn.Linear(_DIM, 4 * _DIM), nn.ReLU(),
                                 nn.Linear(4 * _DIM, _DIM))
        self.norm1, self.norm2 = _MCANorm(), _MCANorm()

    def forward(self, q):
        q = self.norm1(q + self.attention(q, q))
        return self.norm2(q + self.ffn(q))


class _MCADecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attention, self.guided_attention = _MCAAttention(), _MCAAttention()
        self.ffn = nn.Sequential(nn.Linear(_DIM, 4 * _DIM), nn.ReLU(),
                                 nn.Linear(4 * _DIM, _DIM))
        self.norm1, self.norm2, self.norm3 = _MCANorm(), _MCANorm(), _MCANorm()

    def forward(self, v, q):
        v = self.norm1(v + self.self_attention(v, v))
        v = self.norm2(v + self.guided_attention(v, q))
        return self.norm3(v + self.ffn(v))


class _DenseSymmetricLayer(nn.Module):
    """Source parallel-average mode, not projected-value multihead attention."""

    def __init__(self):
        super().__init__()
        self.heads = 4
        self.null_q = nn.Parameter(torch.randn(3, _DIM))
        self.null_v = nn.Parameter(torch.randn(3, _DIM))
        self.project_q = nn.Linear(_DIM, _DIM, bias=False)
        self.project_v = nn.Linear(_DIM, _DIM, bias=False)
        self.fuse_q = nn.Linear(2 * _DIM, _DIM)
        self.fuse_v = nn.Linear(2 * _DIM, _DIM)

    def forward(self, q, v):
        batch = q.shape[0]
        q_all = torch.cat((self.null_q.expand(batch, -1, -1), q), 1)
        v_all = torch.cat((self.null_v.expand(batch, -1, -1), v), 1)
        q_keys = self.project_q(q_all).reshape(batch, -1, self.heads,
                                             _DIM // self.heads).transpose(1, 2)
        v_keys = self.project_v(v_all).reshape(batch, -1, self.heads,
                                             _DIM // self.heads).transpose(1, 2)
        affinity = q_keys @ v_keys.transpose(-2, -1) / math.sqrt(_DIM // self.heads)
        # The same affinity induces two conditional distributions. Full-width
        # values are never split/projected, and parallel reads are averaged.
        read_v = (affinity.softmax(-1) @ v_all[:, None]).mean(1)[:, 3:]
        read_q = (affinity.transpose(-2, -1).softmax(-1) @ q_all[:, None]).mean(1)[:, 3:]
        return (q + self.fuse_q(torch.cat((q, read_v), -1)).relu(),
                v + self.fuse_v(torch.cat((v, read_q), -1)).relu())


class _MACCell(nn.Module):
    """Paper's complete basic control/read/write cell; optional write gates off."""

    def __init__(self):
        super().__init__()
        self.control_merge = nn.Linear(2 * _DIM, _DIM)
        self.control_score = nn.Linear(_DIM, 1)
        self.memory_project = nn.Linear(_DIM, _DIM)
        self.knowledge_project = nn.Linear(_DIM, _DIM)
        self.read_merge = nn.Linear(2 * _DIM, _DIM)
        self.read_score = nn.Linear(_DIM, 1)
        self.write = nn.Linear(2 * _DIM, _DIM)

    def forward(self, control, memory, step_query, q, v):
        cq = self.control_merge(torch.cat((control, step_query), -1))
        control_attention = self.control_score(cq[:, None] * q).softmax(1)
        control = (control_attention * q).sum(1)
        interaction = self.memory_project(memory)[:, None] * self.knowledge_project(v)
        interaction = self.read_merge(torch.cat((interaction, v), -1))
        read_attention = self.read_score(control[:, None] * interaction).softmax(1)
        retrieved = (read_attention * v).sum(1)
        memory = self.write(torch.cat((retrieved, memory), -1))
        return control, memory


class _VQAInput(nn.Module):
    """Typed role/head tokenization, active-bank packing and input writeback."""

    def __init__(self, latent_dim, num_heads, value_dim, update_tokens):
        super().__init__()
        if any(not isinstance(x, int) or isinstance(x, bool) or x <= 0
               for x in (latent_dim, num_heads, value_dim)):
            raise ValueError('latent_dim, num_heads and value_dim must be positive integers')
        self.latent_dim, self.num_heads, self.value_dim = latent_dim, num_heads, value_dim
        self.local_project = nn.Linear(latent_dim, _DIM)
        self.memory_project = nn.ModuleList([nn.Linear(value_dim, _DIM) for _ in range(4)])
        self.role_embedding = nn.Embedding(5, _DIM)
        self.head_embedding = nn.Embedding(num_heads, _DIM)
        self.update_tokens = update_tokens
        self.local_bridge = _zero_linear(_DIM if update_tokens else 2 * _DIM, latent_dim)
        if update_tokens:
            self.memory_bridges = nn.ModuleList([_zero_linear(_DIM, value_dim) for _ in range(4)])

    def _tokenize(self, local, evidence, active):
        values = evidence.reshape(-1, 4, self.num_heads, self.value_dim)
        heads = self.head_embedding.weight[None]
        memory = torch.stack([
            project(values[:, role]) + heads + self.role_embedding.weight[role + 1]
            for role, project in enumerate(self.memory_project)
        ], 1)
        local_token = self.local_project(local) + self.role_embedding.weight[0]
        v = torch.cat((local_token[:, None], memory[:, 1:].flatten(1, 2)), 1)
        v_mask = torch.cat((torch.ones_like(active[:, :1]),
                            active[:, 1:, None].expand(-1, -1, self.num_heads).flatten(1)), 1)
        return memory[:, 0], safe_mask(v, v_mask), v_mask

    def forward(self, local, evidence, active, availability):
        batch = local.shape[0]
        if (local.shape != (batch, self.latent_dim)
                or evidence.shape != (batch, 4, self.num_heads * self.value_dim)
                or active.shape != (batch, 4) or availability.shape != (batch, 3)):
            raise ValueError('Expected Local[N,D], evidence[N,4,H*V], active[N,4], availability[N,3]')
        active = torch.cat((active[:, :1].bool(),
                            active[:, 1:].bool() & ~availability.bool()), 1)
        safe_local = safe_mask(local, active.any(-1))
        safe_evidence = safe_mask(evidence, active)
        local_out, evidence_out = safe_local, safe_evidence
        eligible = active[:, 0].nonzero(as_tuple=True)[0]
        if not eligible.numel():
            return local_out, evidence_out
        q, v, v_mask = self._tokenize(safe_local[eligible], safe_evidence[eligible], active[eligible])
        # Packing avoids all-masked attention and never treats absent/padded
        # projected biases as evidence. Local is always packed column zero.
        for group_rows, columns, packed_v in active_groups(v, v_mask):
            rows = eligible[group_rows]
            q_final, v_final, local_feature = self._interact(
                q[group_rows], packed_v, safe_local[rows])
            local_out = local_out.index_copy(
                0, rows, safe_local[rows] + self.local_bridge(local_feature))
            if self.update_tokens:
                unpacked = v.new_zeros((rows.numel(), v.shape[1], _DIM)).index_copy(1, columns, v_final)
                gap_tokens = unpacked[:, 1:].reshape(-1, 3, self.num_heads, _DIM)
                delta = torch.stack([
                    self.memory_bridges[role](q_final if role == 0 else gap_tokens[:, role - 1])
                    for role in range(4)
                ], 1).flatten(2)
                corrected = safe_mask(safe_evidence[rows] + delta, active[rows])
                evidence_out = evidence_out.index_copy(0, rows, corrected)
        return local_out, evidence_out


class MCANInput(_VQAInput):
    def __init__(self, latent_dim, num_heads, value_dim):
        super().__init__(latent_dim, num_heads, value_dim, update_tokens=True)
        self.encoder = nn.ModuleList([_MCAEncoder() for _ in range(6)])
        self.decoder = nn.ModuleList([_MCADecoder() for _ in range(6)])

    def _interact(self, q, v, local):
        for layer in self.encoder:
            q = layer(q)
        for layer in self.decoder:
            v = layer(v, q)
        return q, v, v[:, 0]


class DenseCoattentionInput(_VQAInput):
    def __init__(self, latent_dim, num_heads, value_dim):
        super().__init__(latent_dim, num_heads, value_dim, update_tokens=True)
        self.layers = nn.ModuleList([_DenseSymmetricLayer() for _ in range(5)])

    def _interact(self, q, v, local):
        for layer in self.layers:
            q, v = layer(q, v)
        return q, v, v[:, 0]


class MACInput(_VQAInput):
    def __init__(self, latent_dim, num_heads, value_dim):
        super().__init__(latent_dim, num_heads, value_dim, update_tokens=False)
        self.control_initial = nn.Parameter(torch.zeros(_DIM))
        self.memory_initial = nn.Parameter(torch.zeros(_DIM))
        self.step_query = nn.ModuleList([nn.Linear(_DIM, _DIM) for _ in range(4)])
        self.cell = _MACCell()

    def _interact(self, q, v, local):
        query = self.local_project(local)
        control = self.control_initial.expand(local.shape[0], -1)
        memory = self.memory_initial.expand(local.shape[0], -1)
        for project in self.step_query:
            control, memory = self.cell(control, memory, project(query), q, v)
        return None, None, torch.cat((query, memory), -1)


def build_vqa(method, latent_dim=256, num_heads=8, value_dim=64):
    constructors = {'mcan_encoder_decoder': MCANInput,
                    'dense_coattention': DenseCoattentionInput,
                    'mac_control_read_write': MACInput}
    if method not in constructors:
        raise ValueError(f'Unknown VQA input method: {method}')
    # Construction is CPU-only. Preserve the legacy model's initialization RNG
    # even if a caller uses this factory without the outer wrapper's fork.
    with torch.random.fork_rng(devices=[]):
        return constructors[method](latent_dim, num_heads, value_dim)
