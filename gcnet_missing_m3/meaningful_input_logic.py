"""Complete four-layer, breadth-three Neural Logic Machine for read evidence.

Independent implementation of Dong et al., arXiv:1904.11694. Author reference:
google/neural-logic-machines@3f8a8966c54d13d2658c77c03793a9a98a283e22,
LogicLayer/LogicMachine, Expander/Reducer/Permutation and LogicInference.
No Jacinle/Jactorch code or dependency is imported.

Objects are Local and complete active Base/Gap role vectors, not individual
heads. Four arities exchange predicates through expansion and interleaved
existential/universal reduction. Every arity includes all variable permutations.
The caller excludes padding/no-history; Local and the original task/Memory path
are untouched. Only zero-start role-specific memory input bridges write back.
"""
from __future__ import annotations

import itertools
import math

import torch
from torch import nn

from .meaningful_blocks_common import active_groups, safe_mask


LOGIC_METHODS = ('neural_logic_machine_evidence',)


def expand_predicates(predicates, objects):
    """Append a variable immediately before the predicate/channel dimension."""
    return predicates.unsqueeze(-2).expand(*predicates.shape[:-1], objects,
                                           predicates.shape[-1])


def quantify_predicates(predicates):
    """Eliminate the last variable; channels are E0,A0,E1,A1,... ."""
    arity = predicates.ndim - 2
    if not 1 <= arity <= 3:
        raise ValueError('Quantification requires arity one, two or three')
    objects = predicates.shape[1]
    if objects == 0 or any(size != objects for size in predicates.shape[1:-1]):
        raise ValueError('All variable axes must share a nonempty object domain')
    indices = torch.arange(objects, device=predicates.device)
    grids = torch.meshgrid(*([indices] * arity), indexing='ij')
    distinct = torch.ones((objects,) * arity, dtype=torch.bool, device=predicates.device)
    for left, right in itertools.combinations(range(arity), 2):
        distinct = distinct & (grids[left] != grids[right])
    # A repeated free variable also invalidates the whole grounding. In a
    # two-object domain all ternary groundings are invalid: exists=0, forall=1.
    mask = distinct.unsqueeze(0).unsqueeze(-1)
    exists = torch.where(mask, predicates, torch.zeros_like(predicates)).max(-2).values
    forall = torch.where(mask, predicates, torch.ones_like(predicates)).min(-2).values
    return torch.stack((exists, forall), -1).flatten(-2)


def permute_predicates(predicates):
    """Concatenate all r! lexicographically ordered variable-axis permutations."""
    arity = predicates.ndim - 2
    if not 0 <= arity <= 3:
        raise ValueError('Supported predicate arities are zero through three')
    if arity <= 1:
        return predicates
    return torch.cat([
        predicates.permute(0, *permutation, arity + 1)
        for permutation in itertools.permutations(range(1, arity + 1))
    ], -1)


class LogicLayer(nn.Module):
    """Simultaneous arity-lattice update; no residual or recursive weight reuse."""

    def __init__(self, input_dims):
        super().__init__()
        if len(input_dims) != 4:
            raise ValueError('A breadth-three layer needs four input dimensions')
        self.input_dims = tuple(input_dims)
        self.inference = nn.ModuleList()
        for arity in range(4):
            channels = input_dims[arity]
            if arity:
                channels += input_dims[arity - 1]
            if arity < 3:
                channels += 2 * input_dims[arity + 1]
            if channels == 0:
                raise ValueError('This complete-core configuration expects every output arity')
            # Fixed hidden32 is a compact adaptation of the author's configurable
            # LogicInference MLP. Its weights are shared across all groundings.
            self.inference.append(nn.Sequential(
                nn.Linear(channels * math.factorial(arity), 32), nn.ReLU(),
                nn.Linear(32, 16), nn.Sigmoid()))

    def forward(self, predicates):
        objects = predicates[1].shape[1]
        outputs = []
        for arity, infer in enumerate(self.inference):
            pieces = []
            if arity and self.input_dims[arity - 1]:
                pieces.append(expand_predicates(predicates[arity - 1], objects))
            if self.input_dims[arity]:
                pieces.append(predicates[arity])
            if arity < 3 and self.input_dims[arity + 1]:
                pieces.append(quantify_predicates(predicates[arity + 1]))
            outputs.append(infer(permute_predicates(torch.cat(pieces, -1))))
        return outputs


class NeuralLogicInput(nn.Module):
    def __init__(self, latent_dim, num_heads, value_dim):
        super().__init__()
        if any(not isinstance(x, int) or isinstance(x, bool) or x <= 0
               for x in (latent_dim, num_heads, value_dim)):
            raise ValueError('Expected positive integer Local/head dimensions')
        self.latent_dim, self.forward_dim = latent_dim, num_heads * value_dim
        self.local_project = nn.Linear(latent_dim, 32)
        self.memory_project = nn.ModuleList([nn.Linear(self.forward_dim, 32) for _ in range(4)])
        self.role_embedding = nn.Embedding(5, 32)
        self.unary = nn.Sequential(nn.Linear(32, 16), nn.Sigmoid())
        self.binary = nn.Sequential(nn.Linear(64, 32), nn.ReLU(),
                                    nn.Linear(32, 4), nn.Sigmoid())
        self.layers = nn.ModuleList([LogicLayer((0, 16, 4, 0))]
                                    + [LogicLayer((16, 16, 16, 16)) for _ in range(3)])
        self.memory_bridges = nn.ModuleList([nn.Linear(32, self.forward_dim) for _ in range(4)])
        for bridge in self.memory_bridges:
            nn.init.zeros_(bridge.weight)
            nn.init.zeros_(bridge.bias)

    def _infer(self, tokens):
        objects = tokens.shape[1]
        left = tokens[:, :, None].expand(-1, -1, objects, -1)
        right = tokens[:, None, :].expand(-1, objects, -1, -1)
        predicates = [None, self.unary(tokens), self.binary(torch.cat((left, right), -1)), None]
        for layer in self.layers:
            predicates = layer(predicates)
        # The complete last layer computes all arities. Only nullary and unary
        # are read out, as in an NLM unary task; terminal binary/ternary-specific
        # inference parameters therefore have no task gradient. Earlier ternary
        # information returns to unary through successive reductions.
        global_predicates = predicates[0][:, None].expand(-1, objects, -1)
        return torch.cat((predicates[1], global_predicates), -1)

    def forward(self, local, evidence, active, availability):
        batch = local.shape[0]
        if (local.shape != (batch, self.latent_dim)
                or evidence.shape != (batch, 4, self.forward_dim)
                or active.shape != (batch, 4) or availability.shape != (batch, 3)):
            raise ValueError('Expected Local[N,D], evidence[N,4,H*V], active[N,4], availability[N,3]')
        active = torch.cat((active[:, :1].bool(),
                            active[:, 1:].bool() & ~availability.bool()), 1)
        safe_evidence = safe_mask(evidence, active)
        # Local is sanitized for internal computation only; the returned Local
        # is exactly the original input, including rows the outer wrapper skips.
        safe_local = safe_mask(local, active.any(-1))
        mask = torch.cat((active.any(-1)[:, None], active), 1)
        memory_tokens = torch.stack([project(safe_evidence[:, role])
                                     for role, project in enumerate(self.memory_project)], 1)
        tokens = torch.cat((self.local_project(safe_local)[:, None], memory_tokens), 1)
        tokens = safe_mask(tokens + self.role_embedding.weight[None], mask)
        evidence_out = safe_evidence
        for rows, columns, packed in active_groups(tokens, mask):
            if columns.numel() <= 1:
                continue
            features = self._infer(packed)
            unpacked = features.new_zeros((rows.numel(), 5, 32)).index_copy(1, columns, features)
            corrections = torch.stack([bridge(unpacked[:, role + 1])
                                       for role, bridge in enumerate(self.memory_bridges)], 1)
            evidence_out = evidence_out.index_copy(
                0, rows, safe_mask(safe_evidence[rows] + corrections, active[rows]))
        return local, evidence_out


def build_logic(method, latent_dim=256, num_heads=8, value_dim=64):
    if method != 'neural_logic_machine_evidence':
        raise ValueError(f'Unknown logic input method: {method}')
    with torch.random.fork_rng(devices=[]):
        return NeuralLogicInput(latent_dim, num_heads, value_dim)
