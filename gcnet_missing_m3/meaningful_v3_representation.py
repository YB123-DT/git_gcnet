"""Utterance-local feature transfers of GRANDE, MPS, NPU and density MERA.

Source contracts: docs/osram_survey80_20261004/representation.json. These
independent equation implementations do not import source classifiers/losses.
Five ordered roles are Local, Base, Gap1, Gap2, Gap3, never head tokens.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .meaningful_blocks_common import safe_mask
from .meaningful_input_new40 import zero_linear


METHODS = (
    'repr80_grande_nonoblivious_leaf_ensemble',
    'repr80_mps_feature_contraction',
    'repr80_neural_power_units',
    'repr80_mera_density_coarsegraining',
)


class Grande(nn.Module):
    """Non-oblivious hard trees, ST feature/split decisions, leaf-conditioned mix."""
    output_dim = 32

    def __init__(self, input_dim, estimators=8, depth=3):
        super().__init__()
        nodes, leaves = 2 ** depth - 1, 2 ** depth
        self.selectors = nn.Parameter(torch.randn(estimators, nodes, input_dim) * .1)
        self.thresholds = nn.Parameter(torch.randn(estimators, nodes, input_dim) * .2)
        self.leaves = nn.Parameter(torch.randn(estimators, leaves, self.output_dim) * .1)
        self.leaf_weights = nn.Parameter(torch.randn(estimators, leaves) * .1)
        paths, directions = [], []
        for leaf in range(leaves):
            node, path, bits = 0, [], []
            for level in range(depth):
                bit = (leaf >> (depth - level - 1)) & 1
                path.append(node)
                bits.append(bit)
                node = 2 * node + 1 + bit
            paths.append(path)
            directions.append(bits)
        self.register_buffer('paths', torch.tensor(paths))
        self.register_buffer('directions', torch.tensor(directions, dtype=torch.float32))

    def forward(self, roles, mask, availability):
        x = torch.cat((roles.flatten(1), mask[:, 1:].to(roles.dtype), availability), -1)
        soft = self.selectors.softmax(-1)
        hard = F.one_hot(soft.argmax(-1), soft.shape[-1]).to(soft.dtype)
        selector = soft + (hard - soft).detach()
        threshold = (selector * self.thresholds).sum(-1)
        selected = torch.einsum('nf,eif->nei', x, selector)
        probability = (F.softsign(threshold - selected) + 1) / 2
        decision = probability + (probability.round() - probability).detach()
        path_decisions = decision[:, :, self.paths]
        paths = ((1 - self.directions) * path_decisions
                 + self.directions * (1 - path_decisions)).prod(-1)
        weights = torch.einsum('nel,el->ne', paths, self.leaf_weights).softmax(-1)
        predictions = torch.einsum('nel,eld->ned', paths, self.leaves)
        return (weights[..., None] * predictions).sum(1)


class MPS(nn.Module):
    """Noncommuting full bond matrices contracted in fixed named-role order."""
    output_dim = 32

    def __init__(self, role_dim, bond=8):
        super().__init__()
        self.angle = nn.Parameter(torch.randn(5, role_dim) / math.sqrt(role_dim))
        self.cores = nn.Parameter(torch.randn(5, 2, bond, bond) * .08)
        with torch.no_grad():
            self.cores[:, 0].add_(torch.eye(bond))
        self.left = nn.Parameter(torch.randn(bond) / math.sqrt(bond))
        self.right = nn.Parameter(torch.randn(bond, self.output_dim) / math.sqrt(bond))
        self.availability = nn.Linear(3, self.output_dim, bias=False)

    def forward(self, roles, mask, availability):
        angle = (roles * self.angle).sum(-1).tanh() * (math.pi / 2)
        lift = torch.stack((angle.cos(), angle.sin()), -1)
        matrices = torch.einsum('nrs,rsij->nrij', lift, self.cores)
        eye = torch.eye(self.left.numel(), device=roles.device, dtype=roles.dtype)
        matrices = torch.where(mask[..., None, None], matrices, eye)
        state = self.left.expand(roles.shape[0], -1)
        for role in range(5):
            state = torch.einsum('ni,nij->nj', state, matrices[:, role])
        return state @ self.right + self.availability(availability)


class NPU(nn.Module):
    """Gated complex power unit: magnitude logarithm AND signed phase path."""
    output_dim = 32

    def __init__(self, role_dim):
        super().__init__()
        dimensions = 5 * role_dim + 3
        self.gate = nn.Parameter(torch.full((dimensions,), .5))
        self.real = nn.Parameter(torch.empty(self.output_dim, dimensions))
        nn.init.xavier_uniform_(self.real)
        self.imag = nn.Parameter(torch.zeros(self.output_dim, dimensions))

    def forward(self, roles, mask, availability):
        # The absent factor is one, not zero: log magnitude and phase vanish.
        factors = torch.where(mask[..., None], roles.tanh(), torch.ones_like(roles))
        x = torch.cat((factors.flatten(1), 1 + availability), -1).float()
        gate = self.gate.clamp(0, 1)
        radius = gate * (x.abs() + 1e-7) + (1 - gate)
        # Exactly neutral absent factors, including the epsilon correction.
        present = torch.cat((mask[..., None].expand_as(roles).flatten(1),
                             torch.ones_like(availability, dtype=torch.bool)), -1)
        radius = torch.where(present, radius, torch.ones_like(radius))
        phase = math.pi * gate * (x < 0).to(x.dtype)
        log_radius = radius.log()
        return (F.linear(log_radius, self.real) - F.linear(phase, self.imag)).exp() * (
            F.linear(log_radius, self.imag) + F.linear(phase, self.real)).cos()


def _apply_gate(state, gate, targets):
    """Apply a two-site gate on any pair of qubit tensor axes (batch excluded)."""
    axes = [target + 1 for target in targets]
    remaining = [axis for axis in range(1, state.ndim) if axis not in axes]
    order = [0, *axes, *remaining]
    inverse = [order.index(axis) for axis in range(state.ndim)]
    moved = state.permute(order)
    result = torch.einsum('ij,njk->nik', gate, moved.reshape(state.shape[0], 4, -1))
    return result.reshape(moved.shape).permute(inverse)


def _trace_pure(state, sites):
    kept, discarded = list(range(0, sites, 2)), list(range(1, sites, 2))
    psi = state.permute([0, *[i + 1 for i in kept + discarded]])
    psi = psi.reshape(state.shape[0], 2 ** len(kept), 2 ** len(discarded))
    density = torch.einsum('nik,njk->nij', psi, psi.conj())
    return density.reshape([state.shape[0]] + [2] * (2 * len(kept)))


def _trace_density(state, sites):
    kept, discarded = list(range(0, sites, 2)), list(range(1, sites, 2))
    axes = kept + [i + sites for i in kept] + discarded + [i + sites for i in discarded]
    dim = 2 ** len(kept)
    reordered = state.permute([0, *[i + 1 for i in axes]]).reshape(-1, dim, dim, dim, dim)
    return reordered.diagonal(dim1=-2, dim2=-1).sum(-1).reshape(
        [state.shape[0]] + [2] * (2 * len(kept)))


class MERA(nn.Module):
    """Exact eight-site density MERA, p=0: boundary disentanglers + partial trace.

    First layer remains a pure global state until pooling, avoiding an N*256^2
    density allocation. Subsequent mixed states retain all off-diagonal terms.
    """
    output_dim = 4

    def __init__(self, role_dim):
        super().__init__()
        self.angle = nn.Parameter(torch.randn(5, role_dim) / math.sqrt(role_dim))
        self.hermitian_real = nn.Parameter(torch.randn(11, 4, 4) * .1)
        self.hermitian_imag = nn.Parameter(torch.randn(11, 4, 4) * .1)

    def unitaries(self):
        real, imag = self.hermitian_real.float(), self.hermitian_imag.float()
        h = torch.complex((real + real.transpose(-1, -2)) / 2,
                          (imag - imag.transpose(-1, -2)) / 2)
        return torch.matrix_exp(1j * h)

    def density(self, roles, mask, availability):
        angles = (roles.float() * self.angle.float()).sum(-1).tanh() * (math.pi / 2)
        angles = torch.where(mask, angles, torch.zeros_like(angles))
        angles = torch.cat((angles, availability.float() * (math.pi / 2)), -1)
        sites = torch.stack((angles.cos(), angles.sin()), -1).to(torch.complex64)
        state = sites[:, 0]
        for site in range(1, 8):
            state = (state.reshape(state.shape[0], -1, 1) * sites[:, site, None]).flatten(1)
        state = state.reshape([roles.shape[0]] + [2] * 8)
        gates = self.unitaries()
        index = 0
        for pair in ((1, 2), (3, 4), (5, 6), (0, 1), (2, 3), (4, 5), (6, 7)):
            state = _apply_gate(state, gates[index], pair)
            index += 1
        state = _trace_pure(state, 8)
        for count, pairs in ((4, ((1, 2), (0, 1), (2, 3))), (2, ((0, 1),))):
            for pair in pairs:
                state = _apply_gate(state, gates[index], pair)
                state = _apply_gate(state, gates[index].conj(), tuple(i + count for i in pair))
                index += 1
            state = _trace_density(state, count)
        return state.reshape(-1, 2, 2)

    def forward(self, roles, mask, availability):
        with torch.autocast(device_type=roles.device.type, enabled=False):
            rho = self.density(roles, mask, availability)
        return torch.stack((rho[:, 0, 0].real, rho[:, 1, 1].real,
                            rho[:, 0, 1].real, rho[:, 0, 1].imag), -1)


class RepresentationAdapter(nn.Module):
    def __init__(self, method, latent_dim, num_heads, value_dim):
        super().__init__()
        width, dim = num_heads * value_dim, 16
        self.local_projection = nn.Linear(latent_dim, dim)
        self.role_projections = nn.ModuleList([nn.Linear(width, dim) for _ in range(4)])
        factories = {METHODS[0]: lambda: Grande(5 * dim + 7),
                     METHODS[1]: lambda: MPS(dim), METHODS[2]: lambda: NPU(dim),
                     METHODS[3]: lambda: MERA(dim)}
        if method not in factories:
            raise ValueError('Unknown representation method: ' + method)
        self.core = factories[method]()
        self.local_decoder = zero_linear(self.core.output_dim, latent_dim)
        self.evidence_decoders = nn.ModuleList([zero_linear(self.core.output_dim, width)
                                              for _ in range(4)])

    def forward(self, local, evidence, active, availability):
        active = active.bool()
        valid = active.any(-1)
        clean = safe_mask(evidence, active)
        av = safe_mask(availability.to(local.dtype), valid)
        mask = torch.cat((valid[:, None], active), -1)
        roles = torch.stack([self.local_projection(safe_mask(local, valid)),
                             *[layer(clean[:, role]) for role, layer in
                               enumerate(self.role_projections)]], 1)
        roles = safe_mask(roles, mask)
        features = self.core(roles, mask, av).to(local.dtype)
        delta_local = safe_mask(self.local_decoder(features), valid)
        delta_evidence = safe_mask(torch.stack([layer(features) for layer in
                                               self.evidence_decoders], 1), active)
        return local + delta_local.to(local.dtype), clean + delta_evidence.to(evidence.dtype)


def build(method, latent_dim=256, num_heads=8, value_dim=64):
    return RepresentationAdapter(method, latent_dim, num_heads, value_dim)
