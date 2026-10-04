"""Equation-derived representation ports, without their source training objectives.

SOS integrates paired Gaussian leaves before contracting signed circuits. GrNet
uses paper A-ProjPooling, not the MATLAB spatial pooling variant. NKN retains
two positive-sum/Schur-product stages. CPFlow evaluates the exact gradient of
a strongly convex ICNN potential analytically (including mixed task derivatives).
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .meaningful_blocks_common import safe_mask
from .meaningful_input_new40 import TokenAdapter, zero_linear


class _FixedAdapter(nn.Module):
    def __init__(self, latent_dim, num_heads, value_dim, feature_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.memory_dim = num_heads * value_dim
        self.local_decoder = zero_linear(feature_dim, latent_dim)
        self.memory_decoder = zero_linear(feature_dim, 4 * self.memory_dim)

    def forward(self, local, evidence, active, availability):
        active = active.bool()
        valid = active.any(-1)
        clean_local = safe_mask(local, valid)
        clean_evidence = safe_mask(evidence, active)
        features = self.features(clean_local, clean_evidence, active, availability)
        dl = safe_mask(self.local_decoder(features), valid)
        de = self.memory_decoder(features).reshape_as(evidence)
        return local + dl, safe_mask(clean_evidence + de, active)

    @staticmethod
    def pack(local, evidence, active, availability):
        return torch.cat((local, evidence.flatten(1), active.to(local.dtype),
                          availability.to(local.dtype)), -1)


class SignedSOS(_FixedAdapter):
    """Four density channels, each a sum of three compatible signed squares.

    Each amplitude has six decomposable products over five scalar coordinates.
    A fixed ((role0,role1),(role2,role3),local) tree contracts pair states.
    Gaussian leaves are normalized densities; all arithmetic in the contraction
    is double precision. The final tiny positive score floor is numerical only.
    """
    def __init__(self, latent_dim, num_heads, value_dim):
        super().__init__(latent_dim, num_heads, value_dim, 4)
        self.role_coordinates = nn.ModuleList([
            nn.Linear(self.memory_dim, 1) for _ in range(4)])
        self.local_coordinate = nn.Linear(latent_dim, 1)
        self.means = nn.Parameter(torch.linspace(-1.5, 1.5, 6).repeat(5, 1))
        self.raw_scales = nn.Parameter(torch.zeros(5, 6))
        self.signed_weights = nn.Parameter(torch.randn(4, 3, 6) / math.sqrt(6))

    def leaf_pairs(self, x, observed):
        means = self.means.double()
        scales = F.softplus(self.raw_scales.double()) + 0.2
        variance = scales.square()
        phi = torch.exp(-0.5 * ((x.double()[..., None] - means) / scales).square())
        phi = phi / (math.sqrt(2 * math.pi) * scales)
        point = phi[..., :, None] * phi[..., None, :]
        total_variance = variance[..., :, None] + variance[..., None, :]
        overlap = torch.exp(-0.5 * (means[..., :, None] - means[..., None, :]).square()
                            / total_variance) / (2 * math.pi * total_variance).sqrt()
        return torch.where(observed[..., None, None], point, overlap), overlap

    @staticmethod
    def tree_contract(pair_leaves):
        left = pair_leaves[..., 0, :, :] * pair_leaves[..., 1, :, :]
        right = pair_leaves[..., 2, :, :] * pair_leaves[..., 3, :, :]
        return left * right * pair_leaves[..., 4, :, :]

    def log_density(self, x, observed):
        pairs, overlap = self.leaf_pairs(x, observed)
        w = self.signed_weights.double()
        score = torch.einsum('csi,nij,csj->nc', w, self.tree_contract(pairs), w)
        partition = torch.einsum('csi,ij,csj->c', w, self.tree_contract(overlap), w)
        return (score.clamp_min(1e-30).log() - partition.clamp_min(1e-30).log()).to(x.dtype)

    def features(self, local, evidence, active, availability):
        x = torch.cat([layer(evidence[:, r]) for r, layer in enumerate(self.role_coordinates)]
                      + [self.local_coordinate(local)], -1).tanh()
        observed = torch.cat((active, torch.ones_like(active[:, :1])), -1)
        return self.log_density(x, observed)


class _TopProjector(torch.autograd.Function):
    """Exact spectral-projector derivative without within-eigenspace divisions.

    Only the selected/unselected eigengap matters. At a vanishing boundary gap,
    the denominator floor explicitly regularizes the otherwise undefined map.
    """
    @staticmethod
    def forward(ctx, matrix, rank):
        values, vectors = torch.linalg.eigh(matrix)
        selected = vectors[..., -rank:]
        ctx.save_for_backward(values, vectors)
        ctx.rank = rank
        return selected @ selected.transpose(-1, -2)

    @staticmethod
    def backward(ctx, grad):
        values, vectors = ctx.saved_tensors
        indicator = torch.zeros_like(values)
        indicator[..., -ctx.rank:] = 1
        delta = values[..., :, None] - values[..., None, :]
        diff = indicator[..., :, None] - indicator[..., None, :]
        denom = delta.abs().clamp_min(1e-5) * torch.where(delta < 0, -1., 1.)
        divided = diff / denom
        sym = (grad + grad.transpose(-1, -2)) * 0.5
        inner = vectors.transpose(-1, -2) @ sym @ vectors
        return vectors @ (divided * inner) @ vectors.transpose(-1, -2), None


class GrassmannPooling(_FixedAdapter):
    def __init__(self, latent_dim, num_heads, value_dim):
        self.ambient, self.rank = 8, 2
        super().__init__(latent_dim, num_heads, value_dim, 64)
        self.lifts = nn.ModuleList([nn.Linear(self.memory_dim + latent_dim, 12)
                                   for _ in range(4)])
        self.frmaps = nn.Parameter(torch.eye(8).repeat(4, 1, 1) + 0.03 * torch.randn(4, 8, 8))
        self.final_map = nn.Parameter(torch.eye(8) + 0.03 * torch.randn(8, 8))
        self.register_buffer('anchor', torch.eye(2))

    def features(self, local, evidence, active, availability):
        pooled = local.new_zeros(local.shape[0], 8, 8)
        for role, lift in enumerate(self.lifts):
            rows = active[:, role].nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            bottom = lift(torch.cat((local[rows], evidence[rows, role]), -1)).reshape(-1, 6, 2)
            basis = torch.cat((self.anchor.expand(rows.numel(), -1, -1), bottom), 1)
            basis = self.frmaps[role] @ basis
            q = torch.linalg.qr(basis, mode='reduced').Q
            pooled[rows] = pooled[rows] + q @ q.transpose(-1, -2)
        pooled = pooled / active.sum(-1).clamp_min(1)[:, None, None]
        projection = _TopProjector.apply(pooled, self.rank)
        # W P W^T spans W Q. Its rank-q projector is exactly the projector of
        # FRMap(W,Q) followed by ReOrth, without differentiating a basis gauge.
        mapped = self.final_map @ projection @ self.final_map.T
        result = _TopProjector.apply(mapped, self.rank)
        return safe_mask(result.flatten(1), active.any(-1))


class NeuralKernelComposition(nn.Module):
    """Packed active tokens: each unit remains PSD by kernel closure."""
    def __init__(self, dim=64):
        super().__init__()
        self.raw_length = nn.Parameter(torch.zeros(2))
        self.raw_alpha = nn.Parameter(torch.zeros(()))
        self.sum_weights = nn.ParameterList([nn.Parameter(torch.full((8, 3), -1.)),
                                             nn.Parameter(torch.full((8, 4), -1.))])
        self.sum_biases = nn.ParameterList([nn.Parameter(torch.full((8,), -3.)),
                                            nn.Parameter(torch.full((8,), -3.))])
        self.readout = nn.Linear(dim * 4, dim)

    def kernels(self, x):
        length = F.softplus(self.raw_length) + 0.1
        alpha = F.softplus(self.raw_alpha) + 0.1
        distance = (x[:, :, None] - x[:, None, :]).square().mean(-1)
        dot = x @ x.transpose(-1, -2) / x.shape[-1]
        k = torch.stack((torch.exp(-distance / (2 * length[0].square())),
                         (1 + distance / (2 * alpha * length[1].square())).pow(-alpha),
                         dot), -1)
        for weight, bias in zip(self.sum_weights, self.sum_biases):
            k = F.linear(k, F.softplus(weight) / weight.shape[1], F.softplus(bias))
            k = k.reshape(*k.shape[:-1], 4, 2).prod(-1)
        return k

    def forward(self, tokens, columns, num_heads):
        k = self.kernels(tokens)
        summaries = torch.einsum('nijc,njd->nicd', k, tokens) / tokens.shape[1]
        return self.readout(summaries.flatten(2))


class ConvexPotentialFlow(_FixedAdapter):
    """Full gradient map, computed by the analytic reverse chain rule.

    This is equivalent to autograd.grad(phi.sum(), x, create_graph=True), but
    also works inside no_grad/inference_mode. Differentiating this forward
    computes the mixed second derivatives required by task training.
    """
    def __init__(self, latent_dim, num_heads, value_dim):
        super().__init__(latent_dim, num_heads, value_dim, 16)
        self.input_projection = nn.Linear(latent_dim + 4 * self.memory_dim + 7, 16)
        self.input_skips = nn.ModuleList([nn.Linear(16, 24) for _ in range(3)])
        self.positive_weights = nn.ParameterList([nn.Parameter(torch.full((24, 24), -2.))
                                                  for _ in range(2)])
        self.positive_output = nn.Parameter(torch.full((24,), -2.))
        self.affine_output = nn.Parameter(torch.zeros(16))
        self.raw_alpha = nn.Parameter(torch.zeros(()))
        self.raw_beta = nn.Parameter(torch.zeros(()))

    def potential_and_gradient(self, x):
        pre = [self.input_skips[0](x)]
        z = F.softplus(pre[0])
        weights = [F.softplus(w) / w.shape[1] for w in self.positive_weights]
        for weight, skip in zip(weights, self.input_skips[1:]):
            pre.append(F.linear(z, weight) + skip(x))
            z = F.softplus(pre[-1])
        output = F.softplus(self.positive_output) / self.positive_output.numel()
        alpha, beta = F.softplus(self.raw_alpha) + 0.1, F.softplus(self.raw_beta)
        potential = alpha * x.square().sum(-1) / 2 + beta * (z @ output + x @ self.affine_output)
        adjoint = output.expand_as(z)
        gradient = self.affine_output.expand_as(x)
        for stage in range(2, -1, -1):
            adjoint = adjoint * pre[stage].sigmoid()
            gradient = gradient + adjoint @ self.input_skips[stage].weight
            if stage:
                adjoint = adjoint @ weights[stage - 1]
        return potential, alpha * x + beta * gradient

    def features(self, local, evidence, active, availability):
        x = self.input_projection(self.pack(local, evidence, active, availability))
        return self.potential_and_gradient(x)[1]


def build(method, latent_dim=256, num_heads=8, value_dim=64):
    if method == 'repr_neural_kernel_composition':
        return TokenAdapter(NeuralKernelComposition(), latent_dim, num_heads, value_dim)
    classes = {
        'repr_sos_signed_probability_circuit': SignedSOS,
        'repr_grassmann_projection_pooling': GrassmannPooling,
        'repr_convex_potential_gradient_flow': ConvexPotentialFlow,
    }
    if method not in classes:
        raise ValueError('Unknown representation-extra method: ' + method)
    return classes[method](latent_dim, num_heads, value_dim)
