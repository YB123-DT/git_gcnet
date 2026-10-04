"""SPDNet congruence/rectification hierarchy for identity-start Flat inputs.

Independent implementation of Huang/Van Gool, arXiv:1608.04233; inspected
author source pin527110a2209e918785075738b81c4ab091664e61. This transfers
BiMap/ReEig/BiMap/ReEig/BiMap/LogEig, not the source classifier or optimizer.
Covariance of typed read descriptors and exp-skew orthogonal parametrization
are declared adaptations. No source MATLAB/Manopt code or dependencies used.

The custom spectral VJP uses continuous Loewner divided differences rather
than differentiating eigenvectors. First-order gradients are supported; this
module deliberately does not promise higher-order spectral derivatives.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.autograd.function import once_differentiable

from .meaningful_blocks_common import active_groups, safe_mask


SPECTRAL_METHODS = ('spdnet_bimap_reeig_logeig',)
_DESCRIPTOR_DIM = 64
_RIDGE = 1e-4
_RECTIFY_FLOORS = (1e-3, 1e-2)


def _symmetric(matrix):
    return (matrix + matrix.transpose(-2, -1)) * 0.5


class _SymmetricSpectral(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matrix, logarithm, floor):
        eigenvalues, eigenvectors = torch.linalg.eigh(_symmetric(matrix))
        if logarithm:
            if bool((eigenvalues <= 0).any()):
                raise ValueError('LogEig requires strictly positive eigenvalues')
            transformed = eigenvalues.log()
        else:
            transformed = eigenvalues.clamp_min(floor)
        ctx.save_for_backward(eigenvalues, eigenvectors, transformed)
        ctx.logarithm, ctx.floor = logarithm, floor
        return _symmetric((eigenvectors * transformed.unsqueeze(-2))
                          @ eigenvectors.transpose(-2, -1))

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        values, vectors, transformed = ctx.saved_tensors
        left, right = values.unsqueeze(-1), values.unsqueeze(-2)
        gap = left - right
        if ctx.logarithm:
            # For nearly equal eigenvalues, the symmetric midpoint derivative
            # has relative error O((gap/lambda)^2), below machine epsilon here.
            close = gap.abs() <= torch.finfo(values.dtype).eps ** 0.5 * torch.maximum(
                left.abs(), right.abs())
            denominator = torch.where(close, torch.ones_like(gap), gap)
            quotient = (transformed.unsqueeze(-1) - transformed.unsqueeze(-2)) / denominator
            loewner = torch.where(close, 2.0 / (left + right), quotient)
        else:
            # Exact piecewise divided differences avoid cancellation in each
            # linear branch. At a repeated eigenvalue on the floor choose zero.
            above_left, above_right = left > ctx.floor, right > ctx.floor
            mixed = above_left != above_right
            denominator = torch.where(mixed, gap, torch.ones_like(gap))
            quotient = (transformed.unsqueeze(-1) - transformed.unsqueeze(-2)) / denominator
            loewner = torch.where(mixed, quotient, (above_left & above_right).to(values.dtype))
        rotated = vectors.transpose(-2, -1) @ _symmetric(grad_output) @ vectors
        grad_matrix = vectors @ (loewner * rotated) @ vectors.transpose(-2, -1)
        return _symmetric(grad_matrix), None, None


def spectral_rectify(matrix, floor=1e-3):
    """Symmetric eigenvalue floor with zero subgradient exactly at the floor."""
    if floor <= 0:
        raise ValueError('ReEig floor must be positive')
    return _SymmetricSpectral.apply(matrix, False, float(floor))


def spectral_log(matrix):
    """SPD matrix logarithm with a basis-invariant repeated-spectrum VJP."""
    return _SymmetricSpectral.apply(matrix, True, 0.0)


class OrthogonalBiMap(nn.Module):
    """W=exp(A-A.T)Q[:, :out]; ordinary Adam cannot violate W.T W=I."""

    def __init__(self, input_dim, output_dim, seed):
        super().__init__()
        if not 0 < output_dim < input_dim:
            raise ValueError('BiMap must reduce a positive SPD dimension')
        self.skew_parameter = nn.Parameter(torch.zeros(input_dim, input_dim))
        self.output_dim = output_dim
        generator = torch.Generator(device='cpu').manual_seed(seed)
        # Fixed independent bases, not differentiable QR/retractions in forward.
        basis = torch.linalg.qr(torch.randn(input_dim, input_dim, dtype=torch.float64,
                                           generator=generator)).Q
        self.register_buffer('initial_basis', basis)

    def weight(self):
        skew = self.skew_parameter - self.skew_parameter.T
        basis = self.initial_basis[:, :self.output_dim].to(dtype=skew.dtype)
        return torch.matrix_exp(skew) @ basis

    def forward(self, matrix):
        weight = self.weight()
        return _symmetric(weight.T @ matrix @ weight)


class SPDNetInput(nn.Module):
    def __init__(self, latent_dim, num_heads, value_dim):
        super().__init__()
        if any(not isinstance(x, int) or isinstance(x, bool) or x <= 0
               for x in (latent_dim, num_heads, value_dim)):
            raise ValueError('Expected positive integer Local/head dimensions')
        self.latent_dim, self.num_heads, self.value_dim = latent_dim, num_heads, value_dim
        self.local_project = nn.Linear(latent_dim, _DESCRIPTOR_DIM)
        self.memory_project = nn.Linear(value_dim, _DESCRIPTOR_DIM)
        self.role_embedding = nn.Embedding(5, _DESCRIPTOR_DIM)
        self.head_embedding = nn.Embedding(num_heads, _DESCRIPTOR_DIM)
        seed = torch.initial_seed()
        widths = (64, 32, 16, 8)
        self.bimaps = nn.ModuleList([
            OrthogonalBiMap(left, right, (seed + 811 + stage) % (2**63 - 1))
            for stage, (left, right) in enumerate(zip(widths[:-1], widths[1:]))
        ])
        self.local_bridge = nn.Linear(64, latent_dim)
        # One affine with independent slices is exactly one output map per
        # role/head. No tied global correction is broadcast to all head slots.
        self.memory_bridge = nn.Linear(64, 4 * num_heads * value_dim)
        for bridge in (self.local_bridge, self.memory_bridge):
            nn.init.zeros_(bridge.weight)
            nn.init.zeros_(bridge.bias)

    def _tokens(self, local, evidence, active):
        batch = local.shape[0]
        values = evidence.reshape(batch, 4, self.num_heads, self.value_dim)
        memory = (self.memory_project(values)
                  + self.role_embedding.weight[None, 1:, None]
                  + self.head_embedding.weight[None, None])
        local_token = self.local_project(local) + self.role_embedding.weight[0]
        tokens = torch.cat((local_token[:, None], memory.flatten(1, 2)), 1)
        mask = torch.cat((active.any(-1)[:, None],
                          active[..., None].expand(-1, -1, self.num_heads).flatten(1)), 1)
        return safe_mask(tokens, mask), mask

    def _representation(self, descriptors):
        centered = descriptors - descriptors.mean(1, keepdim=True)
        covariance = centered.transpose(-2, -1) @ centered / descriptors.shape[1]
        covariance = covariance + _RIDGE * torch.eye(
            _DESCRIPTOR_DIM, dtype=covariance.dtype, device=covariance.device)
        first = spectral_rectify(self.bimaps[0](covariance), _RECTIFY_FLOORS[0])
        second = spectral_rectify(self.bimaps[1](first), _RECTIFY_FLOORS[1])
        last = self.bimaps[2](second)
        return spectral_log(last).flatten(1)

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
        if not bool(active.any()):
            return local_out, evidence_out
        # Spectral operations require float32 or float64. Retain double for
        # derivative checks, and do not inherit an outer mixed-precision context.
        dtype = torch.float64 if local.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=local.device.type, enabled=False):
            tokens, mask = self._tokens(safe_local.to(dtype), safe_evidence.to(dtype), active)
            for rows, columns, packed in active_groups(tokens, mask):
                if columns.numel() <= 1:
                    continue
                representation = self._representation(packed)
                local_change = self.local_bridge(representation).to(local.dtype)
                memory_change = self.memory_bridge(representation).reshape(
                    -1, 4, self.num_heads * self.value_dim).to(evidence.dtype)
                local_out = local_out.index_copy(0, rows, safe_local[rows] + local_change)
                evidence_out = evidence_out.index_copy(
                    0, rows, safe_mask(safe_evidence[rows] + memory_change, active[rows]))
        return local_out, evidence_out


def build_spectral(method, latent_dim=256, num_heads=8, value_dim=64):
    if method != 'spdnet_bimap_reeig_logeig':
        raise ValueError(f'Unknown spectral input method: {method}')
    with torch.random.fork_rng(devices=[]):
        return SPDNetInput(latent_dim, num_heads, value_dim)
