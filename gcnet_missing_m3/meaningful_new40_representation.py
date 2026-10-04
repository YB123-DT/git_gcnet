"""Equation-based representation branches; no upstream implementation copied.

These are feature-domain transfers of the five representation cards, not
reproductions of their classifiers, objectives, or theoretical guarantees.
All state is utterance-local. Fixed ordered projections retain role/head labels.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .meaningful_blocks_common import safe_mask
from .meaningful_input_new40 import zero_linear


class _DempsterShafer(nn.Module):
    """Discounted prototype masses with the corrected Omega intersection."""
    output_dim = 10

    def __init__(self, dim=32, prototypes=3, classes=8):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(5, prototypes, dim) * 0.2)
        self.membership = nn.Parameter(torch.randn(5, prototypes, classes) * 0.2)
        self.discount = nn.Parameter(torch.zeros(5, prototypes))
        self.bandwidth = nn.Parameter(torch.full((5, prototypes), -2.0))

    @staticmethod
    def combine(a, b):
        singleton = (a[..., :-1] * b[..., :-1]
                     + a[..., :-1] * b[..., -1:]
                     + a[..., -1:] * b[..., :-1])
        omega = a[..., -1:] * b[..., -1:]
        numerator = torch.cat((singleton, omega), -1)
        normalizer = numerator.sum(-1, keepdim=True)
        return numerator / normalizer, normalizer

    def forward(self, roles, mask, coordinates):
        # Vacuous roles are never evaluated against prototype centers.
        mass = roles.new_zeros(roles.shape[0], 9)
        mass[:, -1] = 1
        survival = roles.new_ones(roles.shape[0], 1)
        for role in range(5):
            rows = mask[:, role].nonzero(as_tuple=True)[0]
            if not rows.numel():
                continue
            distances = (roles[rows, role, None] - self.centers[role]).square().sum(-1)
            support = (0.95 * self.discount[role].sigmoid()
                       * torch.exp(-F.softplus(self.bandwidth[role]) * distances))
            singletons = support[..., None] * self.membership[role].softmax(-1)
            for prototype in range(self.centers.shape[1]):
                current = torch.cat((singletons[:, prototype],
                                     1 - support[:, prototype, None]), -1)
                fused, normalizer = self.combine(mass[rows], current)
                mass = mass.index_copy(0, rows, fused)
                survival = survival.index_copy(0, rows, survival[rows] * normalizer)
        # Product of step normalizers is the original conjunctive nonconflict
        # mass. This retains accumulated conflict, not only the final step.
        return torch.cat((mass, 1 - survival), -1)


class _BcosLayer(nn.Module):
    def __init__(self, input_dim, output_dim, exponent=2):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(output_dim, 2, input_dim))
        nn.init.normal_(self.weight, std=input_dim ** -0.5)
        self.exponent = exponent

    def forward(self, x):
        weights = F.normalize(self.weight, dim=-1, eps=1e-8)
        alignment = torch.einsum('ni,oki->nok', x, weights)
        cosine = alignment / x.square().sum(-1, keepdim=True).add(1e-8).sqrt()[..., None]
        transformed = alignment * cosine.abs().pow(self.exponent - 1)
        return transformed.max(-1).values


class _BcosNetwork(nn.Module):
    output_dim = 64

    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.ModuleList([_BcosLayer(input_dim, 96),
                                     _BcosLayer(96, 96), _BcosLayer(96, 64)])

    def forward(self, roles, mask, coordinates):
        x = coordinates
        for layer in self.layers:
            x = layer(x)
        return x


class _GaborFilter(nn.Module):
    def __init__(self, dim, width):
        super().__init__()
        self.frequency = nn.Linear(dim, width)
        nn.init.normal_(self.frequency.weight, std=1.0)
        nn.init.uniform_(self.frequency.bias, -math.pi, math.pi)
        self.center = nn.Parameter(torch.randn(width, dim) / math.sqrt(dim))
        self.raw_gamma = nn.Parameter(torch.full((width,), -0.5))

    def forward(self, x):
        distance = (x[:, None] - self.center).square().sum(-1)
        return self.frequency(x).sin() * torch.exp(-0.5 * F.softplus(self.raw_gamma) * distance)


class _MultiplicativeFilterNetwork(nn.Module):
    output_dim = 64

    def __init__(self, input_dim):
        super().__init__()
        self.coordinates = nn.Linear(input_dim, 32)
        self.filters = nn.ModuleList([_GaborFilter(32, 64) for _ in range(4)])
        self.mixing = nn.ModuleList([nn.Linear(64, 64) for _ in range(3)])

    def forward(self, roles, mask, coordinates):
        x = self.coordinates(coordinates).tanh() / math.sqrt(32)
        state = self.filters[0](x)
        for mixing, filtering in zip(self.mixing, self.filters[1:]):
            state = filtering(x) * mixing(state)
        return state


class _DistanceLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(output_dim, input_dim) * 0.2)
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        return (x[:, None] - self.centers).abs().amax(-1) + self.bias


class _DistanceNetwork(nn.Module):
    output_dim = 64

    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.ModuleList([_DistanceLayer(input_dim, 96),
                                     _DistanceLayer(96, 96), _DistanceLayer(96, 64)])

    def forward(self, roles, mask, coordinates):
        state = coordinates
        for layer in self.layers:
            state = layer(state)
        return state


class _ListaCPSS(nn.Module):
    output_dim = 64

    def __init__(self, input_dim):
        super().__init__()
        self.measurement = nn.Linear(input_dim, 32)
        generator = torch.Generator().manual_seed(1729)
        dictionary = F.normalize(torch.randn(32, 64, generator=generator), dim=0)
        self.register_buffer('dictionary', dictionary)
        step = dictionary.square().sum().reciprocal()
        # Coupling is exact at every iteration: z + W_k(y - A z).
        self.encoders = nn.ParameterList([nn.Parameter(step * dictionary.T.clone())
                                         for _ in range(5)])
        self.thresholds = nn.Parameter(torch.full((5,), -7.0))
        self.support_counts = (0, 4, 8, 12, 16)

    @staticmethod
    def shrink_with_support(value, threshold, count):
        shrunk = value.sign() * (value.abs() - threshold).clamp_min(0)
        if count:
            indices = value.abs().argsort(dim=-1, descending=True, stable=True)[:, :count]
            support = torch.zeros_like(value, dtype=torch.bool).scatter(1, indices, True)
            shrunk = torch.where(support, value, shrunk)
        return shrunk

    def forward(self, roles, mask, coordinates):
        y = self.measurement(coordinates)
        state = y.new_zeros(y.shape[0], 64)
        for index, encoder in enumerate(self.encoders):
            residual = y - F.linear(state, self.dictionary)
            candidate = state + F.linear(residual, encoder)
            state = self.shrink_with_support(candidate, F.softplus(self.thresholds[index]),
                                             self.support_counts[index])
        return state


class _RepresentationAdapter(nn.Module):
    def __init__(self, method, latent_dim, num_heads, value_dim):
        super().__init__()
        width = num_heads * value_dim
        self.local_projection = nn.Linear(latent_dim + 3, 32)
        # Fixed ordering inside each projection preserves the true head labels.
        self.role_projections = nn.ModuleList([nn.Linear(width, 32) for _ in range(4)])
        cores = {
            'repr_dst_corrected_evidence_combination': lambda: _DempsterShafer(),
            'repr_bcos_alignment_network': lambda: _BcosNetwork(167),
            'repr_mfn_gabor_filter_chain': lambda: _MultiplicativeFilterNetwork(167),
            'repr_linf_distance_network': lambda: _DistanceNetwork(167),
            'repr_lista_cpss_sparse_pursuit': lambda: _ListaCPSS(167),
        }
        if method not in cores:
            raise ValueError('Unknown representation method: ' + method)
        self.core = cores[method]()
        self.local_decoder = zero_linear(self.core.output_dim, latent_dim)
        self.evidence_decoders = nn.ModuleList([zero_linear(self.core.output_dim, width)
                                              for _ in range(4)])

    def forward(self, local, evidence, active, availability):
        active = active.bool()
        valid = active.any(-1)
        clean = safe_mask(evidence, active)
        av = safe_mask(availability.to(local.dtype), valid)
        projected_local = self.local_projection(torch.cat((safe_mask(local, valid), av), -1))
        projected_roles = [projection(clean[:, role])
                           for role, projection in enumerate(self.role_projections)]
        mask = torch.cat((valid[:, None], active), -1)
        roles = safe_mask(torch.stack([projected_local, *projected_roles], 1), mask)
        coordinates = torch.cat((roles.flatten(1), active.to(local.dtype), av), -1)
        features = local.new_zeros(local.shape[0], self.core.output_dim)
        rows = valid.nonzero(as_tuple=True)[0]
        if rows.numel():
            features = features.index_copy(0, rows, self.core(roles[rows], mask[rows], coordinates[rows]).to(features.dtype))
        delta_local = safe_mask(self.local_decoder(features), valid)
        delta_evidence = safe_mask(torch.stack([decoder(features)
                                               for decoder in self.evidence_decoders], 1), active)
        return local + delta_local.to(local.dtype), clean + delta_evidence.to(evidence.dtype)


def build(method, latent_dim=256, num_heads=8, value_dim=64):
    return _RepresentationAdapter(method, latent_dim, num_heads, value_dim)
