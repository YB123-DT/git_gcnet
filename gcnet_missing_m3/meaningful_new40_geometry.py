"""Determinantal, curved, spectral, scattering and bounded evidence operators."""
import math
import warnings

import torch
from torch import nn
from torch.nn import functional as F

from .meaningful_blocks_common import HeadTokenizer, safe_mask
from .meaningful_input_new40 import zero_linear


class _Residual(nn.Module):
    def __init__(self, latent_dim, num_heads, value_dim):
        super().__init__()
        self.latent_dim, self.num_heads, self.value_dim = latent_dim, num_heads, value_dim
        self.output_dim = latent_dim + 4 * num_heads * value_dim

    def forward(self, local, evidence, active, availability):
        active = active.bool()
        clean = safe_mask(evidence, active)
        if not local.shape[0]:
            return local, clean
        valid = active.any(-1)
        if not valid.any():
            return local, clean
        rows = valid.nonzero(as_tuple=True)[0]
        dtype = torch.float64 if local.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=local.device.type, enabled=False):
            x, e = local[rows].to(dtype), clean[rows].to(dtype)
            av = availability[rows].to(dtype)
            if not all(torch.isfinite(t).all() for t in (x, e, av)):
                raise ValueError('Nonfinite active geometry input')
            dl, de = self.correction(x, e, active[rows], av)
            if not torch.isfinite(dl).all() or not torch.isfinite(de).all():
                raise FloatingPointError('Nonfinite geometry correction')
        delta_local, delta_evidence = torch.zeros_like(local), torch.zeros_like(clean)
        delta_local[rows] = dl.to(local.dtype)
        delta_evidence[rows] = safe_mask(de.to(evidence.dtype), active[rows])
        return local + delta_local, safe_mask(clean + delta_evidence, active)

    def split(self, delta):
        return delta[:, :self.latent_dim], delta[:, self.latent_dim:].reshape(
            -1, 4, self.num_heads * self.value_dim)

    def make_tokenizer(self):
        return HeadTokenizer(self.latent_dim, self.num_heads, self.value_dim,
                             dim=32, shared_projection=True, normalize=False)


class DeterminantalReadout(_Residual):
    def __init__(self, *dims):
        super().__init__(*dims)
        self.tokenizer = self.make_tokenizer()
        self.quality = nn.Linear(64, 1)
        self.diversity = nn.Linear(32, 16)
        self.pair = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 64))
        self.bridge = zero_linear(64, self.output_dim)

    def correction(self, local, evidence, active, av):
        tokens, mask = self.tokenizer(local, evidence, active)
        results = []
        for row in range(local.shape[0]):
            v, anchor = tokens[row, 1:][mask[row, 1:]], tokens[row, 0]
            if len(v) < 2:
                results.append(local.new_zeros(self.output_dim))
                continue
            q = .1 + 1.9 * torch.sigmoid(self.quality(torch.cat(
                (v, anchor.expand_as(v)), -1))).squeeze(-1)
            u = self.diversity(v)
            phi = u / torch.sqrt(u.square().sum(-1, keepdim=True) + 1e-6)
            i, j = torch.triu_indices(len(v), len(v), offset=1, device=v.device)
            p64, q64 = phi.double(), q.double()
            norm = p64.square().sum(-1) + 1e-3
            det = (norm[i] * norm[j] - (p64[i] * p64[j]).sum(-1).square()).clamp_min(0)
            det = q64[i].square() * q64[j].square() * det
            probability = (det / det.sum()).to(v.dtype)
            pair = self.pair(torch.cat((v[i] + v[j], (v[i] - v[j]).abs(),
                                       v[i] * v[j], anchor.expand(len(i), -1)), -1))
            results.append(self.bridge((probability[:, None] * pair).sum(0)))
        return self.split(torch.stack(results))


def _norm(x):
    return torch.linalg.vector_norm(x, dim=-1, keepdim=True)


def _ball(x):
    return x * ((1 - 1e-4) / _norm(x).clamp_min(1e-12)).clamp(max=1)


def _exp0(x):
    norm = _norm(x)
    ratio = torch.tanh(norm) / norm.clamp_min(1e-12)
    return _ball(x * torch.where(norm > 1e-7, ratio, 1 - norm.square() / 3))


def _log0(x):
    norm = _norm(x)
    ratio = torch.atanh(norm.clamp(max=1-1e-4)) / norm.clamp_min(1e-12)
    return x * torch.where(norm > 1e-7, ratio, 1 + norm.square() / 3)


def _mob(x, y):
    xy = (x * y).sum(-1, keepdim=True)
    xx, yy = x.square().sum(-1, keepdim=True), y.square().sum(-1, keepdim=True)
    denominator = 1 + 2 * xy + xx * yy
    # Count numerical safeguard excursions without persistent model state.
    excursions = (denominator < 1e-8).sum()
    if excursions:
        warnings.warn(f'Mobius denominator safeguard excursions: {int(excursions)}',
                      RuntimeWarning, stacklevel=2)
    return _ball(((1 + 2 * xy + yy) * x + (1 - xx) * y) / denominator.clamp_min(1e-8))


class HyperbolicReadout(_Residual):
    def __init__(self, *dims):
        super().__init__(*dims)
        self.local_projection = nn.Linear(self.latent_dim, 32)
        self.role_projections = nn.ModuleList([nn.Linear(self.num_heads*self.value_dim, 32)
                                               for _ in range(4)])
        self.transforms = nn.ModuleList([nn.Linear(32, 32) for _ in range(4)])
        self.bridge = zero_linear(160, self.output_dim)

    def correction(self, local, evidence, active, av):
        xl = _exp0(.5 * torch.tanh(self.local_projection(local)) / math.sqrt(32))
        g, directions = xl, []
        for role, (projection, transform) in enumerate(zip(self.role_projections, self.transforms)):
            v = safe_mask(.5 * torch.tanh(projection(evidence[:, role])) / math.sqrt(32), active[:, role])
            xr = _exp0(v)
            tangent = F.linear(_log0(xr), transform.weight)
            tangent = tangent / torch.sqrt(1 + tangent.square().sum(-1, keepdim=True))
            yr = safe_mask(_mob(_exp0(tangent), _exp0(transform.bias)), active[:, role])
            composed = _mob(g, yr)
            g = torch.where(active[:, role, None], composed, g)
            displacement = (1 - xl.square().sum(-1, keepdim=True)) * _log0(_mob(-xl, yr))
            directions.append(safe_mask(displacement, active[:, role]))
        return self.split(self.bridge(torch.cat([_log0(g)] + directions, -1)))


class _GraphReadout(_Residual):
    def __init__(self, *dims):
        super().__init__(*dims)
        self.tokenizer = self.make_tokenizer()
        # Fifteen immutable availability graphs; inactive vertices never enter degrees.
        for pattern in range(1, 16):
            columns = [0] + [1 + r*self.num_heads+h for r in range(4)
                             if pattern & (1 << r) for h in range(self.num_heads)]
            ids = torch.tensor(columns)
            roles, heads = (ids-1)//self.num_heads, (ids-1) % self.num_heads
            adjacency = ((ids[:, None] == 0) | (ids[None, :] == 0)
                         | (roles[:, None] == roles[None, :])
                         | (heads[:, None] == heads[None, :]))
            adjacency.fill_diagonal_(False)
            a = adjacency.double()
            degree = a.sum(-1)
            s = a / torch.sqrt(degree[:, None] * degree[None, :])
            self.register_buffer(f'columns_{pattern}', ids)
            self.prepare_operators(pattern, s, degree)

    def groups(self, tokens, active):
        codes = (active.long() * active.new_tensor([1, 2, 4, 8], dtype=torch.long)).sum(-1)
        for pattern in range(1, 16):
            rows = (codes == pattern).nonzero(as_tuple=True)[0]
            if rows.numel():
                columns = getattr(self, f'columns_{pattern}')
                yield pattern, rows, columns, tokens[rows[:, None], columns[None, :]]


class BernsteinReadout(_GraphReadout):
    def __init__(self, *dims):
        super().__init__(*dims)
        self.coefficients = nn.Parameter(torch.full((7,), math.log(math.expm1(1))))
        self.local_bridge = zero_linear(32, self.latent_dim)
        self.evidence_bridge = zero_linear(32, self.value_dim)

    def prepare_operators(self, pattern, s, degree):
        identity = torch.eye(len(s), dtype=s.dtype)
        p = (identity - s)/2
        q = identity-p
        operators = torch.stack([math.comb(6, k)*torch.linalg.matrix_power(q, 6-k)
                                 @ torch.linalg.matrix_power(p, k) for k in range(7)])
        self.register_buffer(f'operators_{pattern}', operators)

    def correction(self, local, evidence, active, av):
        tokens, mask = self.tokenizer(local, evidence, active)
        result = torch.zeros_like(tokens)
        for pattern, rows, columns, packed in self.groups(tokens, active):
            operators = getattr(self, f'operators_{pattern}').to(tokens.dtype)
            operator = (F.softplus(self.coefficients)[:, None, None]*operators).sum(0)
            result[rows[:, None], columns[None, :]] = operator @ packed
        dl = self.local_bridge(result[:, 0])
        de = self.evidence_bridge(result[:, 1:]).reshape_as(evidence)
        return dl, safe_mask(de, active)


class ScatteringReadout(_GraphReadout):
    def __init__(self, *dims):
        super().__init__(*dims)
        self.encoder = nn.Linear(416, 64)
        self.bridge = zero_linear(64, self.output_dim)

    def prepare_operators(self, pattern, s, degree):
        identity = torch.eye(len(s), dtype=s.dtype)
        t = (identity+s)/2
        t2 = t @ t
        self.register_buffer(f'operators_{pattern}', torch.stack((identity-t, t-t2, t2-t2@t2)))
        self.register_buffer(f'lowpass_{pattern}', degree/degree.sum())

    def correction(self, local, evidence, active, av):
        tokens, mask = self.tokenizer(local, evidence, active)
        summaries = local.new_zeros((len(local), 416))
        for pattern, rows, columns, packed in self.groups(tokens, active):
            operators = getattr(self, f'operators_{pattern}').to(tokens.dtype)
            lowpass = getattr(self, f'lowpass_{pattern}').to(tokens.dtype)
            first = [(operator @ packed).abs() for operator in operators]
            pool = lambda x: (lowpass[None, :, None]*x).sum(1)
            paths = [pool(packed)] + [pool(x) for x in first]
            paths += [pool((operator @ x).abs()) for x in first for operator in operators]
            summaries[rows] = torch.cat(paths, -1)
        return self.split(self.bridge(F.gelu(self.encoder(summaries))))


class _SandwichLayer(nn.Module):
    def __init__(self, in_features, nonlinear=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(64, 64+in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(64))
        self.raw_psi = nn.Parameter(torch.zeros(64)) if nonlinear else None

    def forward(self, x):
        u, v = self.weight[:, :64].T, self.weight[:, 64:].T
        z = u-u.T+v.T@v
        identity = torch.eye(64, dtype=x.dtype, device=x.device)
        inverse = torch.linalg.solve(identity+z, identity)
        q = torch.cat((torch.linalg.solve(identity+z, identity-z), -2*v@inverse), 0).T
        if not torch.isfinite(q).all():
            raise FloatingPointError('Nonfinite Cayley solve')
        a, b = q[:, :64], q[:, 64:]
        bx = F.linear(x, b)
        if self.raw_psi is None:
            return bx+self.bias
        psi = 3*torch.tanh(self.raw_psi)
        hidden = torch.exp(psi)*F.relu(math.sqrt(2)*torch.exp(-psi)*bx+self.bias)
        return math.sqrt(2)*F.linear(hidden, a.T)


class SandwichReadout(_Residual):
    def __init__(self, *dims):
        super().__init__(*dims)
        self.layers = nn.Sequential(_SandwichLayer(self.output_dim+3),
                                    _SandwichLayer(64), _SandwichLayer(64, nonlinear=False))
        self.bridge_weight = nn.Parameter(torch.zeros(self.output_dim, 64))
        self.bridge_bias = nn.Parameter(torch.zeros(self.output_dim))

    def correction(self, local, evidence, active, av):
        z = self.layers(torch.cat((local, evidence.flatten(1), av), -1))
        weight = self.bridge_weight / torch.sqrt(1+self.bridge_weight.square().sum())
        return self.split(F.linear(z, weight, self.bridge_bias))


def build(method, latent_dim=256, num_heads=8, value_dim=64):
    classes = {
        'optimization_dpp_subset_readout': DeterminantalReadout,
        'next_hyperbolic_gyrovector_readout': HyperbolicReadout,
        'next_bernstein_spectral_readout': BernsteinReadout,
        'next_diffusion_scattering_readout': ScatteringReadout,
        'next_sandwich_lipschitz_readout': SandwichReadout,
    }
    if method not in classes:
        raise ValueError('Unknown geometry method: '+method)
    return classes[method](latent_dim, num_heads, value_dim)
