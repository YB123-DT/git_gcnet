"""Fixed-grid functions, calibrated lattices, logic circuits and finite IVPs.

Independent implementations of the four operators in osram_new40_nine_designs.
There is no training-time state, random forward wiring, or external solver.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .meaningful_blocks_common import safe_mask
from .meaningful_input_new40 import zero_linear


def _finite(value, label):
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError('Nonfinite ' + label)


class _FunctionAdapter(nn.Module):
    def __init__(self, latent_dim, num_heads, value_dim, width):
        super().__init__()
        self.latent_dim = latent_dim
        self.memory_dim = num_heads * value_dim
        self.input_dim = latent_dim + 4 * self.memory_dim + 3
        self.bridge = zero_linear(width, latent_dim + 4 * self.memory_dim)

    def forward(self, local, evidence, active, av):
        n = local.shape[0]
        if evidence.shape != (n, 4, self.memory_dim) or active.shape != (n, 4):
            raise ValueError('Expected four canonical evidence roles')
        clean = safe_mask(evidence, active)
        if n == 0:
            return local, clean
        valid = active.bool().any(-1)
        if not bool(valid.any()):
            return local, clean
        # Exclude empty rows before conditioning or any learned arithmetic.
        dtype = torch.float64 if local.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=local.device.type, enabled=False):
            x, e, a = local[valid].to(dtype), clean[valid].to(dtype), av[valid].to(dtype)
            _finite(x, 'active local input')
            _finite(e, 'active evidence input')
            _finite(a, 'availability input')
            features = self.features(x, e, a)
            _finite(features, 'function features')
            delta = self.bridge(features)
            _finite(delta, 'function residual')
        dl = torch.zeros_like(local).index_copy(0, valid.nonzero(as_tuple=True)[0],
                                               delta[:, :self.latent_dim].to(local.dtype))
        de = torch.zeros_like(clean).index_copy(
            0, valid.nonzero(as_tuple=True)[0],
            delta[:, self.latent_dim:].reshape(-1, 4, self.memory_dim).to(evidence.dtype))
        result_local = local + dl
        result_evidence = safe_mask(clean + de, active)
        _finite(result_local[valid], 'active local output')
        _finite(result_evidence, 'evidence output')
        return result_local, result_evidence

    @staticmethod
    def flatten(local, evidence, av):
        return torch.cat((local, evidence.flatten(1), av), -1)


class CubicKANLayer(nn.Module):
    """Each directed edge owns a SiLU coefficient and eight cubic coefficients."""
    def __init__(self, inputs, outputs):
        super().__init__()
        self.base = nn.Parameter(torch.empty(outputs, inputs))
        self.coefficients = nn.Parameter(torch.empty(outputs, inputs, 8))
        nn.init.uniform_(self.base, -1 / math.sqrt(inputs), 1 / math.sqrt(inputs))
        nn.init.uniform_(self.coefficients, -.1 / math.sqrt(inputs), .1 / math.sqrt(inputs))
        self.register_buffer('knots', -1 + (torch.arange(12, dtype=torch.float64) - 3) * .4)

    def basis(self, x):
        x = x.clamp(-1 + 1e-6, 1 - 1e-6).unsqueeze(-1)
        knots = self.knots.to(x)
        basis = ((x >= knots[:-1]) & (x < knots[1:])).to(x.dtype)
        for degree in range(1, 4):
            count = 11 - degree
            left = (x - knots[:count]) / (knots[degree:degree + count] - knots[:count])
            right = (knots[degree + 1:degree + 1 + count] - x) / (
                knots[degree + 1:degree + 1 + count] - knots[1:1 + count])
            basis = left * basis[..., :-1] + right * basis[..., 1:]
        return basis

    def forward(self, x):
        x = x.clamp(-1 + 1e-6, 1 - 1e-6)
        return F.linear(F.silu(x), self.base) + torch.einsum(
            'niq,oiq->no', self.basis(x), self.coefficients)


class KANAdapter(_FunctionAdapter):
    def __init__(self, latent_dim, num_heads, value_dim):
        super().__init__(latent_dim, num_heads, value_dim, 64)
        self.front = nn.Linear(self.input_dim, 64)
        self.first = CubicKANLayer(64, 32)
        self.second = CubicKANLayer(32, 64)

    def features(self, local, evidence, av):
        x = self.front(self.flatten(local, evidence, av)).tanh()
        return self.second(self.first(x).tanh())


class PWLCalibrator(nn.Module):
    def __init__(self):
        super().__init__()
        identity = torch.linspace(.02, .98, 5).logit().expand(64, -1)
        self.vertices = nn.Parameter(identity + .03 * torch.randn(64, 5))

    def forward(self, x):
        scaled = x.clamp(0, 1) * 4
        cell = scaled.floor().long().clamp(max=3)
        fraction = scaled - cell
        values = self.vertices.sigmoid().unsqueeze(0).expand(x.shape[0], -1, -1)
        left = values.gather(2, cell.unsqueeze(-1)).squeeze(-1)
        right = values.gather(2, (cell + 1).unsqueeze(-1)).squeeze(-1)
        return left + fraction * (right - left)


class LatticeBank(nn.Module):
    def __init__(self):
        super().__init__()
        self.vertices = nn.Parameter(.2 * torch.randn(16, 81))
        corners = (torch.arange(16)[:, None] >> torch.arange(3, -1, -1)) & 1
        self.register_buffer('corners', corners)
        self.register_buffer('strides', torch.tensor([27, 9, 3, 1]))

    def forward(self, x):
        scaled = x.reshape(-1, 16, 4).clamp(0, 1) * 2
        cell = scaled.floor().long().clamp(max=1)
        fraction = scaled - cell
        corner = self.corners[None, None]
        index = ((cell.unsqueeze(-2) + corner) * self.strides).sum(-1)
        weights = torch.where(corner.bool(), fraction.unsqueeze(-2),
                              1 - fraction.unsqueeze(-2)).prod(-1)
        vertices = self.vertices.sigmoid().unsqueeze(0).expand(x.shape[0], -1, -1)
        return (vertices.gather(2, index) * weights).sum(-1)


class LatticeAdapter(_FunctionAdapter):
    def __init__(self, latent_dim, num_heads, value_dim):
        super().__init__(latent_dim, num_heads, value_dim, 16)
        self.front = nn.Linear(self.input_dim, 64)
        self.calibrators = nn.ModuleList([PWLCalibrator() for _ in range(3)])
        self.embedding = nn.Linear(64, 64)
        self.second_embedding = nn.Linear(16, 64)
        self.banks = nn.ModuleList([LatticeBank(), LatticeBank()])

    def features(self, local, evidence, av):
        x = self.calibrators[0](self.front(self.flatten(local, evidence, av)).sigmoid())
        x = self.calibrators[1](self.embedding(x).sigmoid())
        x = self.banks[0](x)
        x = self.calibrators[2](self.second_embedding(x).sigmoid())
        return self.banks[1](x)


class LogicLayer(nn.Module):
    def __init__(self, inputs, layer):
        super().__init__()
        node = torch.arange(256)
        first = node % inputs
        second = (first + 1 + (node // inputs + 17 * layer) % (inputs - 1)) % inputs
        self.register_buffer('first', first)
        self.register_buffer('second', second)
        self.register_buffer('truth', ((torch.arange(16)[:, None] >>
                                       torch.arange(3, -1, -1)) & 1).float())
        self.logits = nn.Parameter(.1 * torch.randn(256, 16))

    def forward(self, x):
        a, b = x[:, self.first], x[:, self.second]
        basis = torch.stack(((1-a)*(1-b), (1-a)*b, a*(1-b), a*b), -1)
        if self.training:
            truth = self.logits.softmax(-1) @ self.truth
        else:
            truth = self.truth[self.logits.argmax(-1)]
        return (basis * truth.unsqueeze(0)).sum(-1)


class LogicAdapter(_FunctionAdapter):
    def __init__(self, latent_dim, num_heads, value_dim):
        super().__init__(latent_dim, num_heads, value_dim, 32)
        self.front = nn.Linear(self.input_dim, 64)
        self.layers = nn.ModuleList([LogicLayer(64 if i == 0 else 256, i) for i in range(4)])

    def features(self, local, evidence, av):
        x = self.front(self.flatten(local, evidence, av)).sigmoid()
        for layer in self.layers:
            x = layer(x)
        return x.reshape(-1, 32, 8).mean(-1)


class FiniteODEAdapter(_FunctionAdapter):
    """Finite-time BS3(2); gradients follow accepted states and a detached mesh."""
    atol, rtol = 1e-5, 1e-4
    min_step, max_step, max_attempts = 1e-5, .25, 256

    def __init__(self, latent_dim, num_heads, value_dim):
        super().__init__(latent_dim, num_heads, value_dim, 64)
        self.initial = nn.Linear(4 * self.memory_dim, 64)
        self.condition = nn.Linear(latent_dim + 3, 32)
        self.field_first = nn.Linear(97, 64)
        self.field_second = nn.Linear(64, 64)

    def field(self, t, z, condition):
        time = z.new_full((*z.shape[:-1], 1), t)
        x = torch.cat((z, condition, time), -1)
        return self.field_second(self.field_first(x).tanh()).tanh()

    def integrate(self, initial, condition, sample):
        z, time, step = initial, 0., .125
        error = float('nan')
        for _ in range(self.max_attempts):
            h = min(step, 1. - time)
            k1 = self.field(time, z, condition)
            k2 = self.field(time + h/2, z + h*k1/2, condition)
            k3 = self.field(time + 3*h/4, z + 3*h*k2/4, condition)
            third = z + h*(2*k1/9 + k2/3 + 4*k3/9)
            k4 = self.field(time + h, third, condition)
            second = z + h*(7*k1/24 + k2/4 + k3/3 + k4/8)
            scale = self.atol + self.rtol * torch.maximum(z.abs(), third.abs())
            error = float((((third-second)/scale).square().mean().sqrt()).detach())
            if not math.isfinite(error) or not bool(torch.isfinite(third).all()):
                raise RuntimeError(f'BS3(2) nonfinite: sample={sample}, t={time}, error={error}')
            if error <= 1:
                z = third
                time += h
                if time >= 1.:
                    return z
            elif h <= self.min_step:
                raise RuntimeError(f'BS3(2) minimum step: sample={sample}, t={time}, error={error}')
            factor = min(5., max(.2, .9 * max(error, 1e-12)**(-1/3)))
            step = min(self.max_step, max(self.min_step, h * factor))
        raise RuntimeError(f'BS3(2) attempt limit: sample={sample}, t={time}, error={error}')

    def features(self, local, evidence, av):
        initial = self.initial(evidence.flatten(1)).tanh()
        condition = self.condition(torch.cat((local, av), -1)).tanh()
        final = torch.stack([self.integrate(z, c, i)
                             for i, (z, c) in enumerate(zip(initial, condition))])
        return final - initial


def build(method, latent_dim=256, num_heads=8, value_dim=64):
    classes = {
        'repr_kan_function_composition': KANAdapter,
        'repr_deep_lattice_composition': LatticeAdapter,
        'repr_differentiable_logic_circuit': LogicAdapter,
        'conditional_03_neural_ode_finite_flow': FiniteODEAdapter,
    }
    if method not in classes:
        raise ValueError('Unknown function method: ' + method)
    return classes[method](latent_dim, num_heads, value_dim)
