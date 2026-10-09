"""Ten conditional computation cores; all state lasts one readout only.

Paper-to-readout adaptations are specified in osram_new40_conditional_cards.json.
No latent coordinate is interpreted as physical time. Original variants retain
Local; the explicit NPS Local-output variant adds a Local adapter residual.
NPS deliberately uses deterministic straight-through argmax, including training,
instead of Gumbel noise to preserve the surrounding cfg84 RNG protocol.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .meaningful_blocks_common import HeadTokenizer, active_groups, safe_mask
from .meaningful_input_new40 import zero_linear


class VectorReadout(nn.Module):
    def __init__(self, core, latent_dim, num_heads, value_dim):
        super().__init__()
        width = num_heads * value_dim
        self.input = nn.Linear(latent_dim + 4 * width + 3, 64)
        self.core = core
        self.bridge = zero_linear(core.output_dim, 4 * width)
        self.width = width
        self.condition = (nn.Linear(latent_dim + 3, 64)
                          if isinstance(core, (Hamiltonian, Lagrangian)) else None)
        self.history_initial = (nn.Linear(4 * width, 64)
                                if isinstance(core, (LTC, CoRNN)) else None)

    def forward(self, local, evidence, active, availability):
        valid = active.any(-1)
        clean = safe_mask(evidence, active)
        rows = valid.nonzero(as_tuple=True)[0]
        delta = torch.zeros_like(evidence)
        if rows.numel():
            x = torch.cat((safe_mask(local, valid), clean.flatten(1),
                           safe_mask(availability, valid).to(local.dtype)), -1)
            encoded = torch.tanh(self.input(x[rows]))
            if self.condition is not None:
                condition = torch.tanh(self.condition(torch.cat((local[rows],
                           availability[rows].to(local.dtype)), -1)))
                features = self.core(encoded, condition)
            elif self.history_initial is not None:
                features = self.core(encoded, self.history_initial(clean[rows].flatten(1)))
            else:
                features = self.core(encoded)
            delta[rows] = self.bridge(features.to(self.bridge.weight.dtype)).reshape(
                -1, 4, self.width).to(evidence.dtype)
        return local, safe_mask(clean + delta, active)


class TokenReadout(nn.Module):
    def __init__(self, core, latent_dim, num_heads, value_dim, local_correction=False):
        super().__init__()
        self.num_heads = num_heads
        self.tokenizer = HeadTokenizer(latent_dim, num_heads, value_dim,
                                       dim=64, normalize=True)
        self.core = core
        self.bridges = nn.ModuleList([zero_linear(64, value_dim)
                                      for _ in range(num_heads)])
        if local_correction:
            with torch.random.fork_rng(devices=[]):
                self.local_bridge = zero_linear(64, latent_dim)

    def forward(self, local, evidence, active, availability):
        tokens, mask = self.tokenizer(local, evidence, active)
        result = torch.zeros_like(tokens)
        for rows, columns, packed in active_groups(tokens, mask):
            result[rows[:, None], columns[None, :]] = self.core(packed)
        values = result[:, 1:].reshape(-1, 4, self.num_heads, 64)
        delta = torch.stack([head(values[:, :, h])
                             for h, head in enumerate(self.bridges)], 2).flatten(2)
        if hasattr(self, 'local_bridge'):
            local = local + safe_mask(self.local_bridge(result[:, 0]), active.any(-1))
        return local, safe_mask(safe_mask(evidence, active) + delta, active)


class FullHypernetwork(nn.Module):
    output_dim = 64

    def __init__(self):
        super().__init__()
        self.condition = nn.Sequential(nn.Linear(64, 64), nn.Tanh())
        self.generators = nn.ModuleList([nn.Linear(64, 64 * 64 + 64)
                                         for _ in range(2)])

    def matrices(self, x):
        c = self.condition(x)
        generated = [g(c) for g in self.generators]
        return [(w[:, :4096].reshape(-1, 64, 64) / 8,
                 w[:, 4096:]) for w in generated]

    def forward(self, x):
        (w1, b1), (w2, b2) = self.matrices(x)
        h = F.gelu(torch.bmm(w1, x.unsqueeze(-1)).squeeze(-1) + b1)
        return torch.bmm(w2, h.unsqueeze(-1)).squeeze(-1) + b2


class LTC(nn.Module):
    output_dim = 64

    def __init__(self):
        super().__init__()
        self.initial = nn.Linear(64, 64)
        self.capacitance = nn.Parameter(torch.zeros(64))
        self.leak = nn.Parameter(torch.zeros(64))
        self.leak_reversal = nn.Parameter(torch.zeros(64))
        for name in ('recurrent', 'sensory'):
            setattr(self, name + '_weight', nn.Parameter(torch.full((64, 64), -2.0)))
            setattr(self, name + '_slope', nn.Parameter(torch.zeros(64, 64)))
            setattr(self, name + '_threshold', nn.Parameter(torch.randn(64, 64) * .1))
            setattr(self, name + '_reversal', nn.Parameter(torch.randn(64, 64)))

    def conductance(self, source, name):
        # Axes: sample, source neuron i, target neuron j.
        return F.softplus(getattr(self, name + '_weight')) * torch.sigmoid(
            F.softplus(getattr(self, name + '_slope')) *
            (source[:, :, None] - getattr(self, name + '_threshold')))

    def forward(self, u, initial=None):
        v = torch.tanh(self.initial(u if initial is None else initial))
        gs = self.conductance(u, 'sensory')
        sensory_current = (gs * self.sensory_reversal).sum(1)
        cap = (F.softplus(self.capacitance) + 1e-4) / .1
        leak = F.softplus(self.leak) + 1e-4
        for _ in range(6):
            gr = self.conductance(v, 'recurrent')
            v = (cap * v + leak * self.leak_reversal +
                 (gr * self.recurrent_reversal).sum(1) + sensory_current) / (
                     cap + leak + gr.sum(1) + gs.sum(1))
        return v


class Hamiltonian(nn.Module):
    output_dim = 64

    def __init__(self):
        super().__init__()
        self.energy = nn.Sequential(nn.Linear(128, 64), nn.Tanh(),
                                    nn.Linear(64, 1))

    def field(self, z, condition, create_graph=True):
        if not z.requires_grad:
            z = z.detach().requires_grad_(True)
        h = self.energy(torch.cat((z, condition), -1)).sum()
        derivative = torch.autograd.grad(h, z, create_graph=create_graph)[0]
        dq, dp = derivative.chunk(2, -1)
        return torch.cat((dp, -dq), -1)

    def forward(self, x, condition=None):
        tracking = torch.is_grad_enabled()
        with torch.inference_mode(False), torch.enable_grad():
            c = (x if condition is None else condition).clone()
            # Distinct node: partial_z H must hold c fixed even initially.
            z = x.clone()
            for _ in range(8):
                k1 = self.field(z, c, tracking)
                k2 = self.field(z + .025 * k1, c, tracking)
                k3 = self.field(z + .025 * k2, c, tracking)
                k4 = self.field(z + .05 * k3, c, tracking)
                z = z + (.05 / 6) * (k1 + 2*k2 + 2*k3 + k4)
                if not tracking:
                    z = z.detach()
        return z if tracking else z.detach()


class Lagrangian(nn.Module):
    """Regular mechanical L = v^T M(q,c) v / 2 - V(q,c).

    M = B B^T + diag(1 + softplus(Aq + Cc)) is everywhere SPD.
    Consequently the actual velocity Hessian has no singular-value cutoff:
    solve(M, rhs) equals pinv(M) @ rhs. q-dependent M preserves the nonzero
    mixed velocity/position derivative; this is a restricted Lagrangian
    parametrization, not a replacement by a learned acceleration field.
    All derivatives and the 16x16 solve run in float64.
    """
    output_dim = 32

    def __init__(self):
        super().__init__()
        self.initial = nn.Linear(64, 32)
        self.mass_diagonal = nn.Linear(80, 16)
        self.mass_factor = nn.Parameter(torch.eye(16) * .2)
        self.potential = nn.Sequential(nn.Linear(80, 32), nn.Tanh(), nn.Linear(32, 1))
        self.double()

    def lagrangian(self, z, c):
        q, v = z.chunk(2, -1)
        context = torch.cat((q, c), -1)
        diagonal = 1 + F.softplus(self.mass_diagonal(context))
        kinetic = .5 * ((v @ self.mass_factor).square().sum(-1) +
                         (diagonal * v.square()).sum(-1))
        return kinetic - self.potential(context).squeeze(-1)

    def acceleration(self, z, c, create_graph=True):
        if not z.requires_grad:
            z = z.detach().requires_grad_(True)
        d_l = torch.autograd.grad(self.lagrangian(z, c).sum(), z,
                                  create_graph=True)[0]
        q_grad, v_grad = d_l.chunk(2, -1)
        # Differentiate with respect to the combined independent coordinates:
        # taking grad w.r.t. evolving q/v separately can introduce false paths.
        second = torch.stack([torch.autograd.grad(v_grad[:, i].sum(), z,
                             create_graph=create_graph, retain_graph=True)[0]
                              for i in range(16)], 1)
        mixed, hessian = second.split(16, -1)
        rhs = q_grad - torch.bmm(mixed, z[:, 16:, None]).squeeze(-1)
        return torch.linalg.solve(hessian, rhs.unsqueeze(-1)).squeeze(-1)

    def forward(self, x, condition=None):
        tracking = torch.is_grad_enabled()
        with torch.inference_mode(False), torch.enable_grad():
            c = (x if condition is None else condition).clone().double()
            z = self.initial(x.clone().double())
            for _ in range(4):
                a = self.acceleration(z, c, tracking)
                q, v = z.chunk(2, -1)
                # Explicit finite-time Euler integration, dt=.025.
                z = torch.cat((q + .025*v, v + .025*a), -1)
                if not tracking:
                    z = z.detach()
        return z.to(x.dtype) if tracking else z.detach().to(x.dtype)


class ContractingREN(nn.Module):
    """Acyclic ContractingREN, H-to-explicit equations from the author code.

    X^T X, E/F/B/C/D orientations follow RobustNeuralNetworks.jl
    src/ParameterTypes/utils.jl::hmatrix_to_explicit (MIT). alpha=.9.
    The contraction statement only concerns its internal fixed-input state.
    """
    output_dim = 64

    def __init__(self, dim=24):
        super().__init__()
        self.dim = dim
        self.X = nn.Parameter(torch.eye(3*dim) + torch.randn(3*dim, 3*dim)*.03)
        self.Y = nn.Parameter(torch.zeros(dim, dim))
        self.B2 = nn.Parameter(torch.randn(dim, 64) / 8)
        self.D12 = nn.Parameter(torch.randn(dim, 64) / 8)
        self.bx = nn.Parameter(torch.zeros(dim))
        self.bv = nn.Parameter(torch.zeros(dim))
        self.initial = nn.Linear(64, dim)
        self.output = nn.Linear(2*dim + 64, 64)

    def explicit(self):
        n = self.dim
        h = self.X.T @ self.X + 1e-3 * torch.eye(3*n, device=self.X.device,
                                                dtype=self.X.dtype)
        h11, h22, h33 = h[:n, :n], h[n:2*n, n:2*n], h[2*n:, 2*n:]
        e = (h11 + h33 / .9**2 + self.Y - self.Y.T) / 2
        inv_lambda = 2 / h22.diag()
        a = torch.linalg.solve(e, h[2*n:, :n])
        b1 = torch.linalg.solve(e, h[2*n:, n:2*n])
        b2 = torch.linalg.solve(e, self.B2)
        c1 = -inv_lambda[:, None] * h[n:2*n, :n]
        d11 = -inv_lambda[:, None] * h22.tril(-1)
        d12 = inv_lambda[:, None] * self.D12
        return a, b1, b2, c1, d11, d12

    def forward(self, u):
        a, b1, b2, c1, d11, d12 = self.explicit()
        x = self.initial(u)
        for _ in range(4):
            pre = F.linear(x, c1) + F.linear(u, d12) + self.bv
            ws = []
            for i in range(self.dim):
                feedback = (torch.stack(ws, -1) * d11[i, :i]).sum(-1) if i else 0
                ws.append(F.relu(pre[:, i] + feedback))
            w = torch.stack(ws, -1)
            x = F.linear(x, a) + F.linear(w, b1) + F.linear(u, b2) + self.bx
        return self.output(torch.cat((x, w, u), -1))


class RIM(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial = nn.Linear(64, 4*64)
        self.query = nn.Linear(64, 64, bias=False)
        self.key = nn.Linear(64, 64, bias=False)
        self.value = nn.Linear(64, 64, bias=False)
        self.cells = nn.ModuleList([nn.LSTMCell(64, 64) for _ in range(4)])
        self.comm_q = nn.Linear(64, 64, bias=False)
        self.comm_k = nn.Linear(64, 64, bias=False)
        self.comm_v = nn.Linear(64, 64, bias=False)
        self.decode = nn.Linear(4*64, 64)

    def step(self, h, cell, tokens):
        null = tokens.new_zeros(tokens.shape[0], 1, 64)
        inputs = torch.cat((tokens, null), 1)
        attention = (self.query(h) @ self.key(inputs).transpose(-1, -2) / 8).softmax(-1)
        chosen = (1 - attention[:, :, -1]).topk(2, -1).indices
        active = torch.zeros_like(h[:, :, 0], dtype=torch.bool).scatter(1, chosen, True)
        reads = attention @ self.value(inputs)
        proposals = [self.cells[i](reads[:, i], (h[:, i], cell[:, i])) for i in range(4)]
        new_h = torch.stack([v[0] for v in proposals], 1)
        new_c = torch.stack([v[1] for v in proposals], 1)
        new_h = torch.where(active[:, :, None], new_h, h)
        new_c = torch.where(active[:, :, None], new_c, cell)
        # Frozen mechanisms can be read but receive no communication gradient.
        senders = torch.where(active[:, :, None], new_h, new_h.detach())
        comm = (self.comm_q(new_h) @ self.comm_k(senders).transpose(-1, -2) / 8).softmax(-1)
        new_h = torch.where(active[:, :, None], new_h + comm @ self.comm_v(senders), h)
        return new_h, new_c, active

    def forward(self, tokens):
        h = self.initial(tokens.mean(1)).reshape(-1, 4, 64)
        cell = torch.zeros_like(h)
        for _ in range(3):
            h, cell, _ = self.step(h, cell, tokens)
        return tokens + self.decode(h.flatten(1))[:, None]


def straight_through_argmax(logits):
    """Hard deterministic forward; softmax Jacobian backward (no RNG use)."""
    soft = logits.softmax(-1)
    hard = F.one_hot(logits.argmax(-1), logits.shape[-1]).to(soft.dtype)
    return hard + (soft - soft.detach())


class NeuralProduction(nn.Module):
    def __init__(self, rule_hidden=96):
        super().__init__()
        self.rules = nn.Parameter(torch.randn(4, 64) / 8)
        self.primary_key = nn.Linear(64, 64)
        self.context_query = nn.Linear(128, 64)
        self.context_key = nn.Linear(64, 64)
        self.rule_mlps = nn.ModuleList([nn.Sequential(nn.Linear(128, rule_hidden), nn.ReLU(),
                                                     nn.Linear(rule_hidden, 64)) for _ in range(4)])

    def step(self, variables):
        batch, count, _ = variables.shape
        logits = torch.einsum('rd,btd->brt', self.rules, self.primary_key(variables)) / 8
        joint = straight_through_argmax(logits.flatten(1)).reshape(batch, 4, count)
        rule_choice, primary_choice = joint.sum(-1), joint.sum(1)
        primary = (variables * primary_choice[:, :, None]).sum(1)
        rule_code = rule_choice @ self.rules
        query = self.context_query(torch.cat((primary, rule_code), -1))
        scores = (self.context_key(variables) * query[:, None]).sum(-1) / 8
        # Local plus >=1 active evidence head means a distinct context exists.
        scores = scores.masked_fill(primary_choice.detach().bool(), -torch.inf)
        context_choice = straight_through_argmax(scores)
        context = (variables * context_choice[:, :, None]).sum(1)
        inputs = torch.cat((primary, context), -1)
        # Dispatch ONLY the hard-selected rule. The selected ST gate and bound
        # inputs retain a surrogate gradient for rule/primary/context selection;
        # no soft execution of all rule MLPs or Gumbel randomness is performed.
        delta = torch.zeros_like(primary)
        selected = rule_choice.detach().argmax(-1)
        for i, rule in enumerate(self.rule_mlps):
            rows = (selected == i).nonzero(as_tuple=True)[0]
            if rows.numel():
                delta[rows] = rule(inputs[rows]) * rule_choice[rows, i, None]
        return variables + primary_choice[:, :, None] * delta[:, None], primary_choice

    def forward(self, tokens):
        for _ in range(3):
            tokens, _ = self.step(tokens)
        return tokens


class NeuralInterpreter(nn.Module):
    """Nonquantized function signatures, inferred types and code-conditioned LOC.

    Hard compatibility cutoff follows author kernel distance <= tau. Each
    function has a normalized type signature; all functions share interpreter
    weights but carry trainable codes conditioning both attention and MLP.
    """
    def __init__(self):
        super().__init__()
        self.types = nn.Sequential(nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 16))
        self.register_buffer('signatures', F.normalize(torch.randn(4, 16), dim=-1))
        self.codes = nn.Parameter(torch.randn(4, 32) * .1)
        self.qkv = nn.Linear(64, 192)
        self.code_qkv = nn.Linear(32, 384)
        self.mlp = nn.Linear(64, 128)
        self.code_mlp = nn.Linear(32, 256)
        self.mlp_out = nn.Linear(128, 64)
        self.attention_out = nn.Linear(64, 64)
        self.norm1, self.norm2 = nn.LayerNorm(64), nn.LayerNorm(64)
        self.tau, self.sigma = 1.5, .5

    def compatibility(self, tokens):
        types = F.normalize(self.types(tokens), dim=-1)
        distance = 1 - torch.einsum('btd,fd->bft', types, self.signatures)
        weight = torch.exp(-distance / self.sigma) * (distance <= self.tau)
        return weight / weight.sum(1, keepdim=True).clamp_min(1e-12)

    def loc(self, values, compatibility):
        scale, shift = self.code_qkv(self.codes).chunk(2, -1)
        q, k, v = (self.qkv(self.norm1(values)) * (1 + scale[None, :, None]) +
                   shift[None, :, None]).chunk(3, -1)
        logits = q @ k.transpose(-1, -2) / 8
        pair = compatibility[:, :, :, None] * compatibility[:, :, None, :]
        weights = torch.exp(logits - logits.amax(-1, keepdim=True)) * pair
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        values = values + self.attention_out(weights @ v)
        scale, shift = self.code_mlp(self.codes).chunk(2, -1)
        hidden = self.mlp(self.norm2(values)) * (1 + scale[None, :, None]) + shift[None, :, None]
        return values + self.mlp_out(F.gelu(hidden))

    def forward(self, tokens):
        for _ in range(2):
            compatibility = self.compatibility(tokens)
            values = tokens[:, None].expand(-1, 4, -1, -1)
            for _ in range(2):
                values = self.loc(values, compatibility)
            proposal = (values * compatibility[:, :, :, None]).sum(1)
            tokens = torch.where((compatibility.sum(1) > 0)[:, :, None], proposal, tokens)
        return tokens


class GradientShear(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.K = nn.Parameter(torch.randn(64, dim) / math.sqrt(dim))
        self.a = nn.Parameter(torch.randn(64) * .1)
        self.b = nn.Parameter(torch.zeros(64))

    def forward(self, q):
        return .1 * (torch.tanh(F.linear(q, self.K, self.b)) * self.a) @ self.K


class SympNet(nn.Module):
    output_dim = 64

    def __init__(self):
        super().__init__()
        self.shears = nn.ModuleList([GradientShear() for _ in range(6)])

    def forward(self, x):
        p, q = x.chunk(2, -1)
        for i, shear in enumerate(self.shears):
            if i % 2 == 0:
                p = p + shear(q)
            else:
                q = q + shear(p)
        return torch.cat((p, q), -1)


class CoRNN(nn.Module):
    output_dim = 128

    def __init__(self):
        super().__init__()
        self.initial = nn.Linear(64, 64)
        self.drive = nn.Linear(192, 64)
        self.dt, self.gamma, self.epsilon = .05, 1., 1.

    def forward(self, u, initial=None):
        y = self.initial(u if initial is None else initial)
        z = torch.zeros_like(y)
        for _ in range(8):
            z = z + self.dt * (torch.tanh(self.drive(torch.cat((u, z, y), -1))) -
                              self.gamma*y - self.epsilon*z)
            y = y + self.dt*z
        return torch.cat((y, z), -1)


def build(method, latent_dim=256, num_heads=8, value_dim=64):
    cores = {
        'conditional_new_01_full_dynamic_hypernetwork': FullHypernetwork,
        'conditional_new_02_ltc_conductance': LTC,
        'conditional_new_03_hamiltonian_flow': Hamiltonian,
        'conditional_new_04_lagrangian_flow': Lagrangian,
        'conditional_new_05_contracting_ren': ContractingREN,
        'conditional_new_06_rim': RIM,
        'conditional_new_07_neural_production': NeuralProduction,
        'conditional_new_08_neural_interpreter': NeuralInterpreter,
        'conditional_new_09_sympnet': SympNet,
        'conditional_new_10_cornn': CoRNN,
    }
    if method not in cores:
        raise ValueError('Unknown conditional method: ' + method)
    core = cores[method]()
    wrapper = TokenReadout if isinstance(core, (RIM, NeuralProduction, NeuralInterpreter)) else VectorReadout
    return wrapper(core, latent_dim, num_heads, value_dim)
