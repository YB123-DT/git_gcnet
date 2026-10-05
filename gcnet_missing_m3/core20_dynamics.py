"""Persistent, causal sequence cores for C05--C08 (see pinned source audit).

All states are local to scan; reads precede current observed key/value writes.
The recurrent/CDE output adapter conditions the historic state on each query.
TTT instead queries its actual fast MLP. Neither path feeds reads back as data.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

MODALITIES = ("audio", "text", "visual")
METHODS = ("C05", "C06", "C07", "C08")


class HierarchicalMultiscaleRNN(nn.Module):
    """Three HM-LSTM layers, hard boundaries with straight-through gradients."""

    def __init__(self, input_dim, dim):
        super().__init__()
        self.dim = dim
        self.bottom = nn.ModuleList([nn.Linear(input_dim if i == 0 else dim, 4 * dim + 1,
                                              bias=False) for i in range(3)])
        self.recurrent = nn.ModuleList([nn.Linear(dim, 4 * dim + 1) for _ in range(3)])
        self.top = nn.ModuleList([nn.Linear(dim, 4 * dim + 1, bias=False) for _ in range(2)])

    def initial(self, x):
        h = x.new_zeros(*x.shape[:-1], 3, self.dim)
        return h, h.clone(), x.new_zeros(*x.shape[:-1], 3, 1)

    @staticmethod
    def transition(h, c, previous_boundary, lower_boundary, gates):
        f, i, o, g, boundary = gates.split((h.shape[-1],) * 4 + (1,), -1)
        proposal = i.sigmoid() * g.tanh()
        flush = previous_boundary
        update = (1 - flush) * lower_boundary
        copy = (1 - flush) * (1 - lower_boundary)
        new_c = copy * c + update * (f.sigmoid() * c + proposal) + flush * proposal
        new_h = copy * h + (1 - copy) * o.sigmoid() * new_c.tanh()
        probability = ((boundary + 1) / 2).clamp(0, 1)
        hard = (probability > .5).to(probability.dtype)
        z = probability + (hard - probability).detach()
        return new_h, new_c, z

    def step(self, state, x, observed):
        hs, cs, zs = state
        out_h, out_c, out_z = [], [], []
        lower, lower_z = x, torch.ones_like(zs[..., 0, :])
        for layer in range(3):
            old_z = zs[..., layer, :] if layer < 2 else torch.zeros_like(lower_z)
            gates = self.recurrent[layer](hs[..., layer, :]) + self.bottom[layer](lower) * lower_z
            if layer < 2:
                gates = gates + self.top[layer](hs[..., layer + 1, :]) * old_z
            h, c, z = self.transition(hs[..., layer, :], cs[..., layer, :], old_z, lower_z, gates)
            mask = observed[..., None]
            h = torch.where(mask, h, hs[..., layer, :])
            c = torch.where(mask, c, cs[..., layer, :])
            z = torch.where(mask, z, zs[..., layer, :])
            out_h.append(h); out_c.append(c); out_z.append(z)
            lower, lower_z = h, z
        return torch.stack(out_h, -2), torch.stack(out_c, -2), torch.stack(out_z, -2)

    def summary(self, state):
        return state[0].flatten(-2)


class NeuralCDE(nn.Module):
    """Causal forward-filled control plus intensity and utterance index-time.

    Each new observation extends a piecewise-linear path. RK4 differentiates
    through the solve directly; no spline fitted to future observations.
    """

    def __init__(self, key_dim, value_dim, dim):
        super().__init__()
        self.dim = dim
        self.slot_dim = key_dim + value_dim
        self.control_dim = 3 * self.slot_dim + 4
        self.initialize = nn.Linear(self.control_dim, dim)
        self.field = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(),
                                   nn.Linear(dim, dim * self.control_dim), nn.Tanh())
        self.substeps = 4

    def initial(self, x):
        return (x.new_zeros(*x.shape[:-1], self.dim),
                x.new_zeros(*x.shape[:-1], self.control_dim),
                torch.zeros(x.shape[:-1], device=x.device, dtype=torch.bool))

    def integrate(self, z, delta):
        def rhs(h):
            matrix = self.field(h).reshape(*h.shape[:-1], self.dim, self.control_dim)
            return (matrix @ delta[..., None]).squeeze(-1)
        dt = 1 / self.substeps
        for _ in range(self.substeps):
            a = rhs(z); b = rhs(z + dt * a / 2)
            c = rhs(z + dt * b / 2); d = rhs(z + dt * c)
            z = z + dt / 6 * (a + 2 * b + 2 * c + d)
        return z

    def step(self, state, x, available, index_time):
        z, previous, seen = state
        slots = x[..., :3 * self.slot_dim].reshape(*x.shape[:-1], 3, self.slot_dim)
        previous_slots = previous[..., :3 * self.slot_dim].reshape_as(slots)
        filled = torch.where(available[..., None], slots, previous_slots).flatten(-2)
        counts = previous[..., -4:-1] + available.to(x.dtype)
        control = torch.cat((filled, counts, index_time[..., None]), -1)
        delta = control - previous
        if self.training and torch.is_grad_enabled():
            # Retain the segment inputs rather than sixteen large f(z) matrices.
            # Non-reentrant checkpoint preserves differentiation w.r.t. field
            # parameters even when the initial state inputs require no grad.
            proposal = checkpoint(self.integrate, z, delta, use_reentrant=False)
        else:
            proposal = self.integrate(z, delta)
        proposal = torch.where(seen[..., None], proposal, self.initialize(control).tanh())
        observed = available.any(-1)
        return (torch.where(observed[..., None], proposal, z),
                torch.where(observed[..., None], control, previous), seen | observed)

    def summary(self, state):
        return state[0]


class RecurrentIndependentMechanisms(nn.Module):
    """Independent LSTMs compete against a null input, then communicate."""

    def __init__(self, slot_dim, dim, mechanisms=4, active=2):
        super().__init__()
        self.dim, self.mechanisms, self.active = dim, mechanisms, active
        self.input_keys = nn.ModuleList([nn.Linear(slot_dim, dim, bias=False) for _ in range(mechanisms)])
        self.input_values = nn.ModuleList([nn.Linear(slot_dim, dim, bias=False) for _ in range(mechanisms)])
        self.input_queries = nn.ModuleList([nn.Linear(dim, dim) for _ in range(mechanisms)])
        self.cells = nn.ModuleList([nn.LSTMCell(dim, dim) for _ in range(mechanisms)])
        self.comm_q = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(mechanisms)])
        self.comm_k = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(mechanisms)])
        self.comm_v = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(mechanisms)])
        self.last_selected = None

    def initial(self, x):
        h = x.new_zeros(*x.shape[:-2], self.mechanisms, self.dim)
        return h, h.clone()

    def step(self, state, slots, available):
        h, c = state
        shape = h.shape
        h, c = h.reshape(-1, self.mechanisms, self.dim), c.reshape(-1, self.mechanisms, self.dim)
        slots, available = slots.reshape(-1, 3, slots.shape[-1]), available.reshape(-1, 3)
        candidates_h, candidates_c, relevance = [], [], []
        for i in range(self.mechanisms):
            k = self.input_keys[i](slots)
            score = torch.einsum("bd,bmd->bm", self.input_queries[i](h[:, i]), k) / math.sqrt(self.dim)
            score = score.masked_fill(~available, -torch.inf)
            weights = torch.cat((score.new_zeros(score.shape[0], 1), score), -1).softmax(-1)
            inp = torch.einsum("bm,bmd->bd", weights[:, 1:], self.input_values[i](slots))
            hi, ci = self.cells[i](inp, (h[:, i], c[:, i]))
            candidates_h.append(hi); candidates_c.append(ci); relevance.append(1 - weights[:, 0])
        relevance = torch.stack(relevance, -1)
        selected = torch.zeros_like(relevance, dtype=torch.bool).scatter(1, relevance.topk(self.active, -1).indices, True)
        selected = selected & available.any(-1, keepdim=True)
        self.last_selected = selected.detach().reshape(*shape[:-2], self.mechanisms)
        proposed_h, proposed_c = torch.stack(candidates_h, 1), torch.stack(candidates_c, 1)
        # Inactive states can provide context, but cannot change.
        temporary = torch.where(selected[..., None], proposed_h, h)
        q = torch.stack([f(temporary[:, i]) for i, f in enumerate(self.comm_q)], 1)
        k = torch.stack([f(temporary[:, i]) for i, f in enumerate(self.comm_k)], 1)
        v = torch.stack([f(temporary[:, i]) for i, f in enumerate(self.comm_v)], 1)
        communication = (q @ k.transpose(-2, -1) / math.sqrt(self.dim)).softmax(-1) @ v
        return (torch.where(selected[..., None], temporary + communication, h).reshape(shape),
                torch.where(selected[..., None], proposed_c, c).reshape(shape))

    def summary(self, state):
        return state[0].flatten(-2)


class TTTMLP(nn.Module):
    """Online primal TTT-MLP, residual + layernorm and exact inner gradients.

    Analytic gradient operators remain differentiable in the outer loop, while
    also running under no_grad/inference_mode at evaluation. Targets are real
    observed OSRAM values; no emotion labels or inferred missing values.
    """

    def __init__(self, heads, key_dim, value_dim):
        super().__init__()
        hidden = 4 * key_dim
        self.w1 = nn.Parameter(torch.randn(heads, key_dim, hidden) * .02)
        self.b1 = nn.Parameter(torch.zeros(heads, hidden))
        self.w2 = nn.Parameter(torch.randn(heads, hidden, value_dim) * .02)
        self.b2 = nn.Parameter(torch.zeros(heads, value_dim))
        self.norm_weight = nn.Parameter(torch.ones(heads, value_dim))
        self.norm_bias = nn.Parameter(torch.zeros(heads, value_dim))
        self.skip = nn.Identity() if key_dim == value_dim else nn.Linear(key_dim, value_dim, bias=False)
        self.lr = nn.Linear(key_dim, heads)
        self.heads = heads

    def initial(self, x):
        return tuple(p.unsqueeze(0).expand(x.shape[0], *p.shape) for p in (self.w1, self.b1, self.w2, self.b2))

    @staticmethod
    def gelu_prime(x):
        a = math.sqrt(2 / math.pi)
        u = a * (x + .044715 * x.pow(3))
        return .5 * (1 + u.tanh()) + .5 * x * (1 - u.tanh().square()) * a * (1 + 3 * .044715 * x.square())

    def predict(self, state, x):
        w1, b1, w2, b2 = state
        z1 = torch.einsum("bhk,bhkd->bhd", x, w1) + b1
        h = F.gelu(z1, approximate="tanh")
        z2 = torch.einsum("bhd,bhdv->bhv", h, w2) + b2
        mean = z2.mean(-1, keepdim=True)
        inv = (z2.var(-1, unbiased=False, keepdim=True) + 1e-6).rsqrt()
        normalized = (z2 - mean) * inv
        result = self.skip(x) + normalized * self.norm_weight + self.norm_bias
        return result, (z1, h, normalized, inv)

    def step(self, state, key, value, observed):
        prediction, (z1, h, norm, inv) = self.predict(state, key)
        # Derivative of half squared L2 reconstruction and LayerNorm.
        g = (prediction - value) * self.norm_weight
        g2 = inv * (g - g.mean(-1, keepdim=True) - norm * (g * norm).mean(-1, keepdim=True))
        g1 = torch.einsum("bhv,bhdv->bhd", g2, state[2]) * self.gelu_prime(z1)
        gradients = (key[..., :, None] * g1[..., None, :], g1,
                     h[..., :, None] * g2[..., None, :], g2)
        eta = self.lr(key).diagonal(dim1=-2, dim2=-1).sigmoid() / key.shape[-1]
        result = []
        for weight, gradient in zip(state, gradients):
            rate = eta.reshape(*eta.shape, *((1,) * (weight.ndim - eta.ndim)))
            mask = observed.reshape(observed.shape[0], *((1,) * (weight.ndim - 1)))
            result.append(torch.where(mask, weight - rate * gradient, weight))
        return tuple(result)


class DynamicsScan(nn.Module):
    def __init__(self, method, num_heads=8, key_dim=64, value_dim=64, latent_dim=256):
        super().__init__()
        self.method, self.num_heads, self.key_dim, self.value_dim = method, num_heads, key_dim, value_dim
        d = max(8, latent_dim // num_heads)
        slot = key_dim + value_dim
        if method == "C05":
            self.core = HierarchicalMultiscaleRNN(3 * slot + 3, d); summary = 3 * d
        elif method == "C06":
            self.core = NeuralCDE(key_dim, value_dim, d); summary = d
        elif method == "C07":
            self.core = RecurrentIndependentMechanisms(slot, d); summary = 4 * d
        elif method == "C08":
            self.core = TTTMLP(num_heads, key_dim, value_dim); summary = None
        else:
            raise ValueError(f"Unknown dynamics method: {method}")
        if summary is not None:
            self.read_state = nn.Linear(summary, value_dim, bias=False)
            self.read_query = nn.Linear(key_dim, value_dim)

    def _read(self, state, q):
        if self.method == "C08":
            return self.core.predict(state, q)[0]
        return self.read_state(self.core.summary(state)) * self.read_query(q).sigmoid()

    def scan(self, keys, values, queries, availability, valid, *, address_residual=None):
        length, batch = valid.shape
        if queries.shape != (length, batch, 4, self.num_heads, self.key_dim):
            raise ValueError("queries must have shape [L,B,4,H,K]")
        if availability.shape != (length, batch, 3):
            raise ValueError("availability must have shape [L,B,3]")
        valid = valid.bool()
        available = availability.bool() & valid[..., None]
        clean_k = torch.stack([torch.where(available[..., i, None, None], keys[n], torch.zeros_like(keys[n])) for i, n in enumerate(MODALITIES)], -2)
        clean_v = torch.stack([torch.where(available[..., i, None, None], values[n], torch.zeros_like(values[n])) for i, n in enumerate(MODALITIES)], -2)
        clean_q = torch.where(valid[..., None, None, None], queries, torch.zeros_like(queries))
        slots = torch.cat((clean_k, clean_v), -1)
        av_heads = available[:, :, None, :].expand(-1, -1, self.num_heads, -1)
        x = torch.cat((slots.flatten(-2), av_heads.to(queries.dtype)), -1)
        if self.method == "C07":
            state = self.core.initial(slots[0] if length else queries.new_zeros(batch, self.num_heads, 3, self.key_dim + self.value_dim))
        else:
            state = self.core.initial(x[0] if length else queries.new_zeros(batch, self.num_heads, x.shape[-1]))
        seen = torch.zeros(batch, device=queries.device, dtype=torch.bool)
        index_time = queries.new_zeros(batch)
        base, gaps = [], []
        diagnostics = {n: {m: [] for m in ("rho", "eta", "cosine")} for n in MODALITIES}
        for t in range(length):
            mask = valid[t, :, None, None] & seen[:, None, None]
            base.append(torch.where(mask, self._read(state, clean_q[t, :, 0]), torch.zeros_like(clean_v[t, :, :, 0])).flatten(1))
            gap = []
            for i, name in enumerate(MODALITIES):
                q = clean_q[t, :, i + 1]
                r = address_residual(clean_k[t].transpose(-2, -1), q) if address_residual is not None else q
                ratio = r.norm(dim=-1) / q.norm(dim=-1).clamp_min(1e-8)
                cosine = F.cosine_similarity(q, r, dim=-1)
                dm = valid[t, :, None].expand(-1, self.num_heads)
                for metric, data in (("rho", ratio), ("eta", 1 - cosine * ratio), ("cosine", cosine)):
                    diagnostics[name][metric].extend(data[dm].detach().cpu().tolist())
                read = self._read(state, r)
                gm = mask & ~available[t, :, i, None, None]
                gap.append(torch.where(gm, read, torch.zeros_like(read)).flatten(1))
            gaps.append(torch.stack(gap, 1))
            observed = available[t].any(-1)
            obs_heads = observed[:, None].expand(-1, self.num_heads)
            index_time = index_time + valid[t].to(queries.dtype)
            if self.method == "C05":
                state = self.core.step(state, x[t], obs_heads)
            elif self.method == "C06":
                state = self.core.step(state, x[t], av_heads[t], index_time[:, None].expand(-1, self.num_heads))
            elif self.method == "C07":
                state = self.core.step(state, slots[t], av_heads[t])
            else:
                for i in range(3):
                    state = self.core.step(state, clean_k[t, :, :, i], clean_v[t, :, :, i], available[t, :, i])
            seen = seen | observed
        self.last_state = tuple(s.detach() for s in state)
        if not length:
            return queries.new_zeros(0, batch, self.num_heads * self.value_dim), queries.new_zeros(0, batch, 3, self.num_heads * self.value_dim), diagnostics
        return torch.stack(base), torch.stack(gaps), diagnostics

    forward = scan


def build(method, num_heads=8, key_dim=64, value_dim=64, latent_dim=256):
    aliases = {"hierarchical_multiscale_rnn": "C05", "neural_cde": "C06",
               "recurrent_independent_mechanisms": "C07", "ttt_mlp": "C08"}
    return DynamicsScan(aliases.get(method, method), num_heads, key_dim, value_dim, latent_dim)
