"""Current-role structured reasoning transfers, not paper reproductions.

Only Local/Base/three Gap slots enter these layers. All reasoning states are
local to forward; the caller's causal memory and objective are untouched.
"""
import torch
from torch import nn

from .meaningful_blocks_common import safe_mask
from .meaningful_input_new40 import zero_linear


class _RoleAdapter(nn.Module):
    def __init__(self, core, latent_dim, num_heads, value_dim):
        super().__init__()
        self.core = core
        width = num_heads * value_dim
        dim = core.dim
        self.encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(size, dim), nn.LayerNorm(dim))
            for size in [latent_dim] + [width] * 4
        ])
        self.decoders = nn.ModuleList([
            zero_linear(dim, size) for size in [latent_dim] + [width] * 4
        ])

    def forward(self, local, evidence, active, availability):
        active = active.bool()
        clean = safe_mask(evidence, active)
        mask = torch.cat((torch.ones_like(active[:, :1]), active), dim=1)
        raw = [local] + list(clean.unbind(1))
        x = torch.stack([encoder(v) for encoder, v in zip(self.encoders, raw)], 1)
        x = safe_mask(x, mask)
        result = self.core(x, mask)
        delta = [decoder(result[:, i]) for i, decoder in enumerate(self.decoders)]
        return (local + delta[0].to(local.dtype),
                safe_mask(clean + torch.stack(delta[1:], 1).to(evidence.dtype), active))


class NeuralBellmanFord(nn.Module):
    """Source-indexed path states, typed extension and repeated boundary.

Source: NBFNet Algorithm 1 / equations 5--6; author nbfnet/model.py and
layer.py. Sum is one of the source's supported generalized aggregators.
"""
    def __init__(self, dim=64, steps=3):
        super().__init__()
        self.dim = dim
        self.query = nn.Linear(dim, dim)
        self.edge = nn.Linear(2 * dim, 1)
        self.relations = nn.ModuleList([nn.Linear(dim, 25 * dim) for _ in range(steps)])
        self.updates = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * dim, dim), nn.ReLU()) for _ in range(steps)
        ])
        self.readout = nn.Linear(5 * dim, dim)

    def forward(self, x, mask):
        batch, count, dim = x.shape
        pair = torch.cat((x[:, :, None].expand(-1, -1, count, -1),
                          x[:, None, :].expand(-1, count, -1, -1)), -1)
        edges = torch.sigmoid(self.edge(pair)).squeeze(-1)
        valid_edges = mask[:, :, None] & mask[:, None, :]
        valid_edges = valid_edges & ~torch.eye(count, dtype=torch.bool, device=x.device)[None]
        edges = torch.where(valid_edges, edges, torch.zeros_like(edges))
        query = safe_mask(self.query(x), mask)
        boundary = query[:, :, None] * torch.eye(count, device=x.device, dtype=x.dtype)[None, :, :, None]
        valid_pairs = mask[:, :, None] & mask[:, None, :]
        h = boundary
        for relation, update in zip(self.relations, self.updates):
            # b,s,i,j,d: source s extends the state at i over typed edge i->j.
            r = relation(query).reshape(batch, count, count, count, dim)
            messages = h[:, :, :, None] * r * edges[:, None, :, :, None]
            aggregate = messages.sum(2) + boundary
            h = safe_mask(update(torch.cat((h, aggregate), -1)), valid_pairs)
        per_destination = h.transpose(1, 2).reshape(batch, count, count * dim)
        return safe_mask(self.readout(per_destination), mask)


class NeuralLogicProgramming(nn.Module):
    """NeuralLP equations 6--11 including operator AND proof-depth attention."""
    def __init__(self, dim=64, steps=3, relations=4):
        super().__init__()
        self.dim, self.steps = dim, steps
        self.relation_scores = nn.Sequential(nn.Linear(2 * dim, dim), nn.Tanh(), nn.Linear(dim, relations))
        self.controller = nn.LSTMCell(dim, dim)
        self.operator = nn.Linear(dim, 2 * relations)
        self.end = nn.Parameter(torch.zeros(dim))
        self.readout = nn.Linear(2 * dim, dim)

    def forward(self, x, mask):
        batch, count, dim = x.shape
        pair = torch.cat((x[:, :, None].expand(-1, -1, count, -1),
                          x[:, None, :].expand(-1, count, -1, -1)), -1)
        matrices = torch.sigmoid(self.relation_scores(pair)).permute(0, 3, 1, 2)
        valid = mask[:, None, :, None] & mask[:, None, None, :]
        matrices = torch.where(valid, matrices, torch.zeros_like(matrices))
        matrices = torch.cat((matrices, matrices.transpose(-1, -2)), 1)
        h, c = torch.zeros_like(x), torch.zeros_like(x)
        seed = torch.eye(count, dtype=x.dtype, device=x.device)[None].expand(batch, -1, -1)
        seed = safe_mask(seed, mask)
        memories, keys = [seed], [h]
        for step in range(self.steps + 1):
            controller_input = x if step < self.steps else self.end.expand_as(x)
            h_flat, c_flat = self.controller(controller_input.reshape(-1, dim),
                                             (h.reshape(-1, dim), c.reshape(-1, dim)))
            h, c = h_flat.reshape_as(x), c_flat.reshape_as(x)
            scores = (torch.stack(keys, 2) * h[:, :, None]).sum(-1)
            attention = scores.softmax(-1)
            selected = (torch.stack(memories, 2) * attention[..., None]).sum(2)
            if step == self.steps:
                distribution = selected
                break
            operator = self.operator(h).softmax(-1)
            # Each source maintains a distribution over destination roles.
            next_state = torch.einsum('bsr,brij,bsi->bsj', operator, matrices, selected)
            next_state = next_state / next_state.sum(-1, keepdim=True).clamp_min(1e-8)
            memories.append(safe_mask(next_state, mask))
            keys.append(h)
        retrieved = distribution @ x
        return safe_mask(self.readout(torch.cat((x, retrieved), -1)), mask)


def _game_kkt(payoff, u, v):
    batch, count, _ = payoff.shape
    matrix = payoff.new_zeros(batch, 2 * count + 2, 2 * count + 2)
    matrix[:, :count, :count] = torch.diag_embed(u.reciprocal())
    matrix[:, :count, count:2 * count] = payoff
    matrix[:, count:2 * count, :count] = payoff.transpose(-1, -2)
    matrix[:, count:2 * count, count:2 * count] = -torch.diag_embed(v.reciprocal())
    matrix[:, :count, -2] = matrix[:, -2, :count] = 1
    matrix[:, count:2 * count, -1] = matrix[:, -1, count:2 * count] = 1
    return matrix


class _EntropyGame(torch.autograd.Function):
    """QRE primal-dual Newton solve and implicit KKT derivative (paper eq. 5--9)."""
    @staticmethod
    def forward(ctx, payoff):
        p = payoff.double()
        batch, count, _ = p.shape
        u = p.new_full((batch, count), 1 / count)
        v = u.clone()
        mu = -(p @ v[..., None]).squeeze(-1).mean(-1, keepdim=True) - u.log().mean(-1, keepdim=True) - 1
        nu = -(p.transpose(-1, -2) @ u[..., None]).squeeze(-1).mean(-1, keepdim=True) + v.log().mean(-1, keepdim=True) + 1

        def residual(a, b, m, n):
            return torch.cat(((p @ b[..., None]).squeeze(-1) + a.log() + 1 + m,
                              (p.transpose(-1, -2) @ a[..., None]).squeeze(-1) - b.log() - 1 + n,
                              a.sum(-1, keepdim=True) - 1, b.sum(-1, keepdim=True) - 1), -1)

        for _ in range(40):
            r = residual(u, v, mu, nu)
            if r.abs().max().item() < 1e-9:
                break
            direction = torch.linalg.solve(_game_kkt(p, u, v), -r[..., None]).squeeze(-1)
            du, dv = direction[:, :count], direction[:, count:2 * count]
            step = p.new_ones(batch, 1)
            old_norm = r.square().sum(-1, keepdim=True)
            for _ in range(40):
                a, b = u + step * du, v + step * dv
                positive = (a > 0).all(-1, keepdim=True) & (b > 0).all(-1, keepdim=True)
                trial = residual(a.clamp_min(1e-30), b.clamp_min(1e-30),
                                 mu + step * direction[:, -2:-1], nu + step * direction[:, -1:])
                accepted = positive & (trial.square().sum(-1, keepdim=True) <= (1 - 1e-4 * step) * old_norm + 1e-24)
                if accepted.all().item():
                    break
                step = torch.where(accepted, step, step * .5)
            u, v = u + step * du, v + step * dv
            mu, nu = mu + step * direction[:, -2:-1], nu + step * direction[:, -1:]
        if not torch.isfinite(u).all() or residual(u, v, mu, nu).abs().max().item() > 1e-7:
            raise RuntimeError('QRE Newton solve did not converge')
        ctx.save_for_backward(p, u, v)
        ctx.input_dtype = payoff.dtype
        return u.to(payoff.dtype), v.to(payoff.dtype)

    @staticmethod
    def backward(ctx, grad_u, grad_v):
        p, u, v = ctx.saved_tensors
        rhs = torch.cat((grad_u.double(), grad_v.double(), p.new_zeros(p.shape[0], 2)), -1)
        adjoint = torch.linalg.solve(_game_kkt(p, u, v), -rhs[..., None]).squeeze(-1)
        count = u.shape[-1]
        grad = adjoint[:, :count, None] * v[:, None, :] + u[:, :, None] * adjoint[:, None, count:2 * count]
        return grad.to(ctx.input_dtype)


class QuantalResponseGame(nn.Module):
    """Proposal/challenge role game; unique entropy-regularized saddle point.

    Source: IJCAI 2018 Learning Game-Theoretic Models from Aggregate Behavioral
    Data, equations 5--9; payoff_learning/src/core/{solve,paynet}.py.
    Costs are bounded squared disagreements; no auxiliary game supervision.
    """
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim
        self.proposal = nn.Linear(dim, dim)
        self.challenge = nn.Linear(dim, dim)
        self.readout = nn.Linear(3 * dim, dim)

    def forward(self, x, mask):
        proposal, challenge = self.proposal(x).tanh(), self.challenge(x).tanh()
        payoff = (proposal[:, :, None] - challenge[:, None, :]).square().mean(-1)
        strategy_u, strategy_v = torch.zeros_like(mask, dtype=x.dtype), torch.zeros_like(mask, dtype=x.dtype)
        # At most 16 current-role masks; omit inactive actions from the simplex.
        for pattern in torch.unique(mask, dim=0):
            rows = (mask == pattern).all(-1).nonzero(as_tuple=True)[0]
            columns = pattern.nonzero(as_tuple=True)[0]
            subgame = payoff[rows][:, columns][:, :, columns]
            u, v = _EntropyGame.apply(subgame)
            strategy_u[rows[:, None], columns[None]] = u
            strategy_v[rows[:, None], columns[None]] = v
        pooled_u = (proposal * strategy_u[..., None]).sum(1)
        pooled_v = (challenge * strategy_v[..., None]).sum(1)
        return safe_mask(self.readout(torch.cat((x, pooled_u[:, None].expand_as(x),
                                                pooled_v[:, None].expand_as(x)), -1)), mask)


def build(method, latent_dim=256, num_heads=8, value_dim=64):
    cores = {
        'survey80_struct_nbfnet': NeuralBellmanFord,
        'survey80_struct_neural_lp': NeuralLogicProgramming,
        'survey80_struct_qre_game': QuantalResponseGame,
    }
    if method not in cores:
        raise ValueError(f'Unknown structured method: {method}')
    return _RoleAdapter(cores[method](dim=value_dim), latent_dim, num_heads, value_dim)
