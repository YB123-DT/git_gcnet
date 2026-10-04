"""Structured evidence-input adaptations, not full-paper reproductions.

Source cards: experiments/osram_meaningful20_round2_20261004/structured_candidates.json.
NSF: four alternating RQ coupling steps, eight bins, support [-3, 3].
LRPCA: rank one, detached deterministic SVD initializer, fourteen scaled updates.
DEQ: nonconservative contractive map and *implicit* adjoint. Solver amendment:
independent Picard convergence replaces Broyden in both solves (50 steps, 1e-6).
There is no unrolled-gradient fallback, batch stopping statistic, auxiliary loss,
persistent forward state, random forward sampling, or added output bridge.

Independent equation implementations; upstream MIT code was inspected, not copied.
Run this file for the bounded CPU smoke and numerical-solver checks.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def _safe_evidence(evidence, active):
    return torch.where(active.bool().unsqueeze(-1), evidence, 0.)


def _context(local, active, availability, heads):
    valid = active.bool().any(-1, keepdim=True)
    local = torch.where(valid, local, 0.)
    availability = torch.where(valid, availability.to(local), 0.)
    eye = torch.eye(heads, dtype=local.dtype, device=local.device)
    return torch.cat((local[:, None].expand(-1, heads, -1),
                      availability[:, None].expand(-1, heads, -1),
                      eye[None].expand(local.shape[0], -1, -1)), -1)


def _rational_quadratic(x, parameters, bins=8, bound=3.):
    """RQ spline in displacement form, giving exact zero-logit identity.

The derivative logits are centered at inverse_softplus(1 - min_derivative).
Subtracting the constant softplus value before adding one is algebraically
the source parametrization but avoids initialization roundoff away from one.
"""
    minimum = .001
    width_logits, height_logits, derivative_logits = torch.split(
        parameters, [bins, bins, bins - 1], -1)
    widths = minimum + (1 - bins * minimum) * F.softmax(width_logits, -1)
    heights = minimum + (1 - bins * minimum) * F.softmax(height_logits, -1)
    x_knots = F.pad(widths.cumsum(-1), (1, 0)) * (2 * bound) - bound
    y_knots = F.pad(heights.cumsum(-1), (1, 0)) * (2 * bound) - bound
    # Set endpoints explicitly; cumsum roundoff must not move the tail boundary.
    endpoints = x.new_tensor([-bound, bound])
    x_knots = torch.cat((endpoints[:1].expand_as(x_knots[..., :1]),
                         x_knots[..., 1:-1], endpoints[1:].expand_as(x_knots[..., :1])), -1)
    y_knots = torch.cat((endpoints[:1].expand_as(y_knots[..., :1]),
                         y_knots[..., 1:-1], endpoints[1:].expand_as(y_knots[..., :1])), -1)
    derivative_bias = x.new_tensor(math.log(math.expm1(1 - minimum)))
    derivatives = 1 + (F.softplus(derivative_logits + derivative_bias)
                       - F.softplus(derivative_bias))
    derivatives = F.pad(derivatives, (1, 1), value=1.)
    inside = (x >= -bound) & (x <= bound)
    interior_x = torch.where(inside, x, 0.)
    index = (interior_x[..., None] >= x_knots[..., 1:]).sum(-1).clamp(max=bins-1)

    def pick(values, offset=0):
        return values.gather(-1, (index + offset)[..., None]).squeeze(-1)

    x0, y0 = pick(x_knots), pick(y_knots)
    width = pick(x_knots, 1) - x0
    height = pick(y_knots, 1) - y0
    slope = height / width
    left, right = pick(derivatives), pick(derivatives, 1)
    theta = (interior_x - x0) / width
    curve = left + right - 2 * slope
    denominator = slope + curve * theta * (1 - theta)
    displacement = ((y0 - x0) + (height - width) * theta
                    + height * theta * (1 - theta)
                    * ((left - slope) - theta * curve) / denominator)
    return x + torch.where(inside, displacement, 0.)


class _SplineCoupling(nn.Module):
    def __init__(self, local_dim, heads, values):
        super().__init__()
        self.heads, self.values = heads, values
        width = 4 * values
        # Even value_dim gives exactly the per-role alternating split in the card.
        if values % 2:
            raise ValueError('Spline coupling requires an even value_dim')
        self.register_buffer('even', torch.arange(width)[torch.arange(width) % 2 == 0])
        self.register_buffer('odd', torch.arange(width)[torch.arange(width) % 2 == 1])
        context_dim = local_dim + 3 + heads
        self.conditioners = nn.ModuleList([
            nn.Sequential(nn.Linear(width // 2 + context_dim, 128), nn.GELU(),
                          nn.Linear(128, 128), nn.GELU(),
                          nn.Linear(128, (width // 2) * 23)) for _ in range(4)])
        for net in self.conditioners:
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)

    def forward(self, local, evidence, active, availability):
        evidence = _safe_evidence(evidence, active)
        n = evidence.shape[0]
        if n == 0:
            return local, evidence
        x = evidence.reshape(n, 4, self.heads, self.values).transpose(1, 2).reshape(n, self.heads, -1)
        mask = active.bool()[:, None, :, None].expand(n, self.heads, 4, self.values).reshape_as(x)
        context = _context(local, active, availability, self.heads)
        for step, net in enumerate(self.conditioners):
            keep, change = (self.even, self.odd) if step % 2 == 0 else (self.odd, self.even)
            params = net(torch.cat((x[..., keep], context), -1)).reshape(n, self.heads, -1, 23)
            # Source coupling scales width and height logits by sqrt(hidden width).
            params = torch.cat((params[..., :16] / math.sqrt(128), params[..., 16:]), -1)
            transformed = _rational_quadratic(x[..., change], params)
            changed = x.clone()
            changed[..., change] = transformed
            x = torch.where(mask, changed, 0.)
        out = x.reshape(n, self.heads, 4, self.values).transpose(1, 2).reshape_as(evidence)
        return local, _safe_evidence(out, active)


class _LearnedRobustPCA(nn.Module):
    def __init__(self):
        super().__init__()
        thresholds = torch.tensor([.1 * .8 ** t for t in range(1, 15)])
        self.threshold_logits = nn.Parameter(torch.log(torch.expm1(thresholds)))
        self.step_logits = nn.Parameter(torch.zeros(14))
        self.gamma = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _threshold(x, threshold):
        return x.sign() * F.relu(x.abs() - threshold)

    def _decompose(self, y):
        # Shape [group_batch, value_coordinates, active_roles]. Reductions never
        # cross the group batch; grouping only amortizes SVD and kernel launches.
        scale = y.abs().mean(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        with torch.no_grad():
            initial = y - self._threshold(y, .1 * scale)
            left, singular, right_t = torch.linalg.svd(initial, full_matrices=False)
            root = singular[:, :1, None].sqrt()
            u = left[..., :1] * root
            v = right_t[:, :1].transpose(-2, -1) * root
        thresholds = F.softplus(self.threshold_logits)
        for coefficient, step in zip(thresholds, self.step_logits.sigmoid()):
            threshold = coefficient * scale
            residual = y - u @ v.transpose(-2, -1)
            sparse = self._threshold(residual, threshold)
            error = residual - sparse
            # Simultaneous updates: neither RHS sees the new factor.
            next_u = u + step * (error @ v) / v.square().sum((-2, -1), keepdim=True).clamp_min(1e-6)
            next_v = v + step * (error.transpose(-2, -1) @ u) / u.square().sum((-2, -1), keepdim=True).clamp_min(1e-6)
            u, v = next_u, next_v
        return self._threshold(y - u @ v.transpose(-2, -1), thresholds[-1] * scale)

    def forward(self, local, evidence, active, availability):
        del availability
        evidence = _safe_evidence(evidence, active)
        active = active.bool()
        out = evidence
        for present in torch.unique(active, dim=0):
            indices = present.nonzero(as_tuple=True)[0]
            if indices.numel() <= 1:
                continue
            rows = (active == present).all(-1).nonzero(as_tuple=True)[0]
            group = evidence.index_select(0, rows)
            y = group.index_select(1, indices).transpose(-2, -1)
            sparse = self._decompose(y)
            transformed = y - self.gamma.tanh() * sparse
            changed = group.index_copy(1, indices, transformed.transpose(-2, -1))
            out = out.index_copy(0, rows, changed)
        return local, _safe_evidence(out, active)


def _converge(function, initial, max_steps=50, tolerance=1e-6):
    """Independent Picard solves; no batch-global convergence or silent fallback."""
    state = initial
    done = torch.zeros(state.shape[0], dtype=torch.bool, device=state.device)
    for _ in range(max_steps):
        candidate = function(state)
        residual = (candidate - state).norm(dim=-1) / (1 + candidate.norm(dim=-1))
        if not torch.isfinite(candidate).all():
            raise RuntimeError('DEQ nonfinite fixed-point iterate')
        state = torch.where(done[:, None], state, candidate)
        done = done | (residual <= tolerance)
        if done.all():
            break
    residual = (function(state) - state).norm(dim=-1) / (1 + state.norm(dim=-1))
    if not torch.isfinite(residual).all() or (residual > tolerance).any():
        raise RuntimeError('DEQ failed to converge within its fixed solve budget')
    return state


def _equilibrium_map(z, x, context, mask, p, q, c, bias):
    p_scaled = p / (1 + torch.linalg.vector_norm(p))
    q_scaled = q / (1 + torch.linalg.vector_norm(q))
    hidden = F.linear(z * mask, q_scaled) + F.linear(context, c, bias)
    return x + .9 * mask * F.linear(hidden.tanh(), p_scaled)


class _ImplicitEquilibrium(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, context, mask, p, q, c, bias, max_steps, tolerance):
        z = _converge(lambda state: _equilibrium_map(state, x, context, mask, p, q, c, bias),
                      x, max_steps, tolerance)
        ctx.save_for_backward(z, x, context, mask, p, q, c, bias)
        ctx.max_steps, ctx.tolerance = max_steps, tolerance
        return z

    @staticmethod
    def backward(ctx, upstream):
        z, x, context, mask, p, q, c, bias = ctx.saved_tensors
        with torch.enable_grad():
            # A single local graph at the equilibrium; do not backpropagate the
            # root-finding trajectory or retain state across forwards.
            z = z.detach().requires_grad_()
            inputs = [value.detach().requires_grad_() for value in (x, context, p, q, c, bias)]
            f = _equilibrium_map(z, inputs[0], inputs[1], mask, *inputs[2:])

            def adjoint_map(v):
                jtv = torch.autograd.grad(f, z, v, retain_graph=True)[0]
                return jtv + upstream

            adjoint = _converge(adjoint_map, torch.zeros_like(upstream),
                                ctx.max_steps, ctx.tolerance)
            grads = torch.autograd.grad(f, inputs, adjoint)
        return grads[0], grads[1], None, *grads[2:], None, None


class _DeepEquilibrium(nn.Module):
    def __init__(self, local_dim, heads, values):
        super().__init__()
        self.heads, self.values = heads, values
        width = 4 * values
        self.p = nn.Parameter(torch.zeros(width, width))
        self.q = nn.Parameter(torch.empty(width, width))
        self.c = nn.Parameter(torch.empty(width, local_dim + 3 + heads))
        self.bias = nn.Parameter(torch.zeros(width))
        nn.init.xavier_uniform_(self.q)
        nn.init.xavier_uniform_(self.c)
        self.max_steps, self.tolerance = 50, 1e-6

    def forward(self, local, evidence, active, availability):
        evidence = _safe_evidence(evidence, active)
        n = evidence.shape[0]
        if n == 0:
            return local, evidence
        width = 4 * self.values
        x = evidence.reshape(n, 4, self.heads, self.values).transpose(1, 2).reshape(-1, width)
        mask = active.bool()[:, None, :, None].expand(n, self.heads, 4, self.values).reshape_as(x).to(x)
        context = _context(local, active, availability, self.heads).reshape(n * self.heads, -1)
        z = _ImplicitEquilibrium.apply(x, context, mask, self.p, self.q, self.c, self.bias,
                                       self.max_steps, self.tolerance)
        out = z.reshape(n, self.heads, 4, self.values).transpose(1, 2).reshape_as(evidence)
        return local, _safe_evidence(out, active)


def build_structured(method, latent_dim=256, num_heads=8, value_dim=64):
    """Construct a task-only input transform without changing caller CPU RNG."""
    if min(latent_dim, num_heads, value_dim) <= 0:
        raise ValueError('Structured dimensions must be positive')
    with torch.random.fork_rng(devices=[]):
        if method == 'conditional_rational_quadratic_spline_coupling':
            return _SplineCoupling(latent_dim, num_heads, value_dim)
        if method == 'learned_robust_pca_evidence_decomposition':
            return _LearnedRobustPCA()
        if method == 'contractive_deep_equilibrium_evidence':
            return _DeepEquilibrium(latent_dim, num_heads, value_dim)
    raise ValueError('Unknown structured input method: ' + str(method))


def _smoke_test():
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(91)
    local = torch.randn(4, 12)
    evidence = torch.randn(4, 4, 8)
    active = torch.tensor([[1, 1, 0, 0], [1, 0, 1, 1],
                           [1, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.bool)
    availability = torch.tensor([[0, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
    expected = torch.where(active[..., None], evidence, 0.)
    poison = torch.where(active[..., None], evidence, float('nan'))
    for method in ('conditional_rational_quadratic_spline_coupling',
                   'learned_robust_pca_evidence_decomposition',
                   'contractive_deep_equilibrium_evidence'):
        assert 'build_structured' in globals(), 'Missing structured implementation'
        model = build_structured(method, 12, 2, 4)
        rng = torch.random.get_rng_state().clone()
        lo, out = model(local, poison, active, availability)
        torch.testing.assert_close(lo, local, rtol=0, atol=0)
        torch.testing.assert_close(out, expected, rtol=2e-6, atol=2e-6)
        assert torch.equal(rng, torch.random.get_rng_state()), 'Forward changed RNG'
        loss = (out * torch.randn_like(out)).sum()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum() > 0 for g in grads), method
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.add_(p.grad, alpha=-1e-3)
        model.zero_grad(set_to_none=True)
        _, moved = model(local, poison, active, availability)
        assert torch.isfinite(moved).all()
        assert (moved - expected).abs().max() > 1e-8, method
        for i in range(4):
            _, one = model(local[i:i+1], poison[i:i+1], active[i:i+1], availability[i:i+1])
            torch.testing.assert_close(one[0], moved[i], atol=3e-6, rtol=3e-6)
        moved.square().sum().backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        print(method + ': identity, inactive poison, RNG, update, batch independence PASS')


def _solver_numerical_test():
    # Closed-form linear roots and adjoints independently check solve orientation.
    matrix = torch.tensor([[.2, .1], [-.1, .3]], dtype=torch.double)
    rhs = torch.tensor([[1., -2.], [.3, .7]], dtype=torch.double)
    root = _converge(lambda z: z @ matrix.T + rhs, torch.zeros_like(rhs), 100, 1e-12)
    expected = torch.linalg.solve(torch.eye(2, dtype=torch.double) - matrix, rhs.T).T
    torch.testing.assert_close(root, expected, atol=1e-10, rtol=1e-10)
    adjoint = _converge(lambda v: v @ matrix + rhs, torch.zeros_like(rhs), 100, 1e-12)
    expected_adjoint = torch.linalg.solve(torch.eye(2, dtype=torch.double) - matrix.T, rhs.T).T
    torch.testing.assert_close(adjoint, expected_adjoint, atol=1e-10, rtol=1e-10)
    try:
        _converge(lambda z: z + 1, torch.zeros(1, 2), 3, 1e-6)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Nonconvergent solve must fail')
    # Central differences verify the custom *implicit* parameter/input backward,
    # not merely the standalone transpose solve, at a nonidentity equilibrium.
    x = torch.randn(2, 4, dtype=torch.double, requires_grad=True)
    context = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
    mask = torch.tensor([[1., 1., 0., 0.], [1., 1., 1., 1.]], dtype=torch.double)
    x = (x * mask).detach().requires_grad_()
    p = (torch.randn(4, 4, dtype=torch.double) * .15).requires_grad_()
    q = (torch.randn(4, 4, dtype=torch.double) * .15).requires_grad_()
    c = (torch.randn(4, 3, dtype=torch.double) * .15).requires_grad_()
    bias = torch.zeros(4, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda xx, cc, pp, qq, ww, bb: _ImplicitEquilibrium.apply(
            xx, cc, mask, pp, qq, ww, bb, 100, 1e-12),
        (x, context, p, q, c, bias), atol=2e-5, rtol=2e-4)
    print('DEQ closed-form roots/adjoints, nonconvergence failure, implicit gradcheck PASS')


if __name__ == '__main__':
    _smoke_test()
    _solver_numerical_test()
