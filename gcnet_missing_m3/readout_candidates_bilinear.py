"""Six independently implemented, paper-grounded bilinear readout operators.

Sources: experiments/osram_readout20_20261003/literature/bilinear.json.
These are Local/evidence adaptations, not reproductions of complete VQA models.
The caller owns typed input projections and the zero-start Flat residual.
Input order is Local, Base, Gap-a, Gap-t, Gap-v. No memory state is accessed.

Fixed settings: MLB rank D; MFB factor4; MUTAN rank4 with tanh latent
projections; BLOCK four chunks and rank4; MCB sketch1024; BAN one Local query,
one glimpse and rank D. No dropout or random operation occurs in forward.
Power normalization uses z/sqrt(abs(z)+1e-8), a continuous finite-gradient
approximation to signed square root, followed by epsilon-safe L2 normalization.
"""

import torch
from torch import nn
from torch.nn import functional as F


def _mask(x, active):
    return torch.where(active.unsqueeze(-1), x, torch.zeros_like(x))


def _power_normalize(x):
    value = x / torch.sqrt(x.abs() + 1e-8)
    return F.normalize(value, p=2, dim=-1, eps=1e-8)


class _Bilinear(nn.Module):
    def __init__(self, dim):
        super().__init__()
        if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
            raise ValueError('dim must be a positive integer')
        self.dim = dim

    def forward(self, tokens, active):
        if (tokens.ndim != 3 or tokens.shape[1:] != (5, self.dim)
                or active.shape != tokens.shape[:2] or active.dtype != torch.bool
                or not tokens.is_floating_point() or active.device != tokens.device):
            raise ValueError('expected floating tokens [N,5,D] and boolean active [N,5]')
        safe = _mask(tokens, active)
        if tokens.shape[0] == 0:
            return safe[:, 0]
        history = safe[:, 1:]
        context = history.sum(1) / active[:, 1:].sum(1, keepdim=True).clamp_min(1)
        result = self.interact(safe[:, 0], context, history, active[:, 1:])
        usable = active[:, 0] & active[:, 1:].any(1)
        return _mask(result, usable)


class MLB(_Bilinear):
    def __init__(self, dim):
        super().__init__(dim)
        self.local = nn.Linear(dim, dim)
        self.context = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def interact(self, local, context, history, active):
        return self.out(torch.tanh(self.local(local)) * torch.tanh(self.context(context)))


class MFB(_Bilinear):
    def __init__(self, dim):
        super().__init__(dim)
        self.local = nn.Linear(dim, 4 * dim)
        self.context = nn.Linear(dim, 4 * dim)
        self.out = nn.Linear(dim, dim)

    def interact(self, local, context, history, active):
        products = self.local(local) * self.context(context)
        pooled = products.reshape(local.shape[0], self.dim, 4).sum(-1)
        return self.out(_power_normalize(pooled))


class MUTAN(_Bilinear):
    def __init__(self, dim):
        super().__init__(dim)
        self.local = nn.Linear(dim, dim)
        self.context = nn.Linear(dim, dim)
        self.local_factors = nn.ModuleList([nn.Linear(dim, dim) for _ in range(4)])
        self.context_factors = nn.ModuleList([nn.Linear(dim, dim) for _ in range(4)])
        self.out = nn.Linear(dim, dim)

    def interact(self, local, context, history, active):
        x, y = torch.tanh(self.local(local)), torch.tanh(self.context(context))
        fused = sum(left(x) * right(y)
                    for left, right in zip(self.local_factors, self.context_factors))
        return self.out(fused)


class BLOCK(_Bilinear):
    def __init__(self, dim):
        super().__init__(dim)
        if dim % 4:
            raise ValueError('BLOCK dim must be divisible by four chunks')
        self.chunk_dim = dim // 4
        self.local = nn.Linear(dim, dim)
        self.context = nn.Linear(dim, dim)
        self.local_blocks = nn.ModuleList([nn.Linear(self.chunk_dim, 4 * self.chunk_dim) for _ in range(4)])
        self.context_blocks = nn.ModuleList([nn.Linear(self.chunk_dim, 4 * self.chunk_dim) for _ in range(4)])
        self.out = nn.Linear(dim, dim)

    def interact(self, local, context, history, active):
        x, y = self.local(local).chunk(4, -1), self.context(context).chunk(4, -1)
        blocks = []
        for i, (left, right) in enumerate(zip(self.local_blocks, self.context_blocks)):
            products = left(x[i]) * right(y[i])
            pooled = products.reshape(local.shape[0], 4, self.chunk_dim).sum(1)
            blocks.append(_power_normalize(pooled))
        return self.out(torch.cat(blocks, -1))


class MCB(_Bilinear):
    def __init__(self, dim):
        super().__init__(dim)
        self.sketch_dim = 1024
        sketch_rng = torch.Generator(device='cpu').manual_seed(1729)
        for name in ('local', 'context'):
            self.register_buffer(name + '_hash', torch.randint(1024, (dim,), generator=sketch_rng))
            signs = torch.randint(2, (dim,), generator=sketch_rng).float().mul(2).sub(1)
            self.register_buffer(name + '_sign', signs)
        self.out = nn.Linear(self.sketch_dim, dim)

    def _sketch(self, values, indices, signs):
        return values.new_zeros((values.shape[0], self.sketch_dim)).scatter_add(
            1, indices.expand(values.shape[0], -1), values * signs)

    def interact(self, local, context, history, active):
        # Float16 FFT is not portable; keep double precision for CPU equation tests.
        compute_dtype = torch.float64 if local.dtype == torch.float64 else torch.float32
        x = self._sketch(local.to(compute_dtype), self.local_hash, self.local_sign.to(compute_dtype))
        y = self._sketch(context.to(compute_dtype), self.context_hash, self.context_sign.to(compute_dtype))
        product = torch.fft.rfft(x, n=self.sketch_dim) * torch.fft.rfft(y, n=self.sketch_dim)
        convolved = torch.fft.irfft(product, n=self.sketch_dim)
        return self.out(_power_normalize(convolved).to(local.dtype))


class BAN(_Bilinear):
    def __init__(self, dim):
        super().__init__(dim)
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.score = nn.Linear(dim, 1, bias=False)
        self.value_local = nn.Linear(dim, dim)
        self.value_context = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def interact(self, local, context, history, active):
        query = F.relu(self.query(local))[:, None]
        keys = _mask(F.relu(self.key(history)), active)
        scores = self.score(query * keys).squeeze(-1)
        scores = scores.masked_fill(~active, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(active, weights, torch.zeros_like(weights))
        values = _mask(F.relu(self.value_context(history)), active)
        joint = F.relu(self.value_local(local))[:, None] * values
        return self.out((weights[..., None] * joint).sum(1))


def build_bilinear(method: str, dim: int = 128) -> nn.Module:
    constructors = {'mlb': MLB, 'mfb': MFB, 'mutan': MUTAN,
                    'block': BLOCK, 'mcb': MCB, 'ban': BAN}
    if method not in constructors:
        raise ValueError(f'Unknown bilinear readout candidate: {method}')
    return constructors[method](dim)
