"""Independent paper-equation adaptations for five current readout tokens.

Input order: Local, Base, Gap-a, Gap-t, Gap-v; output [N, D]. These are
mechanism adaptations, not complete recommendation/vision reproductions.
Sources and code/license caveats: experiments/osram_readout20_20261003/
literature/interactions.json. No source code is vendored.

Fixed settings: DCNv2 two rank-32 cross layers; CIN two 16-map layers;
AutoInt two layers/two heads (paper linear QKV, unscaled attention); DIN
4D->80->40->1 sigmoid MLP, unnormalized scores (unlike author demo's
softmax); DLRM ten distinct pair dots plus Local; AFF channel bottleneck
D//4 and paper convex blend (not author code's factor 2); Non-local one
signed dot-product block, bottleneck D//2, active-count normalization.

AFF replaces image-local/global paths with per-utterance channel transforms
of Local+pooled history and masked all-token mean. No BatchNorm or cross-
example statistics are used. Common input projections, outer residual and
zero-initialized output projection belong to the caller. Constructors use
normal parameter initialization; forward consumes no random numbers.
"""

import torch
from torch import nn


def _mask(x, active):
    return torch.where(active.unsqueeze(-1), x, torch.zeros_like(x))


def _mean(x, active):
    return _mask(x, active).sum(1) / active.sum(1, keepdim=True).clamp_min(1)


class _Interaction(nn.Module):
    def __init__(self, dim):
        super().__init__()
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError('dim must be a positive integer')
        self.dim = dim

    def forward(self, tokens, active):
        if (tokens.ndim != 3 or tokens.shape[1:] != (5, self.dim)
                or active.shape != tokens.shape[:2] or active.dtype != torch.bool):
            raise ValueError('expected tokens [N,5,D] and boolean active [N,5]')
        result = self.interact(_mask(tokens, active), active)
        return _mask(result, active.any(1))


class DCNv2(_Interaction):
    def __init__(self, dim):
        super().__init__(dim)
        self.down = nn.ModuleList([nn.Linear(5 * dim, 32, bias=False) for _ in range(2)])
        self.up = nn.ModuleList([nn.Linear(32, 5 * dim) for _ in range(2)])
        self.out = nn.Linear(5 * dim, dim)

    def interact(self, tokens, active):
        x0 = tokens.flatten(1)
        x = x0
        for down, up in zip(self.down, self.up):
            x = x + x0 * up(down(x))
        return self.out(x)


class CIN(_Interaction):
    def __init__(self, dim):
        super().__init__(dim)
        self.weights = nn.ParameterList([nn.Parameter(torch.empty(16, width, 5)) for width in (5, 16)])
        for weight in self.weights:
            nn.init.xavier_uniform_(weight)
        self.out = nn.Linear(32, dim)

    def interact(self, tokens, active):
        hidden = tokens
        summaries = []
        for weight in self.weights:
            hidden = torch.einsum('hij,nid,njd->nhd', weight, hidden, tokens)
            summaries.append(hidden.sum(-1))
        return self.out(torch.cat(summaries, -1))


class _AutoIntLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.residual = nn.Linear(dim, dim)

    def forward(self, tokens, active):
        # Remask after every biased projection: inactive keys/values cannot
        # enter even an all-inactive row's finite softmax fallback.
        n, slots, dim = tokens.shape
        q, k, v = [_mask(proj(tokens), active).reshape(n, slots, 2, dim // 2).transpose(1, 2)
                   for proj in (self.query, self.key, self.value)]
        score = q @ k.transpose(-1, -2)
        keys = active[:, None, None, :]
        score = score.masked_fill(~keys, torch.finfo(score.dtype).min)
        weight = torch.where(keys, torch.softmax(score, -1), torch.zeros_like(score))
        context = (weight @ v).transpose(1, 2).reshape(n, slots, dim)
        return _mask(torch.relu(context + self.residual(tokens)), active)


class AutoInt(_Interaction):
    def __init__(self, dim):
        super().__init__(dim)
        if dim % 2:
            raise ValueError('AutoInt requires even dim for its two heads')
        self.layers = nn.ModuleList([_AutoIntLayer(dim) for _ in range(2)])

    def interact(self, tokens, active):
        for layer in self.layers:
            tokens = layer(tokens, active)
        return _mean(tokens, active)


class DIN(_Interaction):
    def __init__(self, dim):
        super().__init__(dim)
        self.score = nn.Sequential(nn.Linear(4 * dim, 80), nn.Sigmoid(),
                                   nn.Linear(80, 40), nn.Sigmoid(), nn.Linear(40, 1))
        self.out = nn.Linear(2 * dim, dim)

    def interact(self, tokens, active):
        query = tokens[:, :1].expand(-1, 4, -1)
        history = tokens[:, 1:]
        features = torch.cat((query, history, query - history, query * history), -1)
        weight = _mask(self.score(features), active[:, 1:])
        pooled = (weight * history).sum(1)
        return self.out(torch.cat((tokens[:, 0], pooled), -1))


class DLRM(_Interaction):
    def __init__(self, dim):
        super().__init__(dim)
        self.register_buffer('pairs', torch.triu_indices(5, 5, offset=1), persistent=False)
        self.out = nn.Linear(dim + 10, dim)

    def interact(self, tokens, active):
        dot = tokens @ tokens.transpose(1, 2)
        return self.out(torch.cat((tokens[:, 0], dot[:, self.pairs[0], self.pairs[1]]), -1))


class AFF(_Interaction):
    def __init__(self, dim):
        super().__init__(dim)
        width = max(1, dim // 4)
        self.local = nn.Sequential(nn.Linear(dim, width), nn.ReLU(), nn.Linear(width, dim))
        self.global_context = nn.Sequential(nn.Linear(dim, width), nn.ReLU(), nn.Linear(width, dim))

    def interact(self, tokens, active):
        current = tokens[:, 0]
        history = _mean(tokens[:, 1:], active[:, 1:])
        weight = torch.sigmoid(self.local(current + history) + self.global_context(_mean(tokens, active)))
        return weight * current + (1 - weight) * history


class NonLocal(_Interaction):
    def __init__(self, dim):
        super().__init__(dim)
        width = max(1, dim // 2)
        self.query = nn.Linear(dim, width)
        self.key = nn.Linear(dim, width)
        self.value = nn.Linear(dim, width)
        self.output = nn.Linear(width, dim)

    def interact(self, tokens, active):
        q, k, v = [_mask(proj(tokens), active) for proj in (self.query, self.key, self.value)]
        count = active.sum(1).clamp_min(1).reshape(-1, 1, 1)
        context = ((q @ k.transpose(1, 2)) @ v) / count
        return _mean(_mask(tokens + self.output(context), active), active)


def build_interaction(method, dim=128):
    """Build one fixed candidate, with no changes to caller-owned Memory/head."""
    classes = {'dcnv2': DCNv2, 'cin': CIN, 'autoint': AutoInt, 'din': DIN,
               'dlrm': DLRM, 'aff': AFF, 'nonlocal': NonLocal}
    if method not in classes:
        raise ValueError(f'unknown interaction method: {method}')
    return classes[method](dim)
