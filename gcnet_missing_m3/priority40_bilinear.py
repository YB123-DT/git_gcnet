"""M06--M10 feature cores; the caller owns the zero-start Flat residual.

Primary method evidence inspected 2026-10-04:
TFN: https://aclanthology.org/D17-1115.pdf, tensor fusion section.
LMF: https://aclanthology.org/P18-1209.pdf, section 3.2.
MFB: https://arxiv.org/pdf/1708.01471, equations 3--4 and normalization.
Tucker: https://arxiv.org/pdf/1705.06676, section 3.2 equation 5 ONLY;
  intentionally does not claim the additional MUTAN section 3.3 constraint.
Tucker/BLOCK: https://arxiv.org/pdf/1902.00038, equations 8--9 and section 3.
Implementation cross-checks (equations independently implemented here):
https://github.com/Justin1904/TensorFusionNetworks/blob/master/model.py
  TFN.forward (reference implementation, not asserted original-author code).
https://github.com/Justin1904/Low-rank-Multimodal-Fusion/blob/master/model.py
  LMF.forward (author code, independent modality factors).
https://github.com/yuzcccc/vqa-mfb/blob/7fab8dddca5924ed1023149795cdaeacf029ff34/mfb_baseline/train_mfb_baseline.py
  mfb_baseline lines 80--92 (original MFB author code).
https://github.com/Cadene/block.bootstrap.pytorch/blob/master/block/models/networks/fusions/fusions.py
  Block, Tucker, MFB (BLOCK authors' implementation, not original MFB authors).

Five fixed roles: Local, Base, Gap-a, Gap-t, Gap-v. TFN/LMF extend the
source arity to these five roles. Other methods pair Local with the ordered
concatenation of all four histories, NEVER their mean. Neutral missing factors
are an explicit adaptation only for TFN/LMF, not feature completion. Tucker
uses a full 32x32xD core, not MUTAN's additional low-rank core constraint.
No labels, auxiliary losses, memory operations, dropout or dependencies.
"""
import torch
from torch import nn
from torch.nn import functional as F


METHODS = ('m06_tfn', 'm07_lmf', 'm08_mfb', 'm09_tucker', 'm10_block')


def _mask(value, active):
    while active.ndim < value.ndim:
        active = active.unsqueeze(-1)
    return torch.where(active, value, torch.zeros_like(value))


def _normalize(value):
    # Smooth signed sqrt has a finite derivative at zero (unlike sign*sqrt).
    powered = value / torch.sqrt(value.abs() + 1e-8)
    return F.normalize(powered, p=2, dim=-1, eps=1e-8)


class _Core(nn.Module):
    def __init__(self, dim):
        super().__init__()
        if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
            raise ValueError('dim must be a positive integer')
        self.dim = dim

    def forward(self, tokens, active):
        if (tokens.ndim != 3 or tokens.shape[1:] != (5, self.dim)
                or active.shape != tokens.shape[:2] or active.dtype != torch.bool
                or active.device != tokens.device or not tokens.is_floating_point()):
            raise ValueError('expected floating [N,5,D] and boolean [N,5] on same device')
        clean = _mask(tokens, active)
        if tokens.shape[0] == 0:
            return clean[:, 0]
        result = self.interact(clean, active)
        return _mask(result, self.usable(active))

    def usable(self, active):
        return active[:, 0] & active[:, 1:].any(-1)


class TFN(_Core):
    def __init__(self, dim):
        super().__init__(dim)
        self.projections = nn.ModuleList([nn.Linear(dim, 4) for _ in range(5)])
        self.output = nn.Linear(5 ** 5, dim)

    def usable(self, active):
        return active.any(-1)

    def augmented_roles(self, tokens, active):
        vectors = torch.stack([layer(tokens[:, role]) for role, layer in
                               enumerate(self.projections)], 1)
        vectors = _mask(vectors, active)  # Remove projection bias for absent roles.
        return torch.cat((torch.ones_like(vectors[..., :1]), vectors), -1)

    def interact(self, tokens, active):
        roles = self.augmented_roles(tokens, active)
        joint = roles[:, 0]
        for role in range(1, 5):
            joint = (joint.unsqueeze(-1) * roles[:, role, None]).flatten(1)
        return self.output(joint)


class LMF(_Core):
    rank = 4

    def __init__(self, dim):
        super().__init__(dim)
        # Linear bias is the factor multiplying the source's appended constant 1.
        self.factors = nn.ModuleList([nn.Linear(dim, self.rank * dim) for _ in range(5)])

    def usable(self, active):
        return active.any(-1)

    def interact(self, tokens, active):
        joint = tokens.new_ones(tokens.shape[0], self.rank, self.dim)
        for role, layer in enumerate(self.factors):
            factor = layer(tokens[:, role]).reshape(-1, self.rank, self.dim)
            factor = torch.where(active[:, role, None, None], factor, torch.ones_like(factor))
            joint = joint * factor
        return joint.sum(1)


class MFB(_Core):
    factor = 4

    def __init__(self, dim):
        super().__init__(dim)
        self.local = nn.Linear(dim, dim * self.factor)
        self.history = nn.Linear(4 * dim, dim * self.factor)

    def interact(self, tokens, active):
        left = _mask(self.local(tokens[:, 0]), active[:, 0])
        right = _mask(self.history(tokens[:, 1:].flatten(1)), active[:, 1:].any(-1))
        pooled = (left * right).reshape(-1, self.dim, self.factor).sum(-1)
        return _normalize(pooled)


class Tucker(_Core):
    def __init__(self, dim):
        super().__init__(dim)
        self.local = nn.Linear(dim, 32)
        self.history = nn.Linear(4 * dim, 32)
        self.core = nn.Parameter(torch.empty(32, 32, dim))
        nn.init.normal_(self.core, std=1 / 32)

    def interact(self, tokens, active):
        left = _mask(self.local(tokens[:, 0]), active[:, 0])
        right = _mask(self.history(tokens[:, 1:].flatten(1)), active[:, 1:].any(-1))
        return torch.einsum('ni,ijk,nj->nk', left, self.core, right)


class BLOCK(_Core):
    rank = 4

    def __init__(self, dim):
        super().__init__(dim)
        if dim % 4:
            raise ValueError('BLOCK dim must be divisible by four')
        self.chunk_dim = dim // 4
        self.local = nn.Linear(dim, dim)
        self.history = nn.Linear(4 * dim, dim)
        self.left_blocks = nn.ModuleList([nn.Linear(self.chunk_dim, self.rank * self.chunk_dim)
                                         for _ in range(4)])
        self.right_blocks = nn.ModuleList([nn.Linear(self.chunk_dim, self.rank * self.chunk_dim)
                                          for _ in range(4)])

    def interact(self, tokens, active):
        has_history = active[:, 1:].any(-1)
        left = _mask(self.local(tokens[:, 0]), active[:, 0]).chunk(4, -1)
        right = _mask(self.history(tokens[:, 1:].flatten(1)), has_history).chunk(4, -1)
        blocks = []
        for index, (lp, rp) in enumerate(zip(self.left_blocks, self.right_blocks)):
            a = _mask(lp(left[index]), active[:, 0])
            b = _mask(rp(right[index]), has_history)
            block = (a * b).reshape(-1, self.rank, self.chunk_dim).sum(1)
            blocks.append(_normalize(block))
        return torch.cat(blocks, -1)


def build(method, dim=64):
    constructors = dict(zip(METHODS, (TFN, LMF, MFB, Tucker, BLOCK)))
    if method not in constructors:
        raise ValueError('Unknown priority-40 bilinear method: ' + str(method))
    return constructors[method](dim)
