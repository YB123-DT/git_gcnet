"""M21--M25 current-role interaction cores, not full paper reproductions.

The caller owns role projections, task loss and the zero-initialized outer
residual. No historical state, auxiliary objective or missing-role inference.

Method/source checks (2026-10-04):
* NFM: SIGIR 2017 section 3.1 equations 3--5; author's NeuralFM.py _init_graph,
  https://github.com/hexiangnan/neural_factorization_machine .
* DCNv2: arXiv:2008.13535 section 3.2 equation 1; official TensorFlow
  Recommenders layers/feature_interaction/dcn.py Cross.call (dense branch).
* CIN: arXiv:1803.05170 equations 6--7; Leavingseason/xDeepFM at
  114c4c45b1cb6144b2540f92a2b357c3f445e98e, exdeepfm/src/CIN.py,
  _build_extreme_FM direct=True, bias=False, identity activation branch.
* AFM: IJCAI 2017 equation 5; author's code/AFM.py pair construction and
  attention MLP. We follow the paper's PAIR softmax axis, not the released
  code's tf.nn.softmax on a trailing singleton dimension.
* KAN: arXiv:2404.19756 section 2.2 equations 2.5/2.10--2.12;
  KindXiaoming/pykan kan/spline.py B_batch/coef2curve. Reuse the audited local
  cubic edge layer: five fixed grid intervals, degree three, eight coefficients
  per edge, SiLU base. Its bounded-domain clipping is retained explicitly;
  no adaptive grid updates, pruning, symbolic fitting or regularization loss.
"""

import torch
from torch import nn

from .meaningful_new40_functions import CubicKANLayer
from .readout_candidates_interactions import CIN, _Interaction, _mask


class NFM(_Interaction):
    """Bi-interaction sum over distinct pairs, then a nonlinear hidden layer."""
    def __init__(self, dim=64):
        super().__init__(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def interact(self, tokens, active):
        pooled = .5 * (tokens.sum(1).square() - tokens.square().sum(1))
        return _mask(self.mlp(pooled), active.sum(1) >= 2)


class DenseDCNv2(_Interaction):
    """Two full-rank 5D-by-5D cross layers; no low-rank factorization."""
    def __init__(self, dim=64):
        super().__init__(dim)
        self.cross = nn.ModuleList([nn.Linear(5 * dim, 5 * dim) for _ in range(2)])
        self.out = nn.Linear(5 * dim, dim)

    def interact(self, tokens, active):
        x0 = tokens.flatten(1)
        hidden = x0
        for layer in self.cross:
            hidden = x0 * layer(hidden) + hidden
        return self.out(hidden)


class AFM(_Interaction):
    """A learned score for each valid Hadamard pair, not source attention."""
    def __init__(self, dim=64):
        super().__init__(dim)
        self.register_buffer('pairs', torch.triu_indices(5, 5, offset=1), persistent=False)
        self.attention = nn.Sequential(nn.Linear(dim, 32), nn.ReLU(), nn.Linear(32, 1, bias=False))
        self.out = nn.Linear(dim, dim)

    def interact(self, tokens, active):
        left, right = self.pairs
        valid = active[:, left] & active[:, right]
        products = _mask(tokens[:, left] * tokens[:, right], valid)
        scores = self.attention(products).squeeze(-1)
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        weights = torch.where(valid, scores.softmax(1), torch.zeros_like(scores))
        pooled = (weights[..., None] * products).sum(1)
        return _mask(self.out(pooled), valid.any(1))


class KAN(_Interaction):
    """Flattened role fusion to D, then cubic edge-function D->32->D."""
    def __init__(self, dim=64):
        super().__init__(dim)
        self.fusion = nn.Linear(5 * dim, dim)
        self.first = CubicKANLayer(dim, 32)
        self.second = CubicKANLayer(32, dim)

    def interact(self, tokens, active):
        fused = _mask(self.fusion(tokens.flatten(1)).tanh(), active.any(1))
        return self.second(self.first(fused))


def build(method, dim=64):
    """Return an independently initialized [N,5,D] -> [N,D] interaction."""
    classes = {'m21_nfm': NFM, 'm22_dcnv2': DenseDCNv2, 'm23_cin': CIN,
               'm24_afm': AFM, 'm25_kan': KAN}
    if method not in classes:
        raise ValueError(f'Unknown priority40 cross method: {method}')
    return classes[method](dim)
