"""M16--20/M26--29 readouts over the five *existing* OSRAM roles.

These are small equation-level transfers, not full original-model reproductions.
Role encoding and the zero-initialized 1600-wide residual bridge belong to the
caller. No labels, auxiliary losses, running statistics or cross-example state.
"""
import itertools

import torch
from torch import nn
from torch.nn import functional as F

from .meaningful_blocks_common import safe_mask
from .meaningful_input_sorting import FeaturewiseSortPool


# Actual primary code inspected; paper links identify the transferred operator.
# ASP is paper-grounded only: no original-author code was located/claimed.
SOURCES = {
    'm16_deep_sets': ('https://arxiv.org/abs/1703.06114',
                     'https://github.com/manzilzaheer/DeepSets/blob/master/PopStats/model.py',
                     'Sec. 3.1; DeepSet.forward: shared phi, sum, rho'),
    'm17_janossy': ('https://arxiv.org/abs/1811.01900',
                   'https://github.com/PurdueMINDS/JanossyPooling/blob/master/arithmetic_tasks/models.py',
                   'Definition 2.1 exact permutation mean; ordered MLP instead of sampled RNN'),
    'm18_fspool': ('https://arxiv.org/abs/1906.02795',
                  'https://github.com/Cyanogenoid/fspool/blob/master/fspool.py',
                  'Sec. 4.1--4.2; FSPool.forward/determine_weight; four linear segments'),
    'm19_netvlad': ('https://arxiv.org/abs/1511.07247',
                   'https://github.com/Relja/netvlad/blob/master/layerVLAD.m',
                   'Sec. 3 Eq. 4; soft assignment and center-relative residuals, two L2 normalizations'),
    'm20_attentive_statistics': ('https://arxiv.org/abs/1803.10963', None,
                                'Sec. 3.2--3.3 Eq. 3--6; author-code evidence unavailable'),
    'm26_dolg': ('https://arxiv.org/abs/2108.02927',
                'https://github.com/feymanpriv/DOLG-paddle/blob/main/model/dolg_model.py',
                'DOLG.forward: orthogonal component, average, concat, fc; no spatial backbone'),
    'm27_isqrt_cov': ('https://arxiv.org/abs/1712.01034',
                     'https://github.com/jiangtaoxie/fast-MPN-COV/blob/master/src/representation/MPNCOV.py',
                     'Covpool/Sqrtm/Triuvec: trace normalization, coupled Newton-Schulz, rescale'),
    'm28_xcit_xca': ('https://arxiv.org/abs/2106.09681',
                    'https://github.com/facebookresearch/xcit/blob/main/xcit.py',
                    'XCA.forward: Q/K normalized across tokens; channel-channel softmax'),
    'm29_set_norm': ('https://arxiv.org/abs/2206.11925',
                    'https://github.com/rajesh-lab/deep_permutation_invariant/blob/main/models/norms.py',
                    'Sec. 4.2; SetNormL: joint valid-element/channel moments, shared channel affine'),
}

# Repository HEADs resolved during the 2026-10-04 source inspection above.
SOURCE_REVISIONS = {
    'manzilzaheer/DeepSets': '8b90fe08af08a5e3bb03b18e0ca162faa48ec4fd',
    'PurdueMINDS/JanossyPooling': '58e398d492be96c231a3f8c6684b519302c5c3d6',
    'Cyanogenoid/fspool': 'a9f93cc774610c6d96c2c3095a1ab16f53abbefb',
    'Relja/netvlad': '652dbe71aa45c691961ddd9f6cf902574e6bdc2f',
    'feymanpriv/DOLG-paddle': 'a5d28633400a7291a31c725f5c8d1efb50e118b8',
    'jiangtaoxie/fast-MPN-COV': 'c430c781264b7ce77536493308b09d73d534ce7a',
    'facebookresearch/xcit': '82f5291f412604970c39a912586e008ec009cdca',
    'rajesh-lab/deep_permutation_invariant': '7d25da12329d3d89a69c2f5333f94234248f01a5',
}


def _mlp(inputs, outputs):
    return nn.Sequential(nn.Linear(inputs, outputs), nn.GELU(), nn.Linear(outputs, outputs))


def _mean(x, mask):
    return safe_mask(x, mask).sum(1)/mask.sum(1, keepdim=True).clamp_min(1)


class _Readout(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim

    def forward(self, tokens, mask):
        if tokens.ndim != 3 or tokens.shape[1:] != (5, self.dim):
            raise ValueError('Expected [N,5,dim] existing-role tokens')
        if mask.shape != tokens.shape[:2]:
            raise ValueError('Expected [N,5] role mask')
        mask = mask.bool()
        clean = safe_mask(tokens, mask)  # Clear NaN/Inf BEFORE any learned map.
        if not tokens.shape[0]:
            return clean.sum(1)
        return safe_mask(self.pool_roles(clean, mask), mask.any(1))


class DeepSets(_Readout):
    def __init__(self, dim=64):
        super().__init__(dim)
        self.phi, self.rho = _mlp(dim, dim), _mlp(dim, dim)

    def pool_roles(self, x, mask):
        return self.rho(safe_mask(self.phi(x), mask).sum(1))


class Janossy(_Readout):
    """All n! valid-role permutations; zero-pad ordered MLP input to five.

    Four valid roles use all 24 permutations; five use all 120. Grouping only
    shares computation, never values/statistics across utterances. No sampling.
    """
    def __init__(self, dim=64):
        super().__init__(dim)
        self.ordered = _mlp(5*dim, dim)
        for n in range(1, 6):
            self.register_buffer('permutations_'+str(n),
                                 torch.tensor(list(itertools.permutations(range(n)))))

    def pool_roles(self, x, mask):
        counts = mask.sum(1)
        order = mask.long().argsort(dim=1, descending=True, stable=True)
        packed = x.gather(1, order[..., None].expand_as(x))
        output = x.sum(1)*0
        for n in range(1, 6):
            rows = (counts == n).nonzero(as_tuple=True)[0]
            permutation = getattr(self, 'permutations_'+str(n))
            for ids in rows.split(64):
                if not ids.numel():
                    continue
                sequences = packed[ids, :n][:, permutation]
                sequences = F.pad(sequences, (0, 0, 0, 5-n))
                pooled = self.ordered(sequences.flatten(-2)).mean(1)
                output = output.index_copy(0, ids, pooled)
        return output


class FSPool(_Readout):
    """Exact descending channel-sort, four-segment piecewise rank functional.

    Reuses the existing audited rank-weight interpolation, not its relaxed
    NeuralSort/unpool path. Packing eliminates artificial sentinel values.
    """
    def __init__(self, dim=64):
        super().__init__(dim)
        self.pool = FeaturewiseSortPool(channels=dim, pieces=4)

    def pool_roles(self, x, mask):
        counts = mask.sum(1)
        order = mask.long().argsort(dim=1, descending=True, stable=True)
        packed = x.gather(1, order[..., None].expand_as(x))
        output = x.sum(1)*0
        for n in range(1, 6):
            rows = (counts == n).nonzero(as_tuple=True)[0]
            values = packed[rows, :n].transpose(1, 2).sort(-1, descending=True).values
            pooled = (values*self.pool.rank_weights(n)).sum(-1)
            output = output.index_copy(0, rows, pooled)
        return output


class NetVLAD(_Readout):
    def __init__(self, dim=64):
        super().__init__(dim)
        self.assignment = nn.Linear(dim, 4)
        self.centers = nn.Parameter(torch.empty(4, dim))
        nn.init.normal_(self.centers, std=.02)
        self.output = nn.Linear(4*dim, dim)

    def pool_roles(self, x, mask):
        x = F.normalize(x, dim=-1, eps=1e-6)
        assignment = safe_mask(self.assignment(x).softmax(-1), mask)
        residual = assignment.transpose(1, 2) @ x - assignment.sum(1)[..., None]*self.centers
        descriptor = F.normalize(residual, dim=-1, eps=1e-6).flatten(1)
        descriptor = F.normalize(descriptor, dim=-1, eps=1e-6)
        return self.output(descriptor)


class AttentiveStatistics(_Readout):
    """Same scalar attention weights for mean and centered second moment."""
    def __init__(self, dim=64):
        super().__init__(dim)
        self.score = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, 1))
        self.output = nn.Linear(2*dim, dim)

    def pool_roles(self, x, mask):
        score = self.score(x).squeeze(-1)
        score = torch.where(mask, score, torch.full_like(score, torch.finfo(score.dtype).min))
        weights = safe_mask(score.softmax(-1), mask)
        weights = weights/weights.sum(-1, keepdim=True).clamp_min(1e-8)
        mean = (weights[..., None]*x).sum(1)
        centered = safe_mask(x-mean[:, None], mask)
        variance = (weights[..., None]*centered.square()).sum(1)
        std = variance.clamp_min(1e-8).sqrt()
        return self.output(torch.cat([mean, std], -1))


class DOLG(_Readout):
    """History orthogonal to projected Local, NOT an evidence reliability gate.

    OSRAM Local plays the global-anchor role; the four history vectors replace
    image-local descriptors. Only fusion is transferred, not the image backbone.
    """
    def __init__(self, dim=64):
        super().__init__(dim)
        self.anchor = nn.Linear(dim, dim)
        self.history = nn.Linear(dim, dim)
        self.output = nn.Linear(2*dim, dim)

    def pool_roles(self, x, mask):
        anchor = safe_mask(self.anchor(x[:, 0]), mask[:, 0])
        history = safe_mask(self.history(x[:, 1:]), mask[:, 1:])
        norm = anchor.square().sum(-1, keepdim=True).clamp_min(1e-8)
        coefficient = (history*anchor[:, None]).sum(-1, keepdim=True)/norm[:, None]
        orthogonal = safe_mask(history-coefficient*anchor[:, None], mask[:, 1:])
        return self.output(torch.cat([anchor, _mean(orthogonal, mask[:, 1:])], -1))


class ISqrtCov(_Readout):
    """16-D covariance, ridge 1e-4, five coupled Newton-Schulz iterations.

    Ridge handles <=5 points in 16 dimensions. Upper-triangle vectorization
    is followed by a learned 136->dim projection; no eigenvalue decomposition.
    """
    def __init__(self, dim=64):
        super().__init__(dim)
        self.projection = nn.Linear(dim, 16)
        self.output = nn.Linear(136, dim)
        self.register_buffer('eye', torch.eye(16))
        self.register_buffer('triangle', torch.triu_indices(16, 16))

    def pool_roles(self, x, mask):
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=x.device.type, enabled=False):
            projected = safe_mask(self.projection(x.to(dtype)), mask)
            centered = safe_mask(projected-_mean(projected, mask)[:, None], mask)
            covariance = centered.transpose(1, 2) @ centered
            covariance = covariance/mask.sum(1).clamp_min(1)[:, None, None]
            eye = self.eye.to(dtype)
            covariance = covariance+1e-4*eye
            trace = covariance.diagonal(dim1=-2, dim2=-1).sum(-1)[:, None, None]
            y, z = covariance/trace, eye.expand_as(covariance)
            for _ in range(5):
                correction = .5*(3*eye-z@y)
                y, z = y@correction, correction@z
            root = .5*(y+y.transpose(1, 2))*trace.sqrt()
            result = self.output(root[:, self.triangle[0], self.triangle[1]])
        return result.to(x.dtype)


class XCiTXCA(_Readout):
    """Four-head channel attention; normalize Q/K over active ROLE dimension."""
    def __init__(self, dim=64):
        super().__init__(dim)
        if dim % 4:
            raise ValueError('XCA dim must be divisible by four heads')
        self.qkv = nn.Linear(dim, 3*dim, bias=False)
        self.temperature = nn.Parameter(torch.ones(4, 1, 1))
        self.output = nn.Linear(dim, dim)

    def encode_roles(self, x, mask):
        """Retain safe per-role outputs before the legacy mean readout."""
        mask = mask.bool()
        x = safe_mask(x, mask)
        b = x.shape[0]
        qkv = safe_mask(self.qkv(x), mask).reshape(b, 5, 3, 4, self.dim//4)
        q, k, v = qkv.permute(2, 0, 3, 4, 1).unbind(0)
        q, k = F.normalize(q, dim=-1, eps=1e-6), F.normalize(k, dim=-1, eps=1e-6)
        attention = ((q @ k.transpose(-1, -2))*self.temperature).softmax(-1)
        updated = (attention @ v).permute(0, 3, 1, 2).reshape(b, 5, self.dim)
        return safe_mask(self.output(updated), mask)

    def pool_roles(self, x, mask):
        return _mean(self.encode_roles(x, mask), mask)


class SetNorm(_Readout):
    """Joint valid-set/channel standardization, then shared affine and encoder.

    This is NOT per-token LayerNorm or cross-batch BatchNorm. Only normalization
    plus a simple encoding/pooling readout is transferred, not all DeepSets++.
    """
    eps = 1e-5

    def __init__(self, dim=64):
        super().__init__(dim)
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.encoder = _mlp(dim, dim)
        self.output = nn.Linear(dim, dim)

    def normalize(self, x, mask):
        x = safe_mask(x, mask)
        count = (mask.sum(1)*self.dim).clamp_min(1)[:, None, None]
        mean = x.sum((1, 2), keepdim=True)/count
        centered = safe_mask(x-mean, mask)
        variance = centered.square().sum((1, 2), keepdim=True)/count
        normalized = centered/torch.sqrt(variance+self.eps)
        return safe_mask(normalized*self.weight+self.bias, mask)

    def pool_roles(self, x, mask):
        encoded = safe_mask(self.encoder(self.normalize(x, mask)), mask)
        return self.output(_mean(encoded, mask))


METHODS = {
    'm16_deep_sets': DeepSets, 'm17_janossy': Janossy,
    'm18_fspool': FSPool, 'm19_netvlad': NetVLAD,
    'm20_attentive_statistics': AttentiveStatistics, 'm26_dolg': DOLG,
    'm27_isqrt_cov': ISqrtCov, 'm28_xcit_xca': XCiTXCA,
    'm29_set_norm': SetNorm,
}


def build(method, dim=64):
    return METHODS[method](dim)
