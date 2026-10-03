"""Task-only TabNet/NODE core adaptations; source details in feature_reasoning.json.

Independent tensor implementations of sparse selection and tree path mixtures.
No source preprocessing, auxiliary loss, batch statistics or additional data pass.
"""
import math
import torch
from torch import nn
from .meaningful_blocks_common import safe_mask


def sparse_simplex(scores, allowed=None, alpha=1.5):
    """Exact masked sparsemax (alpha=1) or entmax1.5 along the last axis."""
    if allowed is None:
        allowed = torch.ones_like(scores, dtype=torch.bool)
    allowed = allowed.expand_as(scores).bool()
    if not bool(allowed.any(-1).all()):
        raise ValueError('Sparse selection needs at least one eligible coordinate')
    masked = scores.masked_fill(~allowed, -torch.inf)
    centered = scores - masked.max(-1, keepdim=True).values
    x = centered if alpha == 1. else centered * .5
    ordered, indices = x.masked_fill(~allowed, -torch.inf).sort(-1, descending=True)
    eligible = allowed.gather(-1, indices)
    ordered = safe_mask(ordered, eligible)
    rank = torch.arange(1, scores.shape[-1]+1, device=scores.device, dtype=scores.dtype)
    cumulative = ordered.cumsum(-1)
    if alpha == 1.:
        support = eligible & (1 + rank*ordered > cumulative)
        count = support.sum(-1, keepdim=True).clamp_min(1)
        threshold = (cumulative.gather(-1, count-1)-1)/count
        return safe_mask((x-threshold).clamp_min(0), allowed)
    if alpha != 1.5:
        raise ValueError('Only sparsemax and entmax1.5 are supported')
    mean = cumulative/rank
    variance_sum = ordered.square().cumsum(-1) - rank*mean.square()
    delta = (1-variance_sum)/rank
    # Non-selected negative-discriminant ranks have no gradient contribution.
    # A positive floor avoids 0 * infinite sqrt derivative in autograd.
    thresholds = mean - delta.clamp_min(torch.finfo(scores.dtype).tiny).sqrt()
    count = (eligible & (thresholds <= ordered)).sum(-1, keepdim=True).clamp_min(1)
    threshold = thresholds.gather(-1, count-1)
    return safe_mask((x-threshold).clamp_min(0).square(), allowed)


class FixedSlots(nn.Module):
    def __init__(self, latent_dim, num_heads, value_dim):
        super().__init__()
        self.local_norm = nn.LayerNorm(latent_dim)
        self.memory_norm = nn.LayerNorm(num_heads*value_dim)
        self.local = nn.Linear(latent_dim, 64, bias=False)
        self.memory = nn.Linear(num_heads*value_dim, 128, bias=False)

    def forward(self, local, evidence, active, availability):
        valid = active.any(-1)
        l = safe_mask(self.local(self.local_norm(safe_mask(local, valid))), valid)
        m = safe_mask(self.memory(self.memory_norm(safe_mask(evidence, active))), active)
        x = torch.cat((l, m.flatten(1), safe_mask(availability, valid)), -1)
        eligible = torch.cat((valid[:,None].expand(-1,64),
                              active[...,None].expand(-1,-1,128).flatten(1),
                              valid[:,None].expand(-1,3)), -1)
        return x, eligible


class GLU(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 256, bias=False)
        self.norm = nn.LayerNorm(256)

    def forward(self, x):
        value, gate = self.norm(self.linear(x)).chunk(2, -1)
        return value * gate.sigmoid()


class TabNetEvidence(nn.Module):
    output_dim = 64

    def __init__(self, latent_dim, num_heads, value_dim):
        super().__init__()
        self.slots = FixedSlots(latent_dim, num_heads, value_dim)
        self.shared = nn.ModuleList([GLU(579), GLU(128)])
        self.specific = nn.ModuleList([nn.ModuleList([GLU(128), GLU(128)]) for _ in range(4)])
        self.attention = nn.ModuleList([nn.Sequential(nn.Linear(64,579,bias=False), nn.LayerNorm(579)) for _ in range(3)])

    def transform(self, x, step):
        h = self.shared[0](x)
        h = (h+self.shared[1](h))*math.sqrt(.5)
        for layer in self.specific[step]:
            h = (h+layer(h))*math.sqrt(.5)
        return h

    def forward(self, local, evidence, active, availability):
        x, eligible = self.slots(local, evidence, active, availability)
        valid = active.any(-1)
        result = x.new_zeros((len(x),64))
        if not valid.any():
            return result
        x, eligible = x[valid], eligible[valid]
        prior = eligible.to(x.dtype)
        attention = self.transform(x,0)[:,64:]
        decision = x.new_zeros((len(x),64))
        for step in range(3):
            mask = sparse_simplex(prior*self.attention[step](attention), eligible, alpha=1.)
            features = self.transform(mask*x, step+1)
            decision = decision + features[:,:64].relu()
            attention = features[:,64:]
            prior = prior*(1.5-mask)
        result[valid] = decision
        return result


class ObliviousTrees(nn.Module):
    """32 depth-four trees; choices shared by availability pattern, not by label."""
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.choices = nn.Parameter(torch.empty(32,4,input_dim).uniform_(0,1))
        self.threshold = nn.Parameter(torch.zeros(32,4))
        self.log_temperature = nn.Parameter(torch.zeros(32,4))
        self.response = nn.Parameter(torch.randn(32,16,2))
        self.register_buffer('initialized', torch.tensor(False))
        self.register_buffer('init_seed', torch.tensor((torch.initial_seed()+input_dim) % (2**63-1)))
        self.register_buffer('leaf_bits', ((torch.arange(16)[None] >> torch.arange(4)[:,None]) & 1).bool())

    def feature_values(self, x, eligible):
        values = x.new_zeros((len(x),32,4))
        patterns, ids = torch.unique(eligible, dim=0, return_inverse=True)
        for i, pattern in enumerate(patterns):
            rows = (ids==i).nonzero(as_tuple=True)[0]
            weights = sparse_simplex(self.choices, pattern[None,None,:], alpha=1.5)
            values[rows] = torch.einsum('nf,tdf->ntd', x[rows], weights)
        return values

    @torch.no_grad()
    def initialize(self, values):
        generator = torch.Generator(device='cpu').manual_seed(int(self.init_seed))
        quantiles = torch.rand((32,4), generator=generator, dtype=torch.float64).to(values)
        ordered = values.detach().sort(0).values
        positions = quantiles*(len(values)-1)
        low = positions.floor().long()
        high = positions.ceil().long()
        flat = ordered.flatten(1)
        indices = torch.arange(128, device=values.device)
        lo = flat[low.flatten(), indices].reshape(32,4)
        hi = flat[high.flatten(), indices].reshape(32,4)
        self.threshold.copy_(lo+(hi-lo)*(positions-low))
        self.log_temperature.copy_((values.detach()-self.threshold).abs().amax(0).add(1e-6).log())
        self.initialized.fill_(True)

    def forward(self, x, eligible):
        values = self.feature_values(x, eligible)
        if self.training and not bool(self.initialized) and len(x):
            self.initialize(values)
        logits = (values-self.threshold)*(-self.log_temperature).exp()
        # entmoid15(x) is the first probability of entmax15([x,0]).
        positive = sparse_simplex(torch.stack((logits,torch.zeros_like(logits)), -1))[...,0]
        matches = torch.where(self.leaf_bits[None,None], 1-positive[...,None], positive[...,None])
        paths = matches.prod(2)
        return torch.einsum('ntl,tlc->ntc', paths, self.response).flatten(1)


class NODEEvidence(nn.Module):
    output_dim = 192

    def __init__(self, latent_dim, num_heads, value_dim):
        super().__init__()
        self.slots = FixedSlots(latent_dim,num_heads,value_dim)
        self.layers = nn.ModuleList([ObliviousTrees(579+64*i) for i in range(3)])

    def forward(self, local,evidence,active,availability):
        x, eligible = self.slots(local,evidence,active,availability)
        valid = active.any(-1)
        result = x.new_zeros((len(x),192))
        if not valid.any():
            return result
        x, eligible = x[valid], eligible[valid]
        outputs = []
        for layer in self.layers:
            h = layer(x,eligible)
            outputs.append(h)
            x = torch.cat((x,h), -1)
            eligible = torch.cat((eligible,torch.ones_like(h,dtype=torch.bool)), -1)
        result[valid] = torch.cat(outputs,-1)
        return result


def build_feature_reasoning(method, latent_dim, num_heads, value_dim):
    cls = {'tabnet': TabNetEvidence, 'node': NODEEvidence}.get(method)
    if cls is None:
        raise ValueError('Unknown feature-reasoning method: '+method)
    return cls(latent_dim,num_heads,value_dim)
