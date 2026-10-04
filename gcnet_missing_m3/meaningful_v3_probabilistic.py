"""Five-role probabilistic/robust *transfers*, not original-paper reproductions.

Local, Base and three Gap vectors remain five roles (never 33 head tokens).
Only raw-slot residuals are returned. No loss, persistent filter state, cache,
calibration claim or change to the OSRAM memory is introduced.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .meaningful_blocks_common import safe_mask
from .meaningful_input_new40 import zero_linear


class _Roles(nn.Module):
    dim = 32

    def __init__(self, latent_dim, num_heads, value_dim, readout_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.evidence_dim = num_heads * value_dim
        self.local_projection = nn.Linear(latent_dim, self.dim)
        self.role_projections = nn.ModuleList(
            nn.Linear(self.evidence_dim, self.dim) for _ in range(4))
        self.bridge = zero_linear(readout_dim, latent_dim + 4 * self.evidence_dim)

    def forward(self, local, evidence, active, availability):
        active = active.bool()
        clean = safe_mask(evidence, active)
        rows = active.any(-1).nonzero(as_tuple=True)[0]
        if not rows.numel():
            return local, clean
        dtype = torch.float64 if local.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=local.device.type, enabled=False):
            x, e, mask = local[rows].to(dtype), clean[rows].to(dtype), active[rows]
            roles = torch.stack([self.local_projection(x)] + [
                projection(e[:, r]) for r, projection in enumerate(self.role_projections)], 1)
            present = torch.cat([torch.ones_like(mask[:, :1]), mask], 1)
            roles = safe_mask(roles, present)
            delta = self.bridge(self.readout(roles, present))
        dl, de = torch.zeros_like(local), torch.zeros_like(clean)
        dl[rows] = delta[:, :self.latent_dim].to(local.dtype)
        de[rows] = safe_mask(delta[:, self.latent_dim:].reshape(
            -1, 4, self.evidence_dim).to(evidence.dtype), active[rows])
        return local + dl, safe_mask(clean + de, active)


class ADF(_Roles):
    """Gast/Roth CVPR18 §3.2 Eq3–10, §5.2 SM+XE; contrib/adf.py.

    Fixed latent perturbation variance, affine W² variance propagation and
    analytic Gaussian-ReLU moments. Missing roles have zero mean AND variance.
    This smooths latent features; it does not estimate calibrated uncertainty.
    """
    def __init__(self, *dims):
        super().__init__(*dims, readout_dim=64)
        self.layers = nn.ModuleList([nn.Linear(5*self.dim, 64), nn.Linear(64, 64)])

    @staticmethod
    def relu_moments(mean, variance):
        std = variance.clamp_min(1e-12).sqrt()
        a = mean / std
        cdf = .5 * (1 + torch.erf(a / math.sqrt(2)))
        pdf = torch.exp(-.5*a.square()) / math.sqrt(2*math.pi)
        out = mean*cdf + std*pdf
        var = ((mean.square()+variance)*cdf + mean*std*pdf - out.square()).clamp_min(0)
        return out, var

    def readout(self, roles, present):
        mean = roles.flatten(1)
        variance = safe_mask(torch.full_like(roles, 1e-3), present).flatten(1)
        for layer in self.layers:
            mean = F.linear(mean, layer.weight, layer.bias)
            variance = F.linear(variance, layer.weight.square())
            mean, variance = self.relu_moments(mean, variance)
        return mean


class RKN(_Roles):
    """Becker et al. ICML19; LCAS/RKN RKNTransitionCell.py.

    State-conditioned mixture of banded transitions + upper/lower/cross
    covariance propagation + Kalman correction. Five roles are a fixed-order
    assimilation sequence, not physical time. Missing roles skip BOTH stages.
    """
    def __init__(self, *dims):
        super().__init__(*dims, readout_dim=5*self.dim)
        d = self.dim
        matrices = torch.zeros(4, 4, d, d)
        eye = torch.eye(d)
        matrices[:, 0] = eye
        matrices[:, 1] = .2*eye
        matrices[:, 2] = -.2*eye
        matrices[:, 3] = eye
        # Distinct initial bases keep mixture coefficients identifiable.
        matrices = matrices * torch.tensor([.97, .99, 1.01, 1.03])[:, None, None, None]
        self.transitions = nn.Parameter(matrices)
        index = torch.arange(d)
        self.register_buffer('band', (index[:, None]-index[None, :]).abs() <= 1)
        self.coefficients = nn.Linear(2*d, 4)
        self.observation = nn.Linear(d, d)
        self.observation_noise = nn.Linear(d, d)
        self.process_noise = nn.Parameter(torch.full((2, d), -4.))

    def readout(self, roles, present):
        b, _, d = roles.shape
        upper = roles.new_zeros(b, d)
        lower = torch.zeros_like(upper)
        u, l, s = torch.ones_like(upper)*10, torch.ones_like(upper)*10, torch.zeros_like(upper)
        noise = F.softplus(self.process_noise) + 1e-5
        for role in range(5):
            coef = self.coefficients(torch.cat([upper, lower], -1)).softmax(-1)
            a, c, f, g = torch.einsum('bk,kqij->bqij', coef,
                                     self.transitions*self.band).unbind(1)
            mv = lambda mat, vec: (mat @ vec.unsqueeze(-1)).squeeze(-1)
            diag = lambda left, vec, right: (left*right*vec[:, None, :]).sum(-1)
            pu, pl = mv(a, upper)+mv(c, lower), mv(f, upper)+mv(g, lower)
            vu = (diag(a,u,a)+2*diag(a,s,c)+diag(c,l,c)+noise[0]).clamp_min(1e-6)
            vl = (diag(f,u,f)+2*diag(f,s,g)+diag(g,l,g)+noise[1]).clamp_min(1e-6)
            vs = diag(f,u,a)+diag(g,s,a)+diag(f,s,c)+diag(g,l,c)
            # Preserve the per-coordinate PSD block under roundoff.
            bound = (vu*vl).sqrt()*(1-1e-6)
            vs = torch.maximum(torch.minimum(vs, bound), -bound)
            obs = self.observation(roles[:, role])
            r = F.softplus(self.observation_noise(roles[:, role])) + 1e-5
            ku, kl = vu/(vu+r), vs/(vu+r)
            residual = obs-pu
            valid = present[:, role, None]
            upper = torch.where(valid, pu+ku*residual, upper)
            lower = torch.where(valid, pl+kl*residual, lower)
            u = torch.where(valid, ((1-ku)*vu).clamp_min(1e-6), u)
            l = torch.where(valid, (vl-kl*vs).clamp_min(1e-6), l)
            s = torch.where(valid, (1-ku)*vs, s)
        return torch.cat([upper, lower, u, l, s], -1)


class GNCTLS(_Roles):
    """Yang et al. GNC-TLS; TEASER++ registration.cc GNCTLSRotationSolver.

    Transfer changes rotation fitting to the exact weighted-centre subproblem.
    Auxiliary TLS weights and increasing-mu continuation remain explicit;
    eight steps are a bounded approximation, not a convergence certificate.
    """
    def __init__(self, *dims):
        super().__init__(*dims, readout_dim=self.dim)
        self.threshold_squared = .25
        self.iterations = 8

    def readout(self, roles, present):
        roles = safe_mask(F.layer_norm(roles, (self.dim,)), present)
        weights = present.to(roles.dtype)
        center = (roles*weights[..., None]).sum(1)/weights.sum(1, keepdim=True)
        residual = (roles-center[:, None]).square().mean(-1)
        maximum = safe_mask(residual, present).amax(1, keepdim=True)
        c2 = self.threshold_squared
        denom = 2*maximum/c2-1
        easy = denom <= 0
        mu = 1/denom.clamp_min(1e-3)
        for _ in range(self.iterations):
            mass = weights.sum(1, keepdim=True)
            estimate = (roles*weights[..., None]).sum(1)/mass.clamp_min(1e-8)
            center = torch.where(mass > 1e-8, estimate, center)
            residual = (roles-center[:, None]).square().mean(-1)
            lower, upper = mu*c2/(mu+1), (mu+1)*c2/mu
            middle = (c2*mu*(mu+1)/residual.clamp_min(1e-8)).sqrt()-mu
            weights = torch.where(residual <= lower, 1.,
                                  torch.where(residual >= upper, 0., middle)).clamp(0, 1)
            weights = safe_mask(torch.where(easy, torch.ones_like(weights), weights), present)
            mu = mu*1.4
        mass = weights.sum(1, keepdim=True)
        return torch.where(mass > 1e-8,
                           (roles*weights[..., None]).sum(1)/mass.clamp_min(1e-8), center)


class PFRNN(_Roles):
    """Ma et al. AAAI20 PF-GRU; Yusufma03/pfrnns pfrnns.py.

    Deterministic quadrature transfer: normal-quantile particle innovations,
    systematic CDF ancestors at fixed midpoint, and w/q importance correction.
    Both train/eval consume NO RNG. Particle-axis normalization replaces source
    BatchNorm to avoid cross-utterance state. Only weighted-mean task CE path is
    retained, not the optional particle ELBO. Discrete ancestors have no gradient.
    """
    particles = 8

    def __init__(self, *dims):
        super().__init__(*dims, readout_dim=self.dim)
        d, k = self.dim, self.particles
        self.action = nn.Linear(d, d)
        self.observation = nn.Linear(d, d)
        self.gates = nn.Linear(2*d, 2*d)
        self.proposal = nn.Linear(2*d, 2*d)
        self.likelihood = nn.Linear(2*d, 1)
        quantiles = (torch.arange(k, dtype=torch.float32)+.5)/k
        normal = math.sqrt(2)*torch.erfinv(2*quantiles-1)
        # Dimension/step shifts avoid identical coordinates; no random draws.
        grid = (torch.arange(k)[:, None]+torch.arange(d)[None, :]) % k
        self.register_buffer('innovations', normal[grid])
        self.register_buffer('positions', quantiles)

    def readout(self, roles, present):
        b, _, d = roles.shape
        k = self.particles
        h = roles.new_zeros(b, k, d)
        logw = roles.new_full((b, k), -math.log(k))
        for role in range(5):
            act = F.leaky_relu(self.action(roles[:, role]))[:, None].expand(-1, k, -1)
            obs = F.leaky_relu(self.observation(roles[:, role]))[:, None].expand(-1, k, -1)
            z, r = self.gates(torch.cat([h, act], -1)).sigmoid().chunk(2, -1)
            mean, scale = self.proposal(torch.cat([r*h, act], -1)).chunk(2, -1)
            proposal = mean + (F.softplus(scale)+1e-5)*self.innovations.roll(role, 0)
            proposal = (proposal-proposal.mean(1, keepdim=True))/torch.sqrt(
                proposal.var(1, unbiased=False, keepdim=True)+1e-5)
            candidate = (1-z)*F.leaky_relu(proposal)+z*h
            updated = logw+self.likelihood(torch.cat([candidate, obs], -1)).squeeze(-1)
            updated = updated-torch.logsumexp(updated, -1, keepdim=True)
            weights = updated.exp()
            q = .5*weights+.5/k
            cdf = q.cumsum(-1)
            indices = torch.searchsorted(cdf.contiguous(),
                                         self.positions.expand(b, -1).contiguous()).clamp_max(k-1)
            candidate = candidate.gather(1, indices[..., None].expand(-1, -1, d))
            corrected = (updated-q.log()).gather(1, indices)
            corrected = corrected-torch.logsumexp(corrected, -1, keepdim=True)
            valid = present[:, role, None]
            h = torch.where(valid[..., None], candidate, h)
            logw = torch.where(valid, corrected, logw)
        return (h*logw.exp()[..., None]).sum(1)


METHODS = {
    'survey40_adf_gaussian_moment_propagation': ADF,
    'survey80_recurrent_kalman_network': RKN,
    'survey80_robust_gnc_tls_consensus': GNCTLS,
    'survey80_particle_filter_rnn': PFRNN,
}


def build(method, latent_dim=256, num_heads=8, value_dim=64):
    return METHODS[method](latent_dim, num_heads, value_dim)
