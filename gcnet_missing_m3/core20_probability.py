"""Source-grounded probabilistic mechanisms for core20 C09--C12.

These are feature-space adaptations, not reproductions of published scores.
Author pins, equations, and explicit adaptations live in probability.json.
All recurrent state is local to scan; unavailable inputs are masked before use.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

MODALITIES = ("audio", "text", "visual")


def _mask_inputs(latents, availability, valid):
    observed = availability.bool() & valid[..., None]
    return {name: torch.where(observed[..., i].reshape(*valid.shape, *([1] * (latents[name].ndim - 2))), latents[name],
                              torch.zeros_like(latents[name]))
            for i, name in enumerate(MODALITIES)}, observed


class _Memory(nn.Module):
    def __init__(self, num_heads, key_dim, value_dim, latent_dim):
        super().__init__()
        self.num_heads, self.key_dim, self.value_dim = num_heads, key_dim, value_dim
        self.state_dim = latent_dim
        self.query_address = nn.Linear(key_dim, latent_dim, bias=False)
        self.state_value = nn.Linear(latent_dim, value_dim, bias=False)

    def _read(self, mean, query):
        # A query-conditioned read of the belief mean; zero initial mean reads zero.
        return self.state_value(mean * self.query_address(query))

    def scan(self, keys, values, queries, availability, valid, *, address_residual=None):
        length, batch = availability.shape[:2]
        if valid.shape != (length, batch) or queries.shape != (
                length, batch, 4, self.num_heads, self.key_dim):
            raise ValueError("memory query/mask dimensions do not match")
        valid = valid.bool()
        keys, observed = _mask_inputs(keys, availability, valid)
        values, _ = _mask_inputs(values, availability, valid)
        queries = torch.where(valid[..., None, None, None], queries, torch.zeros_like(queries))
        state = self._initial(queries, batch)
        base, gap = [], []
        diagnostics = {m: {k: [] for k in ("rho", "eta", "cosine")} for m in MODALITIES}
        for t in range(length):
            active = valid[t]
            if not bool(active.any()):
                base.append(queries.new_zeros(batch, self.num_heads * self.value_dim))
                gap.append(queries.new_zeros(batch, 3, self.num_heads * self.value_dim))
                continue
            state = self._before_read(state, active)
            mean = self._mean(state)
            read = self._read(mean, queries[t, :, 0]).reshape(batch, -1)
            base.append(torch.where(active[:, None], read, torch.zeros_like(read)))
            slots = torch.stack([keys[m][t] for m in MODALITIES], -1)
            gap_t = []
            for i, name in enumerate(MODALITIES):
                q = queries[t, :, i + 1]
                rq = address_residual(slots, q) if address_residual is not None else q
                rho = rq.norm(dim=-1) / q.norm(dim=-1).clamp_min(1e-8)
                cosine = F.cosine_similarity(q, rq, dim=-1, eps=1e-8)
                selected = active[:, None].expand(-1, self.num_heads)
                for metric, tensor in (("rho", rho), ("eta", 1 - cosine * rho), ("cosine", cosine)):
                    diagnostics[name][metric].extend(tensor[selected].detach().cpu().tolist())
                r = self._read(mean, rq).reshape(batch, -1)
                mask = active & ~observed[t, :, i]
                gap_t.append(torch.where(mask[:, None], r, torch.zeros_like(r)))
            gap.append(torch.stack(gap_t, 1))
            observations = torch.stack([torch.cat([keys[m][t], values[m][t]], -1)
                                        for m in MODALITIES], 2)
            state = self._assimilate(state, observations, observed[t], active)
        return torch.stack(base), torch.stack(gap), diagnostics


class RKNMemory(_Memory):
    """RKN with banded locally linear transition and upper/lower/side covariance."""
    def __init__(self, num_heads=8, key_dim=64, value_dim=64, latent_dim=256,
                 num_basis=15, bandwidth=3):
        if latent_dim < 2 or latent_dim % 2:
            raise ValueError("RKN latent_dim must be even")
        super().__init__(num_heads, key_dim, value_dim, latent_dim)
        self.obs_dim = d = latent_dim // 2
        self.encoder = nn.Sequential(nn.Linear(key_dim + value_dim, d * 2), nn.Tanh(),
                                     nn.Linear(d * 2, d * 2))
        self.coefficients = nn.Sequential(nn.Linear(latent_dim, 64), nn.Tanh(),
                                         nn.Linear(64, num_basis))
        basis = torch.zeros(num_basis, 4, d, d)
        basis[:, 1] = .2 * torch.eye(d)
        basis[:, 2] = -.2 * torch.eye(d)
        self.transition_basis = nn.Parameter(basis)
        indices = torch.arange(d)
        self.register_buffer("band", (indices[:, None] - indices[None, :]).abs() <= bandwidth)
        self.process_raw = nn.Parameter(torch.full((2, d), math.log(math.expm1(.1))))

    def _initial(self, reference, batch):
        mean = reference.new_zeros(batch, self.num_heads, self.state_dim)
        cov = reference.new_ones(batch, self.num_heads, self.obs_dim)
        return mean, (cov, cov.clone(), cov.new_zeros(cov.shape))

    @staticmethod
    def kalman_update(mean, cov, obs, obs_var):
        u, l, s = cov
        d = obs.shape[-1]
        denom = u + obs_var
        qu, ql = u / denom, s / denom
        residual = obs - mean[..., :d]
        return (mean + torch.cat([qu * residual, ql * residual], -1),
                ((1 - qu) * u, l - ql * s, (1 - qu) * s))

    def _before_read(self, state, active):
        mean, (u, l, s) = state
        coeff = self.coefficients(mean).softmax(-1)
        matrices = torch.einsum("bhk,kcij->bhcij", coeff, self.transition_basis * self.band)
        eye = torch.eye(self.obs_dim, device=mean.device, dtype=mean.dtype)
        a, b, c, d = matrices.unbind(2)
        a, d = a + eye, d + eye
        mv = lambda x, y: (x @ y.unsqueeze(-1)).squeeze(-1)
        mu, ml = mean.chunk(2, -1)
        nxt = torch.cat([mv(a, mu) + mv(b, ml), mv(c, mu) + mv(d, ml)], -1)
        noise = F.softplus(self.process_raw) + 1e-6
        nu = mv(a.square(), u) + 2 * mv(a * b, s) + mv(b.square(), l) + noise[0]
        nl = mv(c.square(), u) + 2 * mv(c * d, s) + mv(d.square(), l) + noise[1]
        ns = mv(c * a, u) + mv(d * a + c * b, s) + mv(d * b, l)
        mask = active[:, None, None]
        return torch.where(mask, nxt, mean), tuple(torch.where(mask, new, old)
                    for new, old in zip((nu, nl, ns), (u, l, s)))

    def _mean(self, state):
        return state[0]

    def _assimilate(self, state, observations, observed, active):
        mean, cov = state
        raw_obs, raw_var = self.encoder(observations).chunk(2, -1)
        obs = F.normalize(raw_obs, dim=-1)
        var = F.softplus(raw_var) + 1e-4
        # Independent Gaussian observations of H z are combined in information form,
        # so assimilation does not depend on audio/text/visual processing order.
        precision = torch.where(observed[:, None, :, None], var.reciprocal(), torch.zeros_like(var))
        total = precision.sum(2)
        combined_var = total.clamp_min(1e-8).reciprocal()
        combined_obs = (precision * obs).sum(2) * combined_var
        updated, ucov = self.kalman_update(mean, cov, combined_obs, combined_var)
        mask = (active & observed.any(-1))[:, None, None]
        return torch.where(mask, updated, mean), tuple(torch.where(mask, x, y) for x, y in zip(ucov, cov))


class PFRNNMemory(_Memory):
    """PF-GRU transitions, learned likelihood weights, and corrected soft resampling."""
    def __init__(self, num_heads=8, key_dim=64, value_dim=64, latent_dim=256,
                 num_particles=16, resamp_alpha=.5):
        super().__init__(num_heads, key_dim, value_dim, latent_dim)
        self.num_particles, self.resamp_alpha = num_particles, resamp_alpha
        self.obs_extractor = nn.Sequential(nn.Linear(key_dim + value_dim, 32), nn.LeakyReLU())
        self.act_extractor = nn.Sequential(nn.Linear(key_dim + value_dim, 32), nn.LeakyReLU())
        self.fc_z = nn.Linear(latent_dim + 32, latent_dim)
        self.fc_r = nn.Linear(latent_dim + 32, latent_dim)
        self.fc_n = nn.Linear(latent_dim + 32, latent_dim * 2)
        self.particle_norm = nn.LayerNorm(latent_dim)
        self.fc_obs = nn.Linear(latent_dim + 32, 1)

    def _initial(self, reference, batch):
        particles = reference.new_zeros(batch, self.num_heads, self.num_particles, self.state_dim)
        log_weights = reference.new_full(particles.shape[:-1], -math.log(self.num_particles))
        return particles, log_weights

    def _before_read(self, state, active):
        return state

    def _mean(self, state):
        particles, log_weights = state
        return (particles * log_weights.exp()[..., None]).sum(-2)

    @staticmethod
    def resampling_correction(log_weights, indices, alpha):
        count = log_weights.shape[-1]
        proposal = alpha * log_weights.exp() + (1 - alpha) / count
        corrected = (log_weights - proposal.log()).gather(-1, indices)
        return corrected - torch.logsumexp(corrected, -1, keepdim=True)

    def _assimilate(self, state, observations, observed, active):
        particles, log_weights = state
        count = observed.sum(-1).clamp_min(1).to(observations.dtype)
        x = observations.sum(2) / count[:, None, None]
        act = self.act_extractor(x).unsqueeze(2).expand(-1, -1, self.num_particles, -1)
        z = torch.sigmoid(self.fc_z(torch.cat([particles, act], -1)))
        r = torch.sigmoid(self.fc_r(torch.cat([particles, act], -1)))
        mu, raw_scale = self.fc_n(torch.cat([r * particles, act], -1)).chunk(2, -1)
        candidate = mu + torch.randn_like(mu) * F.softplus(raw_scale)
        nxt = (1 - z) * F.leaky_relu(self.particle_norm(candidate)) + z * particles
        obs = self.obs_extractor(observations)
        expanded = nxt.unsqueeze(3).expand(-1, -1, -1, 3, -1)
        obs = obs.unsqueeze(2).expand(-1, -1, self.num_particles, -1, -1)
        score = self.fc_obs(torch.cat([expanded, obs], -1)).squeeze(-1)
        score = torch.where(observed[:, None, None, :], score, torch.zeros_like(score)).sum(-1)
        updated = log_weights + score
        updated = updated - torch.logsumexp(updated, -1, keepdim=True)
        proposal = self.resamp_alpha * updated.exp() + (1 - self.resamp_alpha) / self.num_particles
        indices = torch.multinomial(proposal.reshape(-1, self.num_particles),
                                   self.num_particles, replacement=True).reshape_as(updated)
        resampled = nxt.gather(2, indices[..., None].expand_as(nxt))
        corrected = self.resampling_correction(updated, indices, self.resamp_alpha)
        # No observations: propagate particles, preserve weights, do not fabricate evidence.
        has_obs = observed.any(-1)[:, None, None]
        nxt = torch.where(has_obs[..., None], resampled, nxt)
        corrected = torch.where(has_obs, corrected, log_weights)
        mask = active[:, None, None]
        return (torch.where(mask[..., None], nxt, particles),
                torch.where(mask, corrected, log_weights))


def build_memory(method, num_heads=8, key_dim=64, value_dim=64, latent_dim=256):
    names = {"rkn": RKNMemory, "c09": RKNMemory, "pfrnn": PFRNNMemory,
             "pf-rnn": PFRNNMemory, "c10": PFRNNMemory}
    if method.lower() not in names:
        raise ValueError(f"unknown probabilistic memory: {method}")
    return names[method.lower()](num_heads, key_dim, value_dim, latent_dim)


class MMVAE(nn.Module):
    """Observed-expert mixture with all observed conditional generative likelihoods.

    A stratified one-sample mixture ELBO (K=1) includes log p(z), log q_mix(z),
    and every observed original-feature likelihood. Task representations always
    use the deterministic mixture mean; evaluation skips the generative objective.
    No unavailable expert, likelihood target, or predicted latent memory write.
    """
    def __init__(self, latent_dim, reconstruction_dims):
        super().__init__()
        if set(reconstruction_dims) != set(MODALITIES) or any(d < 1 for d in reconstruction_dims.values()):
            raise ValueError("reconstruction_dims must specify positive audio/text/visual dimensions")
        self.latent_dim = latent_dim
        self.reconstruction_dims = dict(reconstruction_dims)
        self.encoders = nn.ModuleDict({m: nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.Tanh(),
                                            nn.Linear(latent_dim, 2 * latent_dim)) for m in MODALITIES})
        self.decoders = nn.ModuleDict({m: nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.Tanh(),
                         nn.Linear(latent_dim, 2 * reconstruction_dims[m])) for m in MODALITIES})

    def forward(self, latents, availability, umask, *, reconstruction_targets=None):
        valid = umask.T.bool()
        safe, observed = _mask_inputs(latents, availability, valid)
        count = observed.sum(-1).clamp_min(1)
        mus, scales = [], []
        for i, m in enumerate(MODALITIES):
            mu, raw = self.encoders[m](safe[m]).chunk(2, -1)
            mus.append(torch.where(observed[..., i, None], mu, torch.zeros_like(mu)))
            scales.append(F.softplus(raw) + 1e-4)
        means, scales = torch.stack(mus, 2), torch.stack(scales, 2)
        weights = observed.to(means.dtype) / count[..., None]
        representation = (means * weights[..., None]).sum(2)
        metadata = {"posterior_mean": means, "posterior_scale": scales,
                    "mixture_weights": weights, "available": observed}
        if not self.training:
            zero = representation.sum() * 0
            metadata.update(mixture_kl_mc=zero, reconstruction_nll=zero)
            return representation, zero, metadata
        if reconstruction_targets is None:
            raise ValueError("MMVAE training requires original observed-feature reconstruction_targets")
        if set(reconstruction_targets) != set(MODALITIES) or any(
                reconstruction_targets[m].shape != (*valid.shape, self.reconstruction_dims[m])
                for m in MODALITIES):
            raise ValueError("MMVAE reconstruction target dimensions do not match original features")
        targets, _ = _mask_inputs(reconstruction_targets, availability, valid)
        mixture = torch.distributions.Normal(means, scales)
        losses, kl_terms, reconstruction_terms = [], [], []
        for i in range(3):
            # Stratified estimator samples each available expert and averages by alpha_m.
            z = means[:, :, i] + scales[:, :, i] * torch.randn_like(means[:, :, i])
            log_components = mixture.log_prob(z.unsqueeze(2)).sum(-1)
            log_components = log_components.masked_fill(~observed, -torch.inf)
            # All-missing rows use prior fallback only; they contribute no generative loss.
            empty = ~observed.any(-1)
            log_components = torch.where(empty[..., None], torch.zeros_like(log_components), log_components)
            log_q = torch.logsumexp(log_components, -1) - count.to(z.dtype).log()
            log_p = torch.distributions.Normal(torch.zeros_like(z), torch.ones_like(z)).log_prob(z).sum(-1)
            reconstruction = torch.zeros_like(log_p)
            for j, m in enumerate(MODALITIES):
                loc, raw = self.decoders[m](z).chunk(2, -1)
                likelihood = torch.distributions.Normal(loc, F.softplus(raw) + 1e-4)
                ll = likelihood.log_prob(targets[m]).sum(-1)
                reconstruction = reconstruction + torch.where(observed[..., j], ll, torch.zeros_like(ll))
            w = weights[..., i]
            losses.append(w * (log_q - log_p - reconstruction))
            kl_terms.append(w * (log_q - log_p))
            reconstruction_terms.append(-w * reconstruction)
        selected = valid & observed.any(-1)
        denominator = selected.sum().clamp_min(1)
        loss = torch.stack(losses).sum(0)[selected].sum() / denominator
        metadata.update(mixture_kl_mc=torch.stack(kl_terms).sum(0)[selected].sum() / denominator,
                        reconstruction_nll=torch.stack(reconstruction_terms).sum(0)[selected].sum() / denominator)
        return representation, loss, metadata


class _RadialFlow(nn.Module):
    """Author radial-flow density with exact Jacobian, per input only."""
    def __init__(self, dim, layers=4):
        super().__init__()
        self.reference = nn.Parameter(torch.randn(layers, dim) / math.sqrt(dim))
        self.alpha_prime = nn.Parameter(torch.empty(layers).uniform_(-1 / math.sqrt(dim), 1 / math.sqrt(dim)))
        self.beta_prime = nn.Parameter(torch.empty(layers).uniform_(-1 / math.sqrt(dim), 1 / math.sqrt(dim)))

    def forward(self, z):
        log_det = z.new_zeros(z.shape[:-1])
        for center, ar, br in zip(self.reference, self.alpha_prime, self.beta_prime):
            alpha = F.softplus(ar)
            beta = -alpha + F.softplus(br)
            diff = z - center
            radius = diff.norm(dim=-1, keepdim=True)
            h = (alpha + radius).reciprocal()
            bh = beta * h
            log_det = log_det + ((z.shape[-1] - 1) * torch.log1p(bh)
                                  + torch.log1p(bh - beta * h.square() * radius)).squeeze(-1)
            z = z + bh * diff
        return -.5 * (z.square() + math.log(2 * math.pi)).sum(-1) + log_det


class NaturalPosteriorNetwork(nn.Module):
    """C12 normal likelihood, radial density, NormalGamma conjugate posterior.

    The task criterion is expected normal NLL minus posterior entropy, not MSE.
    Call loss(y, umask, parameters) with the parameters returned by forward.
    """
    def __init__(self, hidden_dim=1600, latent_dim=16, entropy_weight=1e-5):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(hidden_dim, 64), nn.Tanh(), nn.Linear(64, latent_dim))
        self.output = nn.Linear(latent_dim, 2)
        self.flow = _RadialFlow(latent_dim)
        self.entropy_weight = entropy_weight
        self.log_budget = .5 * math.log(4 * math.pi) * latent_dim

    @staticmethod
    def conjugate_update(loc, precision, evidence):
        # Source NormalGammaPrior(mean=0, scale=10, evidence=1).
        n = 1 + evidence
        mu = evidence * loc / n
        second = (100 + evidence * (loc.square() + precision.reciprocal())) / n
        return {"mu": mu, "lambda": n, "alpha": .5 * n,
                "beta": (.5 * n * (second - mu.square())).clamp_min(1e-6)}

    def forward(self, hidden, umask):
        valid = umask.T.bool()
        safe = torch.where(valid[..., None], hidden, torch.zeros_like(hidden))
        z = self.encoder(safe)
        loc, log_precision = self.output(z).unbind(-1)
        precision = log_precision.clamp(-20, 20).exp() + 1e-10
        log_density = self.flow(z)
        raw = log_density + self.log_budget
        # Source scaler's clamp-preserving-gradient behavior.
        log_evidence = raw + (raw.clamp(-30, 30) - raw).detach()
        params = self.conjugate_update(loc, precision, log_evidence.exp())
        params.update(log_density=log_density, log_evidence=log_evidence,
                      predictive_df=2 * params["alpha"],
                      predictive_scale=(params["beta"] * (params["lambda"] + 1)
                              / (params["alpha"] * params["lambda"])).sqrt())
        mean = torch.where(valid, params["mu"], torch.zeros_like(loc))
        return mean.unsqueeze(-1), params

    def loss(self, y, umask, parameters):
        valid = umask.T.bool()
        if y.shape == umask.shape:
            y = y.T
        elif y.shape == (*valid.shape, 1):
            y = y.squeeze(-1)
        if y.shape != valid.shape:
            raise ValueError("targets must be [B,L], [L,B], or [L,B,1]")
        mu, n, a, b = (parameters[k][valid] for k in ("mu", "lambda", "alpha", "beta"))
        target = y[valid]
        ell = .5 * (-math.log(2 * math.pi) - a / b * (mu - target).square()
                   + torch.digamma(a) - b.log() - n.reciprocal())
        t1 = .5 + .5 * math.log(2 * math.pi)
        exact = torch.lgamma(a) + a - (a + 1.5) * torch.digamma(a)
        entropy = t1 + torch.where(a >= 10000, t1 - 2 * a.log(), exact) + 1.5 * b.log() - .5 * n.log()
        return (-ell - self.entropy_weight * entropy).sum() / valid.sum().clamp_min(1)
