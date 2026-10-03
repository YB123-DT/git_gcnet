"""Four complete grouping cores over current OSRAM reads, never memory queries.

These are declared task-only adaptations, not reproductions of image models.
Equations, pinned primary sources and adaptation choices are recorded in
experiments/osram_meaningful20_20261003/{grouping.json,verification/grouping.md}.
No external source code is copied (in particular, OTKE has no verified license).
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def _initialize_affine(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.GRUCell):
        nn.init.xavier_uniform_(module.weight_ih)
        nn.init.xavier_uniform_(module.weight_hh)
        nn.init.zeros_(module.bias_ih)
        nn.init.zeros_(module.bias_hh)


def _initialize_vote_matrices(weight):
    for matrix in weight.view(-1, *weight.shape[-2:]):
        nn.init.xavier_uniform_(matrix)


class LocalConditionalTokenizer(nn.Module):
    """One Local plus four groups of real head vectors, each projected to 64d."""

    def __init__(self, latent_dim, num_heads, value_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.value_dim = value_dim
        self.local_norm = nn.LayerNorm(latent_dim)
        self.local_projection = nn.Linear(latent_dim, 64)
        self.memory_norm = nn.LayerNorm(value_dim)
        self.memory_projection = nn.ModuleList(nn.Linear(value_dim, 64) for _ in range(4))
        self.local_condition = nn.ModuleList(nn.Linear(64, 64) for _ in range(4))
        self.type_embedding = nn.Parameter(torch.empty(5, 64))
        self.head_embedding = nn.Parameter(torch.empty(num_heads, 64))
        self.apply(_initialize_affine)
        nn.init.xavier_uniform_(self.type_embedding)
        nn.init.xavier_uniform_(self.head_embedding)

    def forward(self, local, evidence, active):
        row_active = active.any(dim=1)
        safe_local = torch.where(row_active[:, None], local, 0.0)
        local_code = F.gelu(self.local_projection(self.local_norm(safe_local)))
        local_code = torch.where(row_active[:, None], local_code, 0.0)
        local_token = torch.where(row_active[:, None], local_code + self.type_embedding[0], 0.0)
        head_values = evidence.reshape(-1, 4, self.num_heads, self.value_dim)
        tokens, masks = [local_token[:, None]], [row_active[:, None]]
        for kind in range(4):
            mask = active[:, kind, None].expand(-1, self.num_heads)
            safe_values = torch.where(mask[..., None], head_values[:, kind], 0.0)
            normalized = torch.where(mask[..., None], self.memory_norm(safe_values), 0.0)
            projected = torch.where(mask[..., None], self.memory_projection[kind](normalized), 0.0)
            conditioned = self.local_condition[kind](local_code)[:, None]
            conditioned = torch.where(mask[..., None], conditioned, 0.0)
            token = F.gelu(projected + conditioned + self.type_embedding[kind + 1] + self.head_embedding)
            tokens.append(torch.where(mask[..., None], token, 0.0))
            masks.append(mask)
        return torch.cat(tokens, dim=1), torch.cat(masks, dim=1)


class _GroupingBase(nn.Module):
    output_dim = 128

    def __init__(self, latent_dim, num_heads, value_dim):
        super().__init__()
        if any(not isinstance(d, int) or isinstance(d, bool) or d <= 0 for d in (latent_dim, num_heads, value_dim)):
            raise ValueError("latent_dim, num_heads and value_dim must be positive integers")
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.value_dim = value_dim
        self.token_count = 1 + 4 * num_heads
        self.tokenizer = LocalConditionalTokenizer(latent_dim, num_heads, value_dim)

    def forward(self, local, evidence, active, availability):
        if local.ndim != 2 or local.shape[1] != self.latent_dim:
            raise ValueError("local must have shape [N, latent_dim]")
        rows = local.shape[0]
        if evidence.shape != (rows, 4, self.num_heads * self.value_dim):
            raise ValueError("evidence must have shape [N, 4, num_heads * value_dim]")
        if active.shape != (rows, 4) or active.dtype != torch.bool:
            raise ValueError("active must be boolean [N, 4]")
        if availability.shape != (rows, 3):
            raise ValueError("availability must have shape [N, 3]")
        if any(value.device != local.device for value in (evidence, active, availability)):
            raise ValueError("all grouping inputs must be on the same device")
        # active is authoritative. availability is metadata, not a second mask.
        selected = active.any(dim=1).nonzero(as_tuple=False).flatten()
        result = local.new_zeros(rows, self.output_dim)
        if selected.numel() == 0:
            return result
        # Keep the FP32 routing policy under autocast, while honoring an
        # explicitly double-precision module (e.g. numerical verification).
        compute_dtype = (torch.float64 if self.tokenizer.local_projection.weight.dtype == torch.float64
                         else torch.float32)
        with torch.autocast(device_type=local.device.type, enabled=False):
            tokens, mask = self.tokenizer(
                local.index_select(0, selected).to(dtype=compute_dtype),
                evidence.index_select(0, selected).to(dtype=compute_dtype),
                active.index_select(0, selected),
            )
            grouped = self.group_tokens(tokens, mask)
        return result.index_copy(0, selected, grouped.to(local.dtype))


def dynamic_routing(votes, mask, iterations=3):
    """Procedure 1: zero logits, parent competition, squash, accumulated agreement."""
    votes = torch.where(mask[..., None, None], votes, 0.0)
    logits = votes.new_zeros(votes.shape[:-1])
    for step in range(iterations):
        assignment = logits.softmax(dim=-1)
        total = (assignment[..., None] * votes).sum(dim=1)
        norm_squared = total.square().sum(dim=-1, keepdim=True)
        capsules = norm_squared / (1.0 + norm_squared) * total / norm_squared.clamp_min(1e-8).sqrt()
        if step < iterations - 1:
            agreement = (votes * capsules[:, None]).sum(dim=-1)
            logits = logits + torch.where(mask[..., None], agreement, 0.0)
    return capsules


class DynamicRoutingGrouping(_GroupingBase):
    def __init__(self, latent_dim, num_heads, value_dim):
        super().__init__(latent_dim, num_heads, value_dim)
        self.vote_weight = nn.Parameter(torch.empty(self.token_count, 4, 32, 64))
        _initialize_vote_matrices(self.vote_weight)

    def group_tokens(self, tokens, mask):
        tokens = torch.where(mask[..., None], tokens, 0.0)
        votes = torch.einsum("nid,ijod->nijo", tokens, self.vote_weight)
        return dynamic_routing(votes, mask, iterations=3).flatten(1)


def slot_assignment_weights(logits, mask):
    competition = logits.softmax(dim=-1)
    weights = torch.where(mask[..., None], competition + 1e-8, 0.0)
    return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)


class SlotAttentionGrouping(_GroupingBase):
    """Three competitive GRU refinements with isolated training-noise state."""

    def __init__(self, latent_dim, num_heads, value_dim):
        super().__init__(latent_dim, num_heads, value_dim)
        self.norm_inputs = nn.LayerNorm(64)
        self.norm_slots = nn.LayerNorm(64)
        self.norm_mlp = nn.LayerNorm(64)
        self.key = nn.Linear(64, 64, bias=False)
        self.value = nn.Linear(64, 64, bias=False)
        self.query = nn.Linear(64, 64, bias=False)
        self.gru = nn.GRUCell(64, 64)
        self.mlp = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64))
        self.phi = nn.Sequential(nn.Linear(64, 64), nn.GELU())
        self.mu = nn.Parameter(torch.empty(1, 1, 64))
        self.log_sigma = nn.Parameter(torch.zeros(1, 1, 64))
        # Initializers consume the normal model-init RNG, never a forward RNG.
        for module in (self.key, self.value, self.query, self.gru, self.mlp, self.phi):
            module.apply(_initialize_affine)
        nn.init.xavier_uniform_(self.mu.view(1, 64))
        private = torch.Generator(device="cpu")
        private.manual_seed((torch.initial_seed() + 104729) % (2 ** 63))
        self.register_buffer("training_noise_rng_state", private.get_state())
        evaluation = torch.Generator(device="cpu").manual_seed(1729)
        self.register_buffer("epsilon_eval", torch.randn(4, 64, generator=evaluation))

    def configure_seed(self, experiment_seed):
        """Optional one-time trainer setup; never call this after checkpoint load."""
        private = torch.Generator(device="cpu")
        private.manual_seed((int(experiment_seed) + 104729) % (2 ** 63))
        self.training_noise_rng_state.copy_(private.get_state().to(self.training_noise_rng_state.device))

    def initial_slots(self, rows, device):
        if self.training:
            private = torch.Generator(device="cpu")
            private.set_state(self.training_noise_rng_state.detach().cpu())
            noise = torch.randn(rows, 4, 64, generator=private, device="cpu")
            self.training_noise_rng_state.copy_(private.get_state().to(self.training_noise_rng_state.device))
            noise = noise.to(device=device, dtype=self.mu.dtype)
        else:
            noise = self.epsilon_eval.to(device=device, dtype=self.mu.dtype)[None].expand(rows, -1, -1)
        return self.mu + self.log_sigma.exp() * noise

    def refine_slots(self, tokens, mask, slots):
        tokens = torch.where(mask[..., None], tokens, 0.0)
        normalized = torch.where(mask[..., None], self.norm_inputs(tokens), 0.0)
        keys = torch.where(mask[..., None], self.key(normalized), 0.0)
        values = torch.where(mask[..., None], self.value(normalized), 0.0)
        for _ in range(3):
            queries = self.query(self.norm_slots(slots))
            logits = torch.einsum("nid,njd->nij", keys, queries) / 8.0
            weights = slot_assignment_weights(logits, mask)
            updates = torch.einsum("nij,nid->njd", weights, values)
            slots = self.gru(updates.reshape(-1, 64), slots.reshape(-1, 64)).reshape(-1, 4, 64)
            slots = slots + self.mlp(self.norm_mlp(slots))
        return slots

    def group_tokens(self, tokens, mask):
        slots = self.refine_slots(tokens, mask, self.initial_slots(tokens.shape[0], tokens.device))
        features = self.phi(slots)
        return torch.cat((features.mean(dim=1), features.amax(dim=1)), dim=-1)


def sinkhorn_plan(cost, iterations=30, epsilon=0.5, mask=None):
    """Unit-mass OT, optionally batched with mathematically absent masked rows.

    Inactive rows receive log mass -inf, not a finite dummy observation. The
    result equals separately packing each utterance, without per-row launches.
    """
    single = cost.ndim == 2
    if single:
        cost = cost[None]
        if mask is not None:
            mask = mask[None]
    rows, count, supports = cost.shape
    if mask is None:
        mask = torch.ones(rows, count, device=cost.device, dtype=torch.bool)
    nonempty = mask.any(dim=1)
    result = cost.new_zeros(cost.shape)
    if count == 0 or not nonempty.any():
        return result[0] if single else result
    if not nonempty.all():
        selected = nonempty.nonzero(as_tuple=False).flatten()
        plans = sinkhorn_plan(cost[selected], iterations, epsilon, mask[selected])
        result = result.index_copy(0, selected, plans)
        return result[0] if single else result
    safe_cost = torch.where(mask[..., None], cost, 0.0)
    log_kernel = -safe_cost / epsilon
    log_mass = -mask.sum(dim=1, keepdim=True).to(cost.dtype).log()
    log_u = cost.new_zeros(rows, count)
    log_v = cost.new_zeros(rows, supports)
    for _ in range(iterations):
        log_u = log_mass - torch.logsumexp(log_kernel + log_v[:, None], dim=2)
        log_u = torch.where(mask, log_u, -torch.inf)
        log_v = -math.log(supports) - torch.logsumexp(log_kernel + log_u[..., None], dim=1)
    result = (log_kernel + log_u[..., None] + log_v[:, None]).exp()
    return result[0] if single else result


class OTKernelGrouping(_GroupingBase):
    """Nonlinear features, learned reference supports, doubly constrained transport."""

    def __init__(self, latent_dim, num_heads, value_dim):
        super().__init__(latent_dim, num_heads, value_dim)
        self.feature_map = nn.Linear(64, 64)
        self.supports = nn.Parameter(torch.empty(4, 64))
        self.output_projection = nn.Linear(256, 128)
        _initialize_affine(self.feature_map)
        _initialize_affine(self.output_projection)
        nn.init.xavier_uniform_(self.supports)

    def group_tokens(self, tokens, mask):
        supports = F.normalize(self.supports, dim=-1, eps=1e-8)
        tokens = torch.where(mask[..., None], tokens, 0.0)
        projected = torch.where(mask[..., None], self.feature_map(tokens), 0.0)
        features = F.normalize(F.relu(projected), dim=-1, eps=1e-8)
        features = torch.where(mask[..., None], features, 0.0)
        transport = sinkhorn_plan(-features @ supports.T, mask=mask)
        summaries = 2.0 * torch.bmm(transport.transpose(1, 2), features)
        return self.output_projection(summaries.flatten(1))


def vb_posterior(votes, mass):
    """Full Gaussian-Wishart/Dirichlet posterior for the card's fixed priors.

    votes: [N, input, 4, 16], mass: [N, input, 4]. All calculations stay
    attached; the returned Cholesky factor is reused by responsibility updates.
    """
    count = mass.sum(dim=1)
    average = (mass[..., None] * votes).sum(dim=1) / count.clamp_min(1e-8)[..., None]
    delta = votes - average[:, None]
    scatter = torch.einsum("nij,nijd,nije->njde", mass, delta, delta)
    alpha = 1.0 + count
    kappa = 1.0 + count
    nu = 17.0 + count
    mean = count[..., None] * average / kappa[..., None]
    outer = average[..., :, None] * average[..., None, :]
    identity = torch.eye(16, device=votes.device, dtype=votes.dtype)
    inverse_scale = (1.0 + 1e-6) * identity + scatter + (count / kappa)[..., None, None] * outer
    cholesky = torch.linalg.cholesky(inverse_scale)
    logdet = 2.0 * cholesky.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
    indices = torch.arange(16, device=votes.device, dtype=votes.dtype)
    expected_logdet = 16.0 * math.log(2.0) - logdet + torch.digamma((nu[..., None] - indices) / 2.0).sum(dim=-1)
    expected_logpi = torch.digamma(alpha) - torch.digamma(alpha.sum(dim=-1, keepdim=True))
    return {
        "count": count,
        "kappa": kappa,
        "nu": nu,
        "mean": mean,
        "inverse_scale": inverse_scale,
        "cholesky": cholesky,
        "expected_logdet": expected_logdet,
        "expected_logpi": expected_logpi,
    }


class VariationalBayesGrouping(_GroupingBase):
    """Matrix votes and three covariance-aware posterior/assignment rounds."""

    def __init__(self, latent_dim, num_heads, value_dim):
        super().__init__(latent_dim, num_heads, value_dim)
        self.pose_projection = nn.ModuleList(nn.Linear(64, 16) for _ in range(5))
        self.activation_projection = nn.ModuleList(nn.Linear(64, 1) for _ in range(5))
        self.vote_weight = nn.Parameter(torch.empty(self.token_count, 4, 4, 4))
        self.beta_a = nn.Parameter(torch.zeros(4))
        self.beta_u = nn.Parameter(torch.zeros(4))
        self.output_projection = nn.Linear(68, 128)
        self.pose_projection.apply(_initialize_affine)
        self.activation_projection.apply(_initialize_affine)
        _initialize_affine(self.output_projection)
        _initialize_vote_matrices(self.vote_weight)

    def project_capsules(self, tokens, mask):
        poses, activations = [], []
        boundaries = [(0, 1)] + [(1 + kind * self.num_heads, 1 + (kind + 1) * self.num_heads) for kind in range(4)]
        for kind, (start, end) in enumerate(boundaries):
            valid = mask[:, start:end]
            safe_tokens = torch.where(valid[..., None], tokens[:, start:end], 0.0)
            pose = self.pose_projection[kind](safe_tokens)
            pose = torch.where(valid[..., None], pose, 0.0)
            activation = self.activation_projection[kind](safe_tokens).squeeze(-1)
            activation = torch.where(valid, activation, 0.0)
            activation = torch.sigmoid(activation).clamp(1e-6, 1.0 - 1e-6)
            poses.append(pose.reshape(tokens.shape[0], end - start, 4, 4))
            activations.append(torch.where(valid, activation, 0.0))
        return torch.cat(poses, dim=1), torch.cat(activations, dim=1)

    def route_votes(self, votes, activations):
        active = activations > 0
        votes = torch.where(active[..., None, None], votes, 0.0)
        responsibility = torch.where(active[..., None], torch.full_like(votes[..., 0], 0.25), 0.0)
        for step in range(3):
            posterior = vb_posterior(votes, activations[..., None] * responsibility)
            if step < 2:
                delta = votes - posterior["mean"][:, None]
                rhs = delta.permute(0, 2, 3, 1)
                solved = torch.cholesky_solve(rhs, posterior["cholesky"]).permute(0, 3, 1, 2)
                mahalanobis = (delta * solved).sum(dim=-1)
                score = posterior["expected_logpi"][:, None] + 0.5 * posterior["expected_logdet"][:, None]
                score = score - 0.5 * (16.0 / posterior["kappa"][:, None] + posterior["nu"][:, None] * mahalanobis)
                responsibility = torch.where(active[..., None], score.softmax(dim=-1), 0.0)
        entropy = 8.0 * math.log(2.0 * math.pi * math.e) - 0.5 * posterior["expected_logdet"]
        parent_activation = torch.sigmoid(self.beta_a - posterior["expected_logpi"].exp() * entropy - self.beta_u)
        parent_active = posterior["count"] > 1e-8
        parent_activation = torch.where(parent_active, parent_activation, 0.0)
        means = torch.where(parent_active[..., None], posterior["mean"], 0.0)
        return means, parent_activation

    def group_tokens(self, tokens, mask):
        poses, activations = self.project_capsules(tokens, mask)
        votes = torch.einsum("nirc,ijcd->nijrd", poses, self.vote_weight).flatten(-2)
        means, parent_activation = self.route_votes(votes, activations)
        summary = torch.cat(((means * parent_activation[..., None]).flatten(1), parent_activation), dim=-1)
        return self.output_projection(summary)


_GROUPING_BUILDERS = {
    "capsule_dynamic_routing": DynamicRoutingGrouping,
    "slot_attention": SlotAttentionGrouping,
    "otke": OTKernelGrouping,
    "capsule_variational_bayes": VariationalBayesGrouping,
}


def build_grouping(method, latent_dim, num_heads, value_dim):
    """Build one current-read core; the caller owns Flat anchoring and history guards."""
    if method not in _GROUPING_BUILDERS:
        raise ValueError(f"Unknown grouping method: {method!r}")
    return _GROUPING_BUILDERS[method](latent_dim, num_heads, value_dim)
