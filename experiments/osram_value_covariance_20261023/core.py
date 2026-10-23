"""Read-only projected-Value perturbation ledger for original embedded-beta CHM.

This is conditional sensitivity at FIXED K/Q, not uncertainty or raw-feature
robustness. G stores all four-role covariance entries. Under independent
value coordinates and heads, full read covariance is sigma² G_h ⊗ I_value.
"""
import torch

MODALITIES = ('audio', 'text', 'visual')


def read_covariance(omega_before, effective_queries):
    q = effective_queries.to(omega_before)
    covariance = q.transpose(-1, -2) @ omega_before @ q
    return (covariance + covariance.transpose(-1, -2)) * .5


def update_omega(omega, alpha, kbar, solved, scale, eta, active):
    """Unit raw-Value noise: Omega+=Aᵀ(alpha² Omega)A+eta² PᵀD²P."""
    kbar, solved, scale = (x.to(omega) for x in (kbar, solved, scale))
    identity = torch.eye(omega.shape[-1], device=omega.device, dtype=omega.dtype)
    transition = identity - eta * (kbar @ solved)
    before = omega * alpha.to(omega).square()[None, :, None, None]
    innovation = solved.transpose(-1, -2) @ (scale.square()[..., None] * solved)
    after = transition.transpose(-1, -2) @ before @ transition + eta**2 * innovation
    after = (after + after.transpose(-1, -2)) * .5
    return torch.where(active[:, None, None, None], after, omega)


def flat_logits(model, local, reads, valid):
    """Original Flat on forward reads, preserving zero backward half and Skip.

    reads: [...,heads,value_dim,4]. Leading dimensions may include draws/heads.
    Inactive Gap columns must already be zero. Only original model modules run.
    """
    roles = reads.movedim(-1, -3).flatten(-2)
    contexts = torch.cat((roles, torch.zeros_like(roles)), dim=-1)
    emotion_input = torch.cat((local, contexts[..., 0, :], contexts[..., 1:, :].flatten(-2)), -1)
    anchor = model.osram.local_skip(local) + model.osram.emotion_adapter(emotion_input)
    logits = model.smax_fc(model.osram.emotion_norm(anchor))
    return torch.where(valid[..., None], logits, torch.zeros_like(logits))


def task_direction_variance(model, local, reads, covariance, valid):
    """Per-head grad_read(f)ᵀ(G⊗I)grad_read(f), unit sigma, no param update."""
    with torch.enable_grad():
        variable = reads.detach().requires_grad_(True)
        prediction = flat_logits(model, local.detach(), variable, valid)
        gradient = torch.autograd.grad(prediction.sum(), variable, create_graph=False)[0]
    g = gradient.detach().to(covariance)
    result = torch.einsum('...hvi,...hij,...hvj->...h', g, covariance, g)
    return result.clamp_min(0).detach()


@torch.no_grad()
def evaluate_sequence_draws(model, local, clean, deltas, valid):
    """Preserve original [T,B] Linear/GEMM shapes, including zero-noise parity.

    Flattening draws into the original batch or evaluating one timestep at a
    time can select different float32 GEMM kernels. Every task replay therefore
    uses precisely the original sequence/batch dimensions.
    """
    prediction = flat_logits(model, local, clean, valid)
    noisy = torch.stack([torch.stack([flat_logits(model, local, clean + draw, valid)
                                     for draw in sigma]) for sigma in deltas])
    head_predictions = []
    for h in range(clean.shape[-3]):
        only_head = torch.zeros_like(clean)
        only_head[..., h, :, :] = deltas[-1, 0, ..., h, :, :]
        head_predictions.append(flat_logits(model, local, clean + only_head, valid))
    return dict(clean_prediction=prediction, noisy_predictions=noisy,
                head_predictions=torch.stack(head_predictions))


@torch.no_grad()
def replay_value_noise(osram, keys, values, queries, availability, valid, *,
                       draws=16, sigmas=(.001, .01), seed=66):
    """Yield each pre-write read, G and actual full-history noisy write reads.

    Noisy trajectories call ORIGINAL block_write on perturbed projected Values.
    An independent linear delta-memory recurrence predicts the same realization;
    Omega predicts its covariance, not the individual Gaussian draw. Noise is
    temporally coherent across the full conversation. Only a private generator
    is used. No original state, parameter, mask, K, Q or RNG is modified.
    """
    if (osram.beta_mode != 'embedded' or osram.bidirectional or osram.forward_slot_reuse
            or draws < 2 or not sigmas or any(s <= 0 for s in sigmas)):
        raise ValueError('Require original causal embedded-beta CHM and positive noise scales')
    length, batch = availability.shape[:2]
    heads, value_dim, key_dim = osram.num_heads, osram.value_dim, osram.key_dim
    device, dtype = queries.device, queries.dtype
    generator = torch.Generator(device=device).manual_seed(seed)
    memory = queries.new_zeros(batch, heads, value_dim, key_dim)
    noisy_memory = queries.new_zeros(len(sigmas), draws, batch, heads, value_dim, key_dim)
    unit_delta = torch.zeros(draws, batch, heads, value_dim, key_dim, device=device, dtype=torch.float64)
    omega = torch.zeros(batch, heads, key_dim, key_dim, device=device, dtype=torch.float64)
    alpha = osram.alpha_logits.sigmoid().to(queries)
    beta = osram.beta_logits.sigmoid().to(queries)
    sigma = queries.new_tensor(sigmas)
    for t in range(length):
        active = valid[t].bool()
        a = availability[t]
        state_mask = active[:, None, None, None]
        memory_before = torch.where(state_mask, memory * alpha[None, :, None, None], memory)
        noisy_before = torch.where(state_mask[None, None],
            noisy_memory * alpha[None, None, None, :, None, None], noisy_memory)
        linear_before = torch.where(state_mask[None],
            unit_delta * alpha.double()[None, None, :, None, None], unit_delta)
        omega_before = torch.where(state_mask, omega * alpha.double().square()[None, :, None, None], omega)
        slot_keys = torch.stack([keys[m][t] for m in MODALITIES], dim=-1)
        slot_values = torch.stack([values[m][t] for m in MODALITIES], dim=-1)
        mask = a[:, None, None, :].to(dtype)
        slot_keys, slot_values = slot_keys * mask, slot_values * mask
        columns = [queries[t, :, 0]]
        for i in range(3):
            q = queries[t, :, i + 1]
            projected = osram._address_residual(slot_keys, q)
            strength = 0. if osram.osram_gap_read == 'raw' else osram.gap_residual_strength
            columns.append(q - strength * (q - projected))
        role_active = torch.cat((active[:, None], active[:, None] & ~a.bool()), dim=-1)
        effective = torch.stack(columns, dim=-1) * role_active[:, None, None, :].to(dtype)
        # Match production's four separate vector reads, not a four-column
        # GEMM with a potentially different float32 accumulation kernel.
        def production_reads(state):
            return torch.stack([
                (state @ column.unsqueeze(-1)).squeeze(-1) *
                role_active[:, r, None, None].to(dtype)
                for r, column in enumerate(columns)], dim=-1)
        clean = production_reads(memory_before)
        actual_delta = production_reads(noisy_before) - clean[None, None]
        predicted_unit = linear_before @ effective.double()[None]
        predicted_delta = sigma.double()[:, None, None, None, None, None] * predicted_unit[None]
        difference = actual_delta.double() - predicted_delta
        covariance = read_covariance(omega_before, effective)
        samples = (actual_delta[-1].double() / float(sigmas[-1])).permute(1, 2, 0, 3, 4)
        samples = samples.reshape(batch, heads, draws * value_dim, 4)
        centered = samples - samples.mean(-2, keepdim=True)
        empirical = centered.transpose(-1, -2) @ centered / (draws * value_dim - 1)
        observed = active[:, None, None, None] & a.bool()[:, None, None, :]
        energy = slot_values.square().masked_select(observed.expand_as(slot_values))
        row = dict(clean=clean, deltas=actual_delta, G=covariance,
            kappa=covariance.diagonal(dim1=-2, dim2=-1), cov_empirical=empirical,
            max_linear_abs_error=float(difference.abs().max()),
            linear_relative_error=float(difference.norm() / predicted_delta.norm().clamp_min(1e-12)),
            value_rms=float(energy.mean().sqrt()) if energy.numel() else 0.)
        yield t, row
        if not bool(active.any()):
            # No retention, write or private noise draw at padding-only t.
            continue
        clean_candidate, (kbar, _vbar, solved) = osram.block_write(
            memory_before, slot_keys, slot_values, a, beta=beta, return_write_terms=True)
        memory = torch.where(state_mask, clean_candidate, memory)
        scale = a[:, None, :].to(dtype) * beta.clamp_min(0).sqrt()[None]
        omega = update_omega(omega, alpha, kbar, solved, scale, osram.write_step, active)
        noise = torch.randn(draws, batch, heads, value_dim, 3, device=device,
                            dtype=dtype, generator=generator)
        identity = torch.eye(key_dim, device=device, dtype=torch.float64)
        transition = identity - osram.write_step * (kbar.double() @ solved.double())
        linear_candidate = linear_before @ transition[None] + osram.write_step * (
            noise.double() * scale.double()[None, :, :, None, :]) @ solved.double()[None]
        unit_delta = torch.where(state_mask[None], linear_candidate, unit_delta)
        noisy_values = slot_values[None, None] + sigma[:, None, None, None, None, None] * (
            noise[None] * mask[None, None])
        shape = (len(sigmas), draws, batch)
        n = len(sigmas) * draws * batch
        noisy_candidate = osram.block_write(
            noisy_before.reshape(n, heads, value_dim, key_dim),
            slot_keys.expand(*shape, heads, key_dim, 3).reshape(n, heads, key_dim, 3),
            noisy_values.reshape(n, heads, value_dim, 3),
            a.expand(*shape, 3).reshape(n, 3), beta=beta).reshape_as(noisy_memory)
        noisy_memory = torch.where(state_mask[None, None], noisy_candidate, noisy_memory)
