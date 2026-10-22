"""Read-only KV measurements and exact skip-source joint-write state forks."""
import torch


@torch.no_grad()
def kv_quality(osram, before, after, keys, values, availability):
    """One joint Frobenius score per conversation, pooling heads/active slots.

    `before` is the actual already-decayed write input, NOT M[t-1]. Both
    errors use exactly the embedded sqrt(beta) weighting of block_write.
    """
    if osram.beta_mode != 'embedded':
        raise ValueError('This diagnostic requires original embedded beta')
    beta = osram.beta_logits.sigmoid().to(keys)
    scale = (availability.to(keys)[:, None, :] * beta.clamp_min(0).sqrt()[None]).unsqueeze(2)
    kb, vb = keys * scale, values * scale
    energy = vb.square().sum((1, 2, 3))
    denominator = energy.clamp_min(1e-12)
    eb = (vb - before @ kb).square().sum((1, 2, 3)) / denominator
    ea = (vb - after @ kb).square().sum((1, 2, 3)) / denominator
    result = dict(e_before=eb, e_after=ea, q_gain=eb-ea,
                  value_energy=energy, write_update_norm=(after-before).flatten(1).norm(dim=-1))
    if any(not bool(torch.isfinite(v).all()) for v in result.values()):
        raise FloatingPointError('Nonfinite joint KV measurement')
    return result


@torch.no_grad()
def omit_write_forks(osram, snapshots, source_index, *, distances=(1, 2, 4, 8)):
    """Skip ONLY the source write; preserve source decay and every later write.

    Yield pre-target-write states. The factual branch must reproduce actual
    captured read state. No production sampler, query or write formula changes.
    """
    if not distances or any(not isinstance(d, int) or d < 1 for d in distances):
        raise ValueError('Distances must be positive integers')
    if not 0 <= source_index < len(snapshots):
        raise ValueError('Source must name an existing joint write')
    source_t, before, _, _, _, after = snapshots[source_index]
    states = dict(keep=after.clone(), omit=before.clone())
    alpha = osram.alpha_logits.sigmoid().to(after).view(1, -1, 1, 1)
    beta = osram.beta_logits.sigmoid().to(after)
    wanted = set(distances)
    for t, actual_read, keys, values, availability, _ in snapshots[source_index+1:]:
        distance = t-source_t
        if distance > max(wanted):
            break
        valid = availability.bool().any(-1)
        active = valid[:, None, None, None]
        states = {name:torch.where(active, memory*alpha, memory)
                  for name,memory in states.items()}
        difference = (states['keep']-actual_read)[valid]
        error = float(difference.abs().max()) if difference.numel() else 0.
        torch.testing.assert_close(states['keep'][valid], actual_read[valid],rtol=2e-5,atol=2e-5)
        if distance in wanted:
            yield distance, {name:value.clone() for name,value in states.items()}, error
        if distance < max(wanted):
            states = {name:torch.where(active,
                osram.block_write(memory,keys,values,availability,beta=beta),memory)
                for name,memory in states.items()}
