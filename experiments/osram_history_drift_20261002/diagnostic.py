"""Read-only paired-history diagnostics. Tensor layouts are [time,batch,...]."""
import torch


def history_metadata(a, b, umask):
    valid = umask.T.bool()
    if a.shape != b.shape or a.shape != (*valid.shape, 3):
        raise ValueError('Expected availability [time,batch,3], umask [batch,time]')
    if not (((a == 0) | (a == 1)).all() and ((b == 0) | (b == 1)).all()):
        raise ValueError('Availability must be binary')
    if (b > a).any() or (b.sum(-1)[valid] == 0).any():
        raise ValueError('View2 must be a nonempty subset on valid utterances')
    deleted = (a.bool() & ~b.bool()) & valid[..., None]
    changed = deleted.any(-1)
    prior = deleted.long().cumsum(0) - deleted.long()
    positions = torch.arange(a.shape[0], device=a.device)[:, None].expand_as(valid)
    last = torch.where(changed, positions, -torch.ones_like(positions)).cummax(0).values
    last_prior = torch.cat((-torch.ones_like(last[:1]), last[:-1]), dim=0)
    distance = torch.where((last_prior >= 0) & valid, positions - last_prior, -1)
    return dict(contrast_mask=valid & ~changed & (last_prior >= 0),
                nearest_prior_change_distance=distance,
                prior_deleted_bit_count=prior.sum(-1), prior_deleted_modalities=prior,
                current_pattern=(a.long() * a.new_tensor([1, 2, 4]).long()).sum(-1),
                deleted=deleted, changed=changed)


def nested_view(availability, umask, mode, seed, prob=.2):
    if mode not in ('A', 'T', 'V', 'mixed') or not 0 <= prob <= 1:
        raise ValueError('Invalid deletion mode or probability')
    valid = umask.T.bool().to(availability.device)
    observed = availability.bool() & valid[..., None]
    generator = torch.Generator(device='cpu').manual_seed(int(seed))
    draws = torch.rand(availability.shape, generator=generator).to(availability.device)
    eligible = torch.ones(3, dtype=torch.bool, device=availability.device)
    if mode != 'mixed':
        eligible[:] = False
        eligible[('A', 'T', 'V').index(mode)] = True
    kept = observed & ~((draws < prob) & eligible)
    empty = valid & ~kept.any(-1)
    # Independent shared random priorities give uniform rescue among observed bits.
    priorities = torch.rand(availability.shape, generator=generator).to(availability.device)
    rescue = priorities.masked_fill(~observed, -1).argmax(-1)
    kept |= torch.nn.functional.one_hot(rescue, 3).bool() & empty[..., None]
    b = kept.to(availability.dtype)
    return b, history_metadata(availability, b, umask)


def vector_drift(v1, v2, eps=1e-8):
    a, b = v1.double(), v2.double()
    n1, n2 = a.norm(dim=-1), b.norm(dim=-1)
    absolute = (a - b).norm(dim=-1)
    valid = (n1 > 0) & (n2 > 0)
    cosine = ((a * b).sum(-1) / (n1 * n2).clamp_min(torch.finfo(a.dtype).tiny)).clamp(-1, 1)
    drift = torch.where(absolute == 0, torch.zeros_like(cosine), 1 - cosine)
    return dict(cos=torch.where(valid, drift, torch.full_like(drift, float('nan'))),
                valid=valid, abs=absolute, rel=absolute / (n1 + eps))


def _validate(model):
    if any(m.training for m in model.modules()):
        raise ValueError('Requires eval mode')
    b = model.osram
    if b.osram_readout_fusion != 'flat' or b.bidirectional or getattr(b, 'forward_slot_reuse', False):
        raise ValueError('Requires original causal cfg84 Flat without forward slot reuse')
    flags = ('osram_post_grn', 'osram_history_input_gate', 'osram_local_evidence_gate',
             'osram_hierarchical_evidence_gate', 'osram_local_skip_gate',
             'osram_memory_only_adapter', 'history_query_adapter')
    if any(getattr(b, key, False) for key in flags):
        raise ValueError('Readout/query adaptations forbidden')
    if any(getattr(b, key, 'full') != 'full' for key in ('osram_ablation', 'osram_emotion_ablation')):
        raise ValueError('Readout ablations forbidden')
    if any(getattr(model, key, False) for key in ('local_context_residual', 'classification_completion', 'pretrained_completion')):
        raise ValueError('Completion/adaptations forbidden')
    if getattr(model, 'readout_type', 'shared') != 'shared' or b.latent_dim != 256:
        raise ValueError('Requires cfg84 shared readout with latent256')
    return b


@torch.no_grad()
def capture(model, view):
    """view is a zero-argument callable executing one full original model forward."""
    backbone = _validate(model)
    inputs = []
    handle = backbone.emotion_adapter.register_forward_pre_hook(
        lambda module, args: inputs.append(args[0].detach().clone()))
    try:
        output = view()
    finally:
        handle.remove()
    if len(inputs) != 1 or inputs[0].shape[-1] != 256 + 4 * 1024:
        raise ValueError('Expected exactly one cfg84 Flat input')
    local = inputs[0][..., :256]
    memory = inputs[0][..., 256:].reshape(*local.shape[:-1], 4, 1024)
    if torch.count_nonzero(memory[..., 512:]).item():
        raise AssertionError('cfg84 backward context must be exactly zero')
    return dict(local=local, base=memory[..., 0, :512], gap=memory[..., 1:, :512],
                hidden=output[1].detach().clone(), prediction=output[0].detach().clone())


def anchor_metrics(first, second, availability):
    result = {}
    for name in ('local', 'base', 'hidden'):
        metric = vector_drift(first[name], second[name])
        for suffix, value in metric.items():
            result[name + '_' + suffix] = value
    result['delta_local'] = result['local_rel']
    result['absmax_local'] = (first['local'] - second['local']).abs().amax(-1)
    active = ~availability.bool()
    gaps = []
    for snapshot in (first, second):
        gaps.append(torch.where(active[..., None], snapshot['gap'], torch.zeros_like(snapshot['gap'])))
    for idx, name in enumerate(('a', 't', 'v')):
        for suffix, value in vector_drift(gaps[0][..., idx, :], gaps[1][..., idx, :]).items():
            result['gap_' + name + '_' + suffix] = value
    for suffix, value in vector_drift(gaps[0].flatten(-2), gaps[1].flatten(-2)).items():
        result['gap_' + suffix] = value
    p1, p2 = first['prediction'].squeeze(-1), second['prediction'].squeeze(-1)
    result.update(pred1=p1, pred2=p2, prediction_shift=(p1-p2).abs(),
                  sign_flip=(p1 > 0) != (p2 > 0), mathematical_sign_flip=p1.sign() != p2.sign())
    return result
