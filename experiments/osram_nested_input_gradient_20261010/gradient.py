"""Frozen readout task gradients at the shared pre-Nested/Flat inputs."""
import math

import torch


def measure_input_gradients(model, local, base, gap, availability, umask, labels, kind='flat'):
    """Measure sum-MSE gradients and scalar-output Jacobians, without updates.

    Inputs are time-major except umask/labels [B,L]. Memory derivatives use
    only four 512-dimensional forward reads. Local derivatives include both
    the unchanged skip and the adapter path. Returned tensors are detached.
    The Jacobian of the sum of predictions equals each row's own derivative
    because these eval readouts do not mix utterances or conversations.
    """
    if kind not in ('flat', 'nested'):
        raise ValueError('kind must be flat or nested')
    length, batch, latent_dim = local.shape
    expected = (length, batch)
    if (base.shape != (*expected, 1024) or gap.shape != (*expected, 3, 1024)
            or availability.shape != (*expected, 3) or umask.shape != (batch, length)
            or labels.shape != (batch, length)):
        raise ValueError('Expected cfg84 forward-512 read shapes and batch-major masks/labels')
    valid = umask.T.bool()
    observed = availability[valid]
    if not (((observed == 0) | (observed == 1)).all() and observed.bool().any(-1).all()):
        raise ValueError('Valid availability must be binary and nonempty')
    if not valid.any():
        raise ValueError('At least one valid utterance is required')
    model.eval()
    model.requires_grad_(False)
    active = torch.cat((valid[..., None], valid[..., None] & ~availability.bool()), -1)
    safe_av = torch.where(valid[..., None], availability, 0).detach()
    raw = torch.cat((base[..., :512].unsqueeze(2), gap[..., :512]), 2)
    local_values = torch.where(valid[..., None], local, 0).detach()
    memory_values = torch.where(active[..., None], raw, 0).detach()
    targets = torch.where(valid, labels.T, 0).detach()
    if not all(torch.isfinite(x).all() for x in (local_values, memory_values, targets)):
        raise ValueError('Nonfinite active input or label')
    with torch.enable_grad():
        local_leaf = local_values.clone().requires_grad_(True)
        memory_leaf = memory_values.clone().requires_grad_(True)
        safe_local = torch.where(valid[..., None], local_leaf, 0).contiguous()
        safe_memory = torch.where(active[..., None], memory_leaf, 0)
        full_memory = torch.cat((safe_memory, torch.zeros_like(safe_memory)), -1)
        readout_local = safe_local
        readout_base, readout_gap = full_memory[..., 0, :], full_memory[..., 1:, :]
        if kind == 'nested':
            readout_local, readout_base, readout_gap = model.osram.meaningful_block(
                safe_local, readout_base, readout_gap, safe_av, umask)
        readout_local = torch.where(valid[..., None], readout_local, 0)
        readout_base = torch.where(valid[..., None], readout_base, 0)
        readout_gap = torch.where(active[..., 1:, None], readout_gap, 0)
        emotion_input = torch.cat((readout_local, readout_base, readout_gap.flatten(2)), -1)
        anchor = model.osram.local_skip(safe_local) + model.osram.emotion_adapter(emotion_input)
        hidden = model.osram.emotion_norm(anchor)
        hidden = torch.where(valid[..., None], hidden, 0)
        prediction = model.smax_fc(hidden)
        if prediction.shape != (*expected, 1):
            raise ValueError('This MOSI diagnostic requires scalar regression predictions')
        prediction = torch.where(valid, prediction.squeeze(-1), 0)
        error = prediction - targets
        squared_error = torch.where(valid, error.square(), 0)
        gradients = torch.autograd.grad(squared_error.sum(), (local_leaf, memory_leaf),
                                        retain_graph=True)
        jacobians = torch.autograd.grad(prediction[valid].sum(), (local_leaf, memory_leaf))
    result = dict(prediction=prediction.detach(), valid=valid, active=active,
                  squared_error=squared_error.detach(),
                  feature_norm_local=local_values.norm(dim=-1),
                  feature_norm_memory=memory_values.norm(dim=-1))
    for prefix, pair in (('grad', gradients), ('jacobian', jacobians)):
        for name, tensor, mask, dimension in (
                ('local', pair[0], valid, latent_dim),
                ('memory', pair[1], active, 512)):
            tensor = torch.where(mask[..., None], tensor, 0).detach()
            if not torch.isfinite(tensor).all():
                raise ValueError('Nonfinite input gradient')
            result[f'{prefix}_{name}'] = tensor
            metric_prefix = 'loss_grad' if prefix == 'grad' else prefix
            result[f'{metric_prefix}_norm_{name}'] = tensor.norm(dim=-1)
            result[f'{metric_prefix}_rms_{name}'] = tensor.norm(dim=-1) / math.sqrt(dimension)
    return result
