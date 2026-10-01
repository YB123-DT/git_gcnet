"""Training-only nested missing views and cross-conversation contrastive loss."""

import torch
import torch.nn.functional as F


def make_paired_view(availability, umask, drop_prob, generator):
    """Return a subset view, strict-history anchors and additive count diagnostics.

    A dedicated generator is mandatory: augmentation must not consume the RNG
    stream used by the baseline mask schedule. If independent drops remove an
    entire observed set, restore one uniformly sampled original observation.
    Consequently the *actual* drop fraction is smaller than ``drop_prob``.
    """
    if availability.ndim != 3 or availability.shape[-1] != 3:
        raise ValueError('availability must have shape [T, B, 3]')
    if tuple(umask.shape) != (availability.shape[1], availability.shape[0]):
        raise ValueError('umask must have shape [B, T]')
    if not 0 <= drop_prob <= 1:
        raise ValueError('drop_prob must be in [0, 1]')
    if generator is None:
        raise ValueError('a dedicated generator is required')
    valid = umask.T.to(device=availability.device, dtype=torch.bool)
    clean = torch.where(valid.unsqueeze(-1), availability, torch.zeros_like(availability))
    if not torch.all((clean == 0) | (clean == 1)):
        raise ValueError('valid availability must be binary and finite')
    observed = clean.bool()
    if torch.any(valid & ~observed.any(-1)):
        raise ValueError('every valid utterance must have an observed modality')
    draws = torch.rand(availability.shape, generator=generator, device=generator.device).to(availability.device)
    retained = observed & (draws >= drop_prob)
    empty = valid & ~retained.any(-1)
    # IID continuous scores give each originally observed modality equal chance.
    restoration = torch.rand(availability.shape, generator=generator, device=generator.device).to(availability.device)
    chosen = restoration.masked_fill(~observed, -1).argmax(-1)
    retained = retained | (F.one_hot(chosen, 3).bool() & empty.unsqueeze(-1))
    view2 = retained.to(availability.dtype)
    changed = valid & (observed != retained).any(-1)
    prior_changed = (changed.long().cumsum(0) - changed.long()) > 0
    contrast = valid & ~changed & prior_changed
    stats = {
        'valid_utterances': int(valid.sum().item()),
        'observed_before': int(observed.sum().item()),
        'observed_dropped': int((observed & ~retained).sum().item()),
        'changed_utterances': int(changed.sum().item()),
        'contrast_anchors': int(contrast.sum().item()),
    }
    return view2, contrast, stats


def symmetric_info_nce(z1, z2, conversation_ids, temperature=0.1):
    """Cross-view InfoNCE with own positives and only cross-conversation negatives.

    ``conversation_ids`` identifies each selected anchor, not each batch slot;
    repeated IDs deliberately exclude negatives across duplicate batch entries.
    Returns the loss and number of usable anchors (not twice that number).
    """
    if z1.ndim != 2 or z1.shape != z2.shape:
        raise ValueError('projected views must have equal [N, D] shapes')
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    n = z1.shape[0]
    if len(conversation_ids) != n:
        raise ValueError('one conversation ID is required per anchor')
    if torch.is_tensor(conversation_ids):
        conversation_ids = conversation_ids.detach().cpu().tolist()
    codes = {}
    encoded = [codes.setdefault(value, len(codes)) for value in conversation_ids]
    ids = torch.tensor(encoded, device=z1.device, dtype=torch.long)
    cross_conversation = ids[:, None] != ids[None, :]
    eligible = cross_conversation.any(-1)
    count = int(eligible.sum().item())
    if count == 0:
        return (z1.sum() + z2.sum()) * 0., 0
    allowed = cross_conversation | torch.eye(n, device=z1.device, dtype=torch.bool)
    logits = F.normalize(z1, dim=-1) @ F.normalize(z2, dim=-1).T / temperature
    logits = logits.masked_fill(~allowed, -float('inf'))
    targets = torch.arange(n, device=z1.device)[eligible]
    loss = .5 * (
        F.cross_entropy(logits[eligible], targets)
        + F.cross_entropy(logits.T[eligible], targets)
    )
    return loss, count
