"""Final-input interventions only: never call an encoder, query or Memory scan."""
from __future__ import annotations

import torch


@torch.no_grad()
def replay_readouts(osram, head, local, emotion_input, source_reads, umask,
                    availability, full_logits):
    """Return12 single-source removals and8 whole-source coalitions.

    Retain original full-shape tensor layout and untouched Local Skip. Sources
    index the existing joint write's Value columns; no semantic purity implied.
    """
    if any(layer.training for layer in (osram.local_skip, osram.emotion_adapter,
                                        osram.emotion_norm, head)):
        raise ValueError('Frozen eval readout required')
    length, batch, ld = local.shape
    fd = osram.num_heads * osram.value_dim
    if (ld != osram.latent_dim or emotion_input.shape != (length, batch, ld + 8 * fd)
            or source_reads.shape != (length, batch, 4, 3, fd)
            or availability.shape != (length, batch, 3) or umask.shape != (batch, length)):
        raise ValueError('Unexpected original Flat/source tensor dimensions')
    contexts = emotion_input[..., ld:].reshape(length, batch, 4, 2 * fd)
    if bool(contexts[..., fd:].ne(0).any()):
        raise ValueError('Audit requires forward-only context with zero backward half')
    valid = umask.T.bool()
    active = torch.cat((valid[..., None], valid[..., None] & ~availability.bool()), -1)
    if bool(source_reads[~active].ne(0).any()):
        raise ValueError('Inactive Gap/padding must have zero source reads')
    torch.testing.assert_close(source_reads.sum(-2), contexts[..., :fd], rtol=5e-5, atol=1e-5)
    torch.testing.assert_close(local, emotion_input[..., :ld], rtol=0, atol=0)
    skip = osram.local_skip(local)

    def classify(value):
        hidden = osram.emotion_norm(skip + osram.emotion_adapter(value))
        hidden = hidden * valid[..., None].to(hidden.dtype)
        return head(hidden)

    reconstructed = classify(emotion_input)
    # This also rejects unexpected extra classification/context-fusion paths.
    torch.testing.assert_close(reconstructed, full_logits, rtol=0, atol=0)
    removed = []
    for row in range(4):
        values = []
        for source in range(3):
            candidate = emotion_input.clone()
            start = ld + row * 2 * fd
            candidate[..., start:start+fd] -= source_reads[..., row, source, :]
            values.append(classify(candidate))
        removed.append(torch.stack(values, -2))
    coalitions = []
    for bits in range(8):
        if bits == 7:
            # Full coalition is the ORIGINAL read (not a floating point
            # reconstruction of it). Sum-source reconstruction error is checked.
            coalitions.append(full_logits)
            continue
        candidate = emotion_input.clone()
        slots = candidate[..., ld:].reshape(length, batch, 4, 2 * fd)
        kept = [s for s in range(3) if bits & (1 << s)]
        slots[..., :fd] = (source_reads[..., kept, :].sum(-2)
                           if kept else torch.zeros_like(contexts[..., :fd]))
        coalitions.append(classify(candidate))
    return dict(logits_removed=torch.stack(removed, -3),
                logits_coalitions=torch.stack(coalitions, -2), row_active=active)
