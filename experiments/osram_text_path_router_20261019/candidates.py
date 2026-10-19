"""Three frozen Flat candidates computed from one unchanged CHM scan."""

import torch


@torch.no_grad()
def candidate_readouts(osram, head, local, emotion_input, base_text, gap_text,
                       umask, eligible, full_logits):
    """Return [full, retain Base<-T, retain Gap-T<-T] without rescanning Memory."""
    if any(module.training for module in (
            osram.local_skip, osram.emotion_adapter, osram.emotion_norm, head)):
        raise ValueError('Frozen eval readout modules required')
    if local.ndim != 3:
        raise ValueError('Need sequence-major Local [T,B,D]')
    length, batch, local_dim = local.shape
    forward_dim = osram.num_heads * osram.value_dim
    if (emotion_input.shape != (length, batch, local_dim + 8 * forward_dim)
            or base_text.shape != (length, batch, forward_dim)
            or gap_text.shape != (length, batch, forward_dim)
            or umask.shape != (batch, length)
            or eligible.shape != (length, batch) or eligible.dtype != torch.bool
            or full_logits.ndim != 3 or full_logits.shape[:2] != (length, batch)
            or bool((eligible & ~umask.T.bool()).any())):
        raise ValueError('Invalid candidate shapes or eligible padding')
    contexts = emotion_input[..., local_dim:].reshape(length, batch, 4, 2 * forward_dim)
    if bool(contexts[..., forward_dim:].ne(0).any()):
        raise ValueError('Only causal forward evidence; backward half must be zero')
    if not torch.equal(local, emotion_input[..., :local_dim]):
        raise ValueError('Candidate Local must be actual unchanged original Flat Local')
    skip = osram.local_skip(local)
    predictions = [full_logits]
    # Fixed original concat rows: Base=0, Gap-A=1, Gap-T=2, Gap-V=3.
    for row, source in ((2, gap_text), (0, base_text)):
        candidate = emotion_input.clone()
        start = local_dim + row * 2 * forward_dim
        candidate[..., start:start + forward_dim] -= torch.where(
            eligible[..., None], source, torch.zeros_like(source))
        hidden = osram.emotion_norm(skip + osram.emotion_adapter(candidate))
        hidden = torch.where(umask.T.bool()[..., None], hidden, torch.zeros_like(hidden))
        logits = head(hidden)
        predictions.append(torch.where(eligible[..., None], logits, full_logits))
    return torch.stack(predictions, dim=-2)


@torch.no_grad()
def pack_view(bundle, view):
    """Scan once, capture genuine Flat inputs, return detached valid batch-major rows."""
    from experiments.osram_tasr_20261018 import runner_helpers as helper
    from experiments.osram_12_source_evidence_audit_20261019.run import (
        capture_flat, require_same_rng,
    )

    model = bundle['model']
    if model.training:
        raise ValueError('Original backbone must remain frozen in eval mode')
    rng = helper.rng_state()
    with capture_flat(model) as captured:
        full_logits = helper.forward(model, view)
    trace = model.osram.last_source_trace
    base_text = trace['source_base'][..., 1, :]
    gap_text = trace['source_gap'][..., 1, 1, :]
    trajectory = trace['trajectory'][..., 1, :]
    availability = view['availability']
    valid = view['umask'].bool()
    eligible = valid.T & ~availability[..., 1].bool() & trajectory[..., 3].bool()
    candidates = candidate_readouts(model.osram, model.smax_fc, captured['local'],
        captured['emotion_input'], base_text, gap_text, view['umask'], eligible, full_logits)
    require_same_rng(rng, helper.rng_state())

    def flatten(value):
        return value.transpose(0, 1)[valid].detach()

    return dict(local=flatten(captured['local']), base_text=flatten(base_text),
                gap_text=flatten(gap_text), text_trajectory=flatten(trajectory),
                candidates=flatten(candidates), eligible=flatten(eligible),
                labels=view['labels'][valid].detach(), availability=flatten(availability))
