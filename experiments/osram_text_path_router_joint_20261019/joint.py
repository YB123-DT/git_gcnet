"""Single-scan, graph-connected three-candidate Text-source probability routing.

Uniform router initialization is a probability mixture, NOT baseline identity.
Only the final Flat is replayed: no second view, scan, query or task head exists.
"""
from contextlib import contextmanager

import torch
from torch import nn

from experiments.osram_tasr_20261018 import runner_helpers as helper


@contextmanager
def _capture_graph_flat(backbone):
    """Capture actual graph nodes and the original adapter's dropout RNG."""
    captured = {}

    def local(module, inputs, output):
        if 'local' in captured:
            raise RuntimeError('Expected exactly one original Local Skip invocation')
        captured['local'], captured['skip'] = inputs[0], output

    def adapter(module, inputs):
        if 'emotion_input' in captured:
            raise RuntimeError('Expected exactly one original Flat Adapter invocation')
        captured['emotion_input'] = inputs[0]
        captured['adapter_rng'] = helper.rng_state()

    handles = [backbone.osram.local_skip.register_forward_hook(local),
               backbone.osram.emotion_adapter.register_forward_pre_hook(adapter)]
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


class JointTextPathModel(nn.Module):
    """Train original CHM/Flat/head and a conditional router in a single stage."""

    def __init__(self, backbone, router):
        super().__init__()
        if router.detach_inputs:
            raise ValueError('One-stage Router requires detach_inputs=False')
        self.backbone = backbone
        self.router = router
        self.backbone.osram.osram_source_tracking = True
        self.backbone.osram.osram_source_audit = True
        self.backbone.osram.osram_source_grad = True

    def forward(self, view):
        with _capture_graph_flat(self.backbone) as captured:
            full = helper.forward(self.backbone, view)
        post_full_rng = helper.rng_state()
        osram = self.backbone.osram
        local, original = captured['local'], captured['emotion_input']
        length, batch, local_dim = local.shape
        forward_dim = osram.num_heads * osram.value_dim
        if original.shape != (length, batch, local_dim + 8 * forward_dim):
            raise ValueError('Expected original Flat Local plus four full contexts')
        if not torch.equal(local, original[..., :local_dim]):
            raise ValueError('Original Flat and Local Skip must share unchanged Local')
        if original[..., local_dim:].reshape(length, batch, 4, 2 * forward_dim)[..., forward_dim:].ne(0).any():
            raise ValueError('Only causal forward evidence is supported')
        trace = osram.last_source_trace
        base_text = trace['source_base'][..., 1, :]
        gap_text = trace['source_gap'][..., 1, 1, :]
        trajectory = trace['trajectory'][..., 1, :]
        valid = view['umask'].T.bool()
        eligible = valid & ~view['availability'][..., 1].bool() & trajectory[..., 3].bool()
        if (self.training and torch.is_grad_enabled() and eligible.any()
                and (not base_text.requires_grad or not gap_text.requires_grad)):
            raise RuntimeError('One-stage source traces are detached from the CHM graph')
        candidates = [full]
        try:
            for row, source in ((2, gap_text), (0, base_text)):
                modified = original.clone()
                start = local_dim + row * 2 * forward_dim
                modified[..., start:start + forward_dim] = (
                    modified[..., start:start + forward_dim]
                    - torch.where(eligible[..., None], source, torch.zeros_like(source)))
                # Same shape, same RNG: all candidates see the same dropout mask.
                helper.restore_rng(captured['adapter_rng'])
                hidden = osram.emotion_norm(captured['skip'] + osram.emotion_adapter(modified))
                # Match production multiplication and head-input strides exactly.
                hidden = hidden * valid[..., None].to(hidden.dtype)
                logits = self.backbone.smax_fc(hidden)
                candidates.append(torch.where(eligible[..., None], logits, full))
        finally:
            # Replaying readouts must not perturb future sampler/dropout draws.
            helper.restore_rng(post_full_rng)
        candidates = torch.stack(candidates, -2)
        log_probs, weights = self.router(
            local.reshape(-1, local_dim), base_text.reshape(-1, forward_dim),
            gap_text.reshape(-1, forward_dim), trajectory.reshape(-1, 4),
            candidates.reshape(-1, 3, full.shape[-1]), eligible.reshape(-1))
        return dict(log_probs=log_probs.reshape_as(full), candidates=candidates,
                    weights=weights.reshape(length, batch, 3), eligible=eligible)
