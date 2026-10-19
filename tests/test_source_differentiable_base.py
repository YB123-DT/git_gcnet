"""Opt-in gradients through original joint-Value Base source reads."""

import copy

import pytest
import torch

from gcnet_missing_m3.tasr import JointSourceTracker
from test_osram import _inputs
from test_source_evidence_trace import _model


def _finite_nonzero(tensor):
    assert tensor is not None
    assert torch.isfinite(tensor).all()
    assert tensor.abs().sum() > 0


def test_tracker_differentiable_base_preserves_sum_and_past_write_gradients():
    torch.manual_seed(815)
    availability = torch.tensor([[[1., 1., 1.]], [[1., 0., 1.]], [[0., 1., 0.]]], dtype=torch.float64)
    valid = torch.ones(3, 1, dtype=torch.bool)
    memory = torch.zeros(1, 2, 4, 3, dtype=torch.float64)
    tracker = JointSourceTracker(availability, valid, memory, audit=True, differentiable_base=True)
    alpha = torch.tensor([.95, .97], dtype=torch.float64, requires_grad=True)
    keys, values, queries = [], [], []
    for t in range(3):
        tracker.decay(alpha, valid[t])
        memory = memory * alpha[None, :, None, None]
        query = torch.randn(1, 2, 3, dtype=torch.float64, requires_grad=True)
        queries.append(query)
        base = (memory @ query[..., None]).squeeze(-1).reshape(1, 8)
        reads = tracker.read_base(t, query, base, valid[t], memory)
        torch.testing.assert_close(reads.sum(1), base, atol=1e-12, rtol=1e-12)
        if t == 0:
            assert not reads.any()
        k = torch.randn(1, 2, 3, 3, dtype=torch.float64, requires_grad=True)
        v = torch.randn(1, 2, 4, 3, dtype=torch.float64, requires_grad=True)
        keys.append(k)
        values.append(v)
        mask = availability[t, :, None, None]
        kb, vb = k * mask, v * mask
        p = torch.linalg.solve(kb.transpose(-1, -2) @ kb + .1 * torch.eye(3, dtype=torch.float64), kb.transpose(-1, -2))
        memory = memory + .6 * (vb - memory @ kb) @ p
        tracker.update(t, kb, vb, p, .6, valid[t], memory)
    trace = tracker.finish()
    trace['source_base'][2].square().sum().backward()
    for tensor in (keys[0].grad, values[0].grad, queries[2].grad, alpha.grad):
        _finite_nonzero(tensor)
    assert not trace['source_base'][0].any()
    assert not trace['state_source_norms'].requires_grad
    assert not trace['state_joint_norm'].requires_grad


def test_legacy_tracker_base_read_remains_detached():
    availability = torch.ones(2, 1, 3)
    valid = torch.ones(2, 1, dtype=torch.bool)
    memory = torch.randn(1, 2, 4, 3, requires_grad=True)
    tracker = JointSourceTracker(availability, valid, memory, audit=True)
    tracker.states = torch.randn(1, 3, 2, 4, 3, requires_grad=True)
    query = torch.randn(1, 2, 3, requires_grad=True)
    reads = tracker.read_base(1, query, torch.zeros(1, 8), valid[1], memory)
    assert not reads.requires_grad
    assert not tracker.finish()['source_base'].requires_grad


def test_osram_source_grad_is_opt_in_and_does_not_change_outputs_rng_or_state_keys():
    torch.manual_seed(816)
    original = _model(dropout=.2).train()
    original.osram_source_audit = True
    enabled = copy.deepcopy(original)
    enabled.osram_source_grad = True
    args = _inputs()
    initial = torch.get_rng_state().clone()
    expected, original_context = original(*args)
    expected_rng = torch.get_rng_state().clone()
    torch.set_rng_state(initial)
    actual, context = enabled(*args)
    assert torch.equal(expected_rng, torch.get_rng_state())
    assert torch.equal(expected, actual)
    assert list(original.state_dict()) == list(enabled.state_dict())
    assert all(torch.equal(original_context[k], context[k]) for k in context)
    trace = enabled.last_source_trace
    assert trace['source_base'].requires_grad
    assert not original.last_source_trace['source_base'].requires_grad
    torch.testing.assert_close(trace['source_base'].sum(2), context['base'][..., :8], atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(trace['source_gap'].sum(3), context['gap'][..., :8], atol=3e-6, rtol=3e-5)
    # Consuming the complete captured sequence after all in-place slot fills
    # must not trigger autograd version errors through multi-step writes.
    loss = trace['source_base'].square().sum() + trace['source_gap'].square().sum()
    loss.backward()
    for parameter in (enabled.key_projectors['audio'].weight,
                      enabled.value_projectors['audio'].weight,
                      enabled.query_projector.weight, enabled.alpha_logits):
        _finite_nonzero(parameter.grad)
    assert not trace['source_base'][0].any()
    assert not trace['source_base'][3].any()


def test_differentiable_base_requires_capture_and_tracking():
    with pytest.raises(ValueError, match='audit'):
        JointSourceTracker(torch.ones(1, 1, 3), torch.ones(1, 1, dtype=torch.bool),
                           torch.zeros(1, 2, 4, 3), differentiable_base=True)
    model = _model()
    model.osram_source_grad = True
    with pytest.raises(ValueError, match='source.*audit'):
        model(*_inputs())
