"""Read-only 4x3 Value-source audit; no alternative memory or classifier."""

import copy

import pytest
import torch

from gcnet_missing_m3.osram import OSRAMBackbone
from gcnet_missing_m3.tasr import JointSourceTracker
from test_osram import _inputs


def _model(**kwargs):
    return OSRAMBackbone(latent_dim=8, output_dim=10, num_heads=2,
                         key_dim=3, value_dim=4, n_speakers=2,
                         dropout=kwargs.pop('dropout', 0.), bidirectional=False,
                         write_step=.6, osram_source_tracking=True, **kwargs)


def test_audit_records_base_and_prewrite_state_norms_without_touching_outputs():
    torch.manual_seed(81)
    original = _model().eval()
    torch.nn.init.normal_(original.emotion_adapter[-1].weight, std=.1)
    audit = copy.deepcopy(original)
    audit.osram_source_audit = True
    args = _inputs()
    with torch.no_grad():
        expected, ec = original(*args)
        actual, ac = audit(*args)
    assert torch.equal(expected, actual)
    assert all(torch.equal(ec[k], ac[k]) for k in ec)
    trace = audit.last_source_trace
    assert 'source_base' in trace, 'audit must capture the three missing Base source slots'
    assert trace['source_base'].shape == (4, 2, 3, 8)
    assert trace['state_source_norms'].shape == (4, 2, 3)
    assert trace['state_joint_norm'].shape == (4, 2)
    torch.testing.assert_close(trace['source_base'].sum(2), ac['base'][..., :8], atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(trace['source_gap'].sum(3), ac['gap'][..., :8], atol=3e-6, rtol=3e-5)
    for key in ('source_base', 'source_gap', 'state_source_norms', 'state_joint_norm'):
        assert not trace[key][0].any(), 'first utterance must have no historical evidence'
        assert not trace[key][3].any(), 'padding must not contribute to audit summaries'
    assert trace['diagnostics']['base_sum_max_abs_error'] < 3e-6
    assert trace['diagnostics']['prewrite_memory_sum_max_abs_error'] < 3e-6
    assert 'source_base' not in original.last_source_trace


def test_audit_retains_rng_parameter_keys_and_strict_legacy_restore():
    torch.manual_seed(82)
    original = _model(dropout=.2).train()
    audit = copy.deepcopy(original)
    audit.osram_source_audit = True
    state = copy.deepcopy(original.state_dict())
    audit.load_state_dict(state, strict=True)
    assert list(audit.state_dict()) == list(state)
    args = _inputs()
    before = torch.get_rng_state().clone()
    expected = original(*args)[0]
    after = torch.get_rng_state().clone()
    torch.set_rng_state(before)
    actual = audit(*args)[0]
    assert torch.equal(expected, actual)
    assert torch.equal(after, torch.get_rng_state())
    assert list(audit.state_dict()) == list(state)


def test_audit_sources_remain_causal_and_conversation_local():
    model = _model().eval()
    model.osram_source_audit = True
    args = _inputs()
    with torch.no_grad():
        model(*args)
        reference = {k: v.clone() for k, v in model.last_source_trace.items() if torch.is_tensor(v)}
        changed = list(args)
        changed[0] = args[0].clone()
        changed[0][2:, 0] += 15
        changed[1] = {k: v.clone() for k, v in args[1].items()}
        for value in changed[1].values():
            value[2:, 0] -= 11
        model(*changed)
    for key in ('source_base', 'source_gap', 'state_source_norms', 'state_joint_norm'):
        assert torch.equal(reference[key][:2], model.last_source_trace[key][:2])
        assert torch.equal(reference[key][:, 1], model.last_source_trace[key][:, 1])


@pytest.mark.parametrize('dtype,tolerance', [(torch.float64, 1e-12), (torch.float32, 4e-6)])
def test_base_source_reads_use_decayed_prewrite_state_and_zero_unseen_source(dtype, tolerance):
    torch.manual_seed(83)
    availability = torch.tensor([[[1., 0, 1]], [[1., 0, 1]],
                                 [[0., 1, 0]], [[1., 0, 1]], [[1., 0, 1]]], dtype=dtype)
    valid = torch.tensor([[1], [1], [1], [0], [1]], dtype=torch.bool)
    memory = torch.zeros(1, 2, 4, 3, dtype=dtype)
    tracker = JointSourceTracker(availability, valid, memory, audit=True)
    alpha = torch.tensor([.95, .98], dtype=dtype)
    for t in range(5):
        previous = tracker.states.clone()
        tracker.decay(alpha, valid[t])
        memory = torch.where(valid[t, :, None, None, None], memory * alpha[None, :, None, None], memory)
        if t == 3:
            assert torch.equal(previous, tracker.states), 'padding must not decay source states'
        q = torch.randn(1, 2, 3, dtype=dtype)
        base = (memory @ q[..., None]).squeeze(-1).reshape(1, 8)
        base = torch.where(valid[t, :, None], base, torch.zeros_like(base))
        tracker.read_base(t, q, base, valid[t], memory)
        source = tracker.source_base[t]
        torch.testing.assert_close(source.sum(1), base, atol=tolerance, rtol=tolerance)
        if t <= 2:
            assert not source[:, 1].any(), 'Text has never been written before this read'
        if t == 4:
            assert source[:, 1].norm() > 0, 'prior Text write should now supply a real source'
        expected_norms = torch.where(valid[t, :, None], tracker.states.flatten(2).norm(dim=-1), torch.zeros(1, 3, dtype=dtype))
        assert torch.equal(tracker.state_source_norms[t], expected_norms)
        scale = availability[t, :, None, None]
        k = torch.randn(1, 2, 3, 3, dtype=dtype) * scale
        v = torch.randn(1, 2, 4, 3, dtype=dtype) * scale
        p = torch.linalg.solve(k.transpose(-1, -2) @ k + .01 * torch.eye(3, dtype=dtype), k.transpose(-1, -2))
        candidate = memory + .6 * (v - memory @ k) @ p
        memory = torch.where(valid[t, :, None, None, None], candidate, memory)
        tracker.update(t, k, v, p, .6, valid[t], memory)
    trace = tracker.finish()
    assert trace['diagnostics']['base_sum_max_abs_error'] < tolerance
    assert not trace['source_base'][3].any()


def test_audit_requires_original_source_tracking_to_be_enabled():
    model = OSRAMBackbone(latent_dim=8, output_dim=10, num_heads=2,
                          key_dim=3, value_dim=4, n_speakers=2,
                          dropout=0., bidirectional=False)
    model.osram_source_audit = True
    with pytest.raises(ValueError, match='source.*tracking'):
        model(*_inputs())
