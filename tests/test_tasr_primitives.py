"""TASR algebra, causal trajectory and bounded adapter contracts."""

import importlib.util

import pytest
import torch


def api():
    assert importlib.util.find_spec('gcnet_missing_m3.tasr') is not None, 'TASR primitives are not implemented'
    from gcnet_missing_m3.tasr import JointSourceTracker, TASRAdapter, strict_past_trajectory
    return JointSourceTracker, TASRAdapter, strict_past_trajectory


def test_trajectory_is_strictly_past_and_counts_valid_steps():
    _, _, trajectory = api()
    availability = torch.tensor([[[1., 0, 1]], [[0., 1, 0]], [[1., 1, 1]], [[0., 0, 1]]])
    valid = torch.tensor([[1, 1, 0, 1]])
    result = trajectory(availability, valid)
    assert torch.equal(result[0], torch.zeros_like(result[0]))
    assert torch.equal(result[1, 0, :, 0], torch.tensor([1., 0, 1]))
    assert torch.equal(result[1, 0, :, 2], torch.tensor([1., 0, 1]))
    assert torch.equal(result[2], torch.zeros_like(result[2]))
    assert torch.equal(result[3, 0, :, 0], torch.tensor([1., 1, 1]))
    assert torch.equal(result[3, 0, :, 1], torch.tensor([.5, .5, .5]))
    assert torch.equal(result[3, 0, :, 2], torch.tensor([2., 1, 2]))
    altered = availability.clone()
    altered[3] = 1 - altered[3]
    assert torch.equal(result, trajectory(altered, valid))


@pytest.mark.parametrize('dtype,tolerance', [(torch.float64, 1e-12), (torch.float32, 5e-6)])
def test_joint_tracking_reconstructs_memory_and_gap(dtype, tolerance):
    Tracker, _, _ = api()
    torch.manual_seed(84)
    length, batch, heads, vd, kd = 7, 2, 2, 4, 3
    availability = torch.tensor([[[1, 0, 1], [1, 0, 1]], [[1, 0, 1], [1, 0, 1]],
                                 [[0, 1, 0], [0, 1, 0]], [[1, 0, 1], [1, 0, 1]],
                                 [[1, 1, 1], [1, 1, 1]], [[1, 0, 0], [1, 0, 0]],
                                 [[1, 0, 1], [1, 0, 1]]], dtype=dtype)
    valid = torch.ones(length, batch, dtype=torch.bool)
    valid[4, 1] = False
    memory = torch.zeros(batch, heads, vd, kd, dtype=dtype)
    tracker = Tracker(availability, valid, memory)
    alpha = torch.tensor([.95, .99], dtype=dtype)
    beta = torch.tensor([[.3, .7, .5], [.8, .2, .6]], dtype=dtype)
    old_text = None
    for t in range(length):
        previous = tracker.states.clone()
        tracker.decay(alpha, valid[t])
        memory = torch.where(valid[t, :, None, None, None], memory * alpha[None, :, None, None], memory)
        if not valid[t, 1]:
            assert torch.equal(previous[1], tracker.states[1])
        queries = torch.randn(batch, 3, heads, kd, dtype=dtype)
        for m in range(3):
            active = valid[t] & ~availability[t, :, m].bool()
            original = (memory @ queries[:, m, :, :, None]).squeeze(-1).reshape(batch, -1)
            original = torch.where(active[:, None], original, torch.zeros_like(original))
            tracker.read(t, m, queries[:, m], original, active)
            assert torch.allclose(tracker.source_gap[t, :, m].sum(1), original, atol=tolerance, rtol=tolerance)
        scale = availability[t, :, None, None, :] * beta.sqrt()[None, :, None, :]
        k = torch.randn(batch, heads, kd, 3, dtype=dtype) * scale
        v = torch.randn(batch, heads, vd, 3, dtype=dtype) * scale
        solved = torch.linalg.solve(k.transpose(-1, -2) @ k + .03 * torch.eye(3, dtype=dtype), k.transpose(-1, -2))
        candidate = memory + .6 * (v - memory @ k) @ solved
        memory = torch.where(valid[t, :, None, None, None], candidate, memory)
        tracker.update(t, k, v, solved, .6, valid[t], memory)
        assert torch.allclose(tracker.states.sum(1), memory, atol=tolerance, rtol=tolerance)
        if t < 2:
            assert torch.count_nonzero(tracker.states[:, 1]) == 0
        if t == 2:
            old_text = tracker.states[:, 1].clone()
            assert old_text.norm() > 0
        if t == 3:
            assert not torch.allclose(tracker.states[:, 1], old_text * alpha[None, :, None, None])
    result = tracker.finish()
    assert result['source_gap'].shape == (length, batch, 3, 3, heads * vd)
    assert torch.count_nonzero(result['source_gap'][0]) == 0
    assert torch.count_nonzero(result['source_gap'][4, 1]) == 0
    assert result['diagnostics']['memory_sum_max_abs_error'] < tolerance
    assert result['diagnostics']['gap_sum_max_abs_error'] < tolerance
    print(dtype, result['diagnostics'])


def test_tracking_is_conversation_local_and_causal():
    Tracker, _, _ = api()
    av = torch.ones(2, 2, 3)
    memory = torch.zeros(2, 1, 2, 2)
    tracker = Tracker(av, torch.ones(2, 2, dtype=torch.bool), memory)
    k = torch.zeros(2, 1, 2, 3)
    v = torch.zeros_like(k)
    k[0] = 1
    v[0] = 2
    solved = torch.linalg.solve(k.transpose(-1, -2) @ k + torch.eye(3), k.transpose(-1, -2))
    memory = .6 * v @ solved
    tracker.update(0, k, v, solved, .6, torch.ones(2, dtype=torch.bool), memory)
    assert tracker.states[0].norm() > 0
    assert torch.count_nonzero(tracker.states[1]) == 0
    assert torch.count_nonzero(tracker.source_gap) == 0


def adapter_inputs():
    _, _, trajectory = api()
    torch.manual_seed(11)
    availability = torch.tensor([[[1., 0, 1]], [[1., 0, 0]], [[1., 1, 1]], [[0., 0, 0]]])
    umask = torch.tensor([[1., 1, 1, 0]])
    local = torch.randn(4, 1, 6)
    source = torch.randn(4, 1, 3, 3, 8)
    gap = torch.randn(4, 1, 3, 16)
    hist = trajectory(availability, umask)
    return local, gap, source, availability, umask, hist


def test_adapter_zero_init_exactness_masking_and_neutral_parameter_parity():
    _, Adapter, _ = api()
    inputs = adapter_inputs()
    torch.manual_seed(7)
    real = Adapter(6, 8)
    torch.manual_seed(7)
    neutral = Adapter(6, 8, source_mode='neutral')
    assert real.state_dict().keys() == neutral.state_dict().keys()
    assert all(torch.equal(v, neutral.state_dict()[k]) for k, v in real.state_dict().items())
    output = real(*inputs)
    assert torch.equal(output, torch.zeros_like(output))
    assert real.score[-1].in_features == 64
    assert real.score[0].normalized_shape == (151,)
    assert not real.last_gamma.requires_grad
    with torch.no_grad():
        real.score[-1].bias.fill_(1.)
    local, gap, source, av, mask, hist = inputs
    unsafe_source = source.clone()
    unsafe_source[0] = torch.nan  # no history
    unsafe_source[2:] = torch.nan  # observed targets or padding
    unsafe_source[:, :, :, 1] = torch.nan  # Text never observed before active row
    unsafe_gap = gap.clone()
    unsafe_gap[0] = torch.nan
    unsafe_gap[2:] = torch.nan
    unsafe_local = local.clone()
    unsafe_local[3] = torch.nan
    output = real(unsafe_local, unsafe_gap, unsafe_source, av, mask, hist)
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output[0]) == 0
    assert torch.count_nonzero(output[2:]) == 0
    assert torch.count_nonzero(output[..., 8:]) == 0
    assert torch.count_nonzero(output[1, :, 0]) == 0  # Audio observed
    assert output[1, :, 1:, :8].norm() > 0


def test_neutral_does_not_erase_never_seen_source_and_backward_is_finite():
    _, Adapter, _ = api()
    local, gap, source, av, mask, hist = adapter_inputs()
    adapter = Adapter(6, 8, source_mode='neutral')
    with torch.no_grad():
        adapter.score[-1].bias.fill_(.5)
    output = adapter(local, gap, torch.full_like(source, float('nan')), av, mask, hist)
    expected = .25 * torch.tanh(torch.tensor(.5)) * gap[1, 0, 1, :8]
    assert torch.allclose(output[1, 0, 1, :8], expected)
    assert adapter.last_gamma[1, 0, 1, 1] > 0  # Text unseen, neutral thirds still valid
    optimizer = torch.optim.Adam(adapter.parameters(), lr=.001)
    for _ in range(2):
        optimizer.zero_grad()
        result = adapter(local, gap, source, av, mask, hist)
        result.square().sum().backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in adapter.parameters())
        optimizer.step()
    assert adapter.local_projection.weight.grad.norm() > 0
    assert adapter.memory_projection.weight.grad.norm() > 0
    clone = Adapter(6, 8, source_mode='neutral')
    clone.load_state_dict(adapter.state_dict(), strict=True)
    assert torch.equal(clone(*adapter_inputs()), adapter(*adapter_inputs()))


def test_invalid_shapes_are_rejected():
    _, Adapter, trajectory = api()
    with pytest.raises(ValueError):
        trajectory(torch.ones(2, 1, 4), torch.ones(1, 2))
    with pytest.raises(ValueError):
        Adapter(6, 8, source_mode='anything')


def test_zero_initialization_blocks_upstream_only_until_output_head_updates():
    _, Adapter, _ = api()
    adapter = Adapter(6, 8)
    inputs = adapter_inputs()
    target = torch.ones_like(inputs[1])
    optimizer = torch.optim.SGD(adapter.parameters(), lr=.05)
    optimizer.zero_grad()
    (adapter(*inputs) - target).square().sum().backward()
    assert adapter.score[-1].weight.grad.norm() > 0
    assert adapter.local_projection.weight.grad.norm() == 0
    assert adapter.memory_projection.weight.grad.norm() == 0
    optimizer.step()
    optimizer.zero_grad()
    (adapter(*inputs) - target).square().sum().backward()
    assert adapter.local_projection.weight.grad.norm() > 0
    assert adapter.memory_projection.weight.grad.norm() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in adapter.parameters())


def test_real_tracking_retains_differentiable_key_value_graph():
    Tracker, Adapter, _ = api()
    torch.manual_seed(22)
    availability = torch.tensor([[[1., 0, 1]], [[1., 0, 1]], [[1., 0, 1]]])
    valid = torch.ones(3, 1, dtype=torch.bool)
    memory = torch.zeros(1, 1, 4, 4)
    tracker = Tracker(availability, valid, memory)
    key = torch.randn(3, 1, 1, 4, 3, requires_grad=True)
    value = torch.randn(3, 1, 1, 4, 3, requires_grad=True)
    query = torch.randn(3, 1, 3, 1, 4, requires_grad=True)
    originals = []
    for t in range(3):
        tracker.decay(torch.tensor([.95]), valid[t])
        memory = .95 * memory
        gaps = []
        for m in range(3):
            missing = ~availability[t, :, m].bool()
            gap = (memory @ query[t, :, m, :, :, None]).squeeze(-1).reshape(1, 4)
            gap = torch.where(missing[:, None], gap, torch.zeros_like(gap))
            tracker.read(t, m, query[t, :, m], gap, missing)
            gaps.append(torch.cat((gap, torch.zeros_like(gap)), -1))
        originals.append(torch.stack(gaps, 1))
        k = key[t] * availability[t, :, None, None, :]
        v = value[t] * availability[t, :, None, None, :]
        p = torch.linalg.solve(k.transpose(-1, -2) @ k + .01 * torch.eye(3), k.transpose(-1, -2))
        memory = memory + .6 * (v - memory @ k) @ p
        tracker.update(t, k, v, p, .6, valid[t], memory)
    adapter = Adapter(6, 4)
    with torch.no_grad():
        adapter.score[-1].bias.fill_(.4)
    result = tracker.finish()
    delta = adapter(torch.randn(3, 1, 6), torch.stack(originals), result['source_gap'],
                    availability, valid.T, result['trajectory'])
    delta.square().sum().backward()
    for parameter in (key, value, query):
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.norm() > 0
    # Absent Text writes have no path, even though the Text Gap is active.
    assert torch.count_nonzero(key.grad[..., 1]) == 0
    assert torch.count_nonzero(value.grad[..., 1]) == 0
    assert torch.count_nonzero(key.grad[-1]) == 0  # final write cannot affect earlier reads
