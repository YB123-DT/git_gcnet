"""Contracts for V3 role-preserving same-source and cross-source interaction."""
import io

import pytest
import torch


def cls():
    from gcnet_missing_m3.source_role_v3 import HierarchicalSourceRoleV3
    return HierarchicalSourceRoleV3


def inputs(t=3, b=2):
    trajectory = torch.ones(t, b, 3, 4)
    trajectory[..., 0] = 3
    trajectory[..., 1] = .6
    trajectory[..., 2] = 2
    return [torch.randn(t, b, 5), torch.randn(t, b, 3, 7),
            torch.randn(t, b, 3, 3, 7), trajectory,
            torch.tensor([1., 0., 0.]).expand(t, b, 3).clone(), torch.ones(b, t)]


def make(open_output=True):
    torch.manual_seed(31)
    model = cls()(local_dim=5, forward_dim=7)
    if open_output:
        with torch.no_grad():
            for layer in model.decoders:
                layer.weight.normal_(std=.08)
                layer.bias.fill_(.3)
    return model


def test_zero_init_and_four_independent_forward_outputs():
    model = make(False)
    output = model(*inputs())
    assert output.shape == (3, 2, 4, 7)
    assert torch.equal(output, torch.zeros_like(output))
    assert len(model.decoders) == len(model.role_queries) == 4
    assert len({id(p) for p in model.decoders}) == 4
    assert model.memory_projection.bias is None
    assert model.pair_mlp[0].normalized_shape == (343,)


def test_pair_layout_shared_coordinates_and_no_early_gap_mean():
    model = make()
    args = inputs(1, 1)
    captured = []
    handle = model.pair_mlp.register_forward_pre_hook(
        lambda _, v: captured.append(v[0].detach().clone()))
    model(*args)
    args[2][:, :, 1] += 1
    args[2][:, :, 2] -= 1
    model(*args)
    handle.remove()
    first, second = captured
    assert first.shape == (1, 1, 3, 3, 343)  # gap-role, source, features
    b = model.memory_projection(args[1])
    for role in (1, 2):
        torch.testing.assert_close(first[:, :, role, :, :64], b)
        expected = model.memory_projection(args[2][:, :, role] + (1 if role == 2 else -1))
        torch.testing.assert_close(first[:, :, role, :, 64:128], expected)
        torch.testing.assert_close(first[:, :, role, :, 128:192], b * expected)
        torch.testing.assert_close(first[:, :, role, :, 192:256], (b - expected).abs())
    assert not first[:, :, 0].any()  # observed A, inactive Gap-A
    assert not torch.equal(first[:, :, 1, :, 64:128], second[:, :, 1, :, 64:128])
    assert not torch.equal(first[:, :, 2, :, 64:128], second[:, :, 2, :, 64:128])
    torch.testing.assert_close(first[:, :, 1:, :, 64:128].mean(2),
                              second[:, :, 1:, :, 64:128].mean(2))


def test_single_source_layer2_is_exact_passthrough_and_zero_source_is_zero():
    model = make()
    tokens = torch.randn(2, 1, 4, 3, 64)
    active = torch.zeros(2, 1, 4, 3, dtype=torch.bool)
    active[0, 0, :, 1] = True
    calls = []
    handle = model.source_attention.register_forward_pre_hook(lambda *_: calls.append(True))
    out = model._mix_sources(tokens, active)
    handle.remove()
    assert not calls
    assert torch.equal(out[0, 0, :, 1], tokens[0, 0, :, 1])
    assert not out[0, 0, :, [0, 2]].any() and not out[1].any()


def test_layer2_never_mixes_different_roles():
    model = make()
    tokens = torch.randn(2, 1, 4, 3, 64)
    active = torch.ones(2, 1, 4, 3, dtype=torch.bool)
    out = model._mix_sources(tokens, active)
    changed = tokens.clone()
    changed[:, :, 2] *= 3
    new = model._mix_sources(changed, active)
    assert torch.equal(out[:, :, [0, 1, 3]], new[:, :, [0, 1, 3]])
    assert not torch.equal(out[:, :, 2], new[:, :, 2])


def test_no_history_all_visible_padding_and_inactive_gap_remain_zero_after_bias():
    model = make()
    args = inputs(4, 2)
    args[3][0, ..., 3] = 0
    args[4][1] = 1
    args[5][1, 3] = 0
    out = model(*args)
    assert not out[0].any() and not out[1].any() and not out[3, 1].any()
    assert not out[:, :, 1].any()  # A present everywhere
    assert out[2, :, 0].any() and out[2, :, 2:].any()


def test_inactive_nan_slots_cannot_leak_outputs_or_gradients():
    model = make()
    args = inputs()
    args[3][:, :, 2, 3] = 0  # unseen historical V
    clean = model(*args)
    args[1][:, :, 2] = float('nan')
    args[2][:, :, :, 2] = float('nan')
    args[2][:, :, 0] = float('nan')  # observed A target
    out = model(*args)
    torch.testing.assert_close(clean, out, rtol=0, atol=0)
    out.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_pair_messages_are_kept_per_gap_and_base_aggregates_after_pair_mlp():
    model = make()
    args = inputs(1, 1)
    pair_outputs, attention_inputs = [], []
    h1 = model.pair_mlp.register_forward_hook(lambda _, __, v: pair_outputs.append(v))
    h2 = model.source_attention.register_forward_pre_hook(
        lambda _, v: attention_inputs.append(v[2].detach().clone()))
    model(*args)
    h1.remove(); h2.remove()
    values = attention_inputs[0]
    assert values.shape == (3, 3, 64)  # active Base, Gap-T, Gap-V roles
    pairs = pair_outputs[0][0, 0]
    base = model.memory_projection(args[1])[0, 0]
    torch.testing.assert_close(values[0], base + pairs[1:].mean(0))
    for index, role in enumerate((1, 2), start=1):
        gap = model.memory_projection(args[2][:, :, role])[0, 0]
        torch.testing.assert_close(values[index], gap + pairs[role])


def test_training_opens_gradients_to_all_roles_sources_and_upstream_parameters():
    model = make(False)
    args = inputs(2, 3)
    args[4][:] = torch.eye(3)[None]
    args[1].requires_grad_(); args[2].requires_grad_()
    opt = torch.optim.Adam(model.parameters(), lr=.01)
    target = torch.randn(2, 3, 4, 7)
    for _ in range(3):
        opt.zero_grad()
        if args[1].grad is not None: args[1].grad.zero_()
        if args[2].grad is not None: args[2].grad.zero_()
        (model(*args) - target).square().mean().backward()
        opt.step()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    for name in ('memory_projection.weight', 'pair_mlp.1.weight', 'source_attention.in_proj_weight'):
        assert model.get_parameter(name).grad.abs().sum() > 0
    assert (args[1].grad.abs().sum((0, 1, 3)) > 0).all()
    active_gap_gradient = args[2].grad.abs().sum((0, 1, 4))
    assert (active_gap_gradient > 0).all()


def test_state_restore_and_no_cross_time_or_conversation_mixing():
    model = make()
    args = inputs()
    expected = model(*args)
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream); stream.seek(0)
    restored = make(False); restored.load_state_dict(torch.load(stream, weights_only=True))
    assert torch.equal(expected, restored(*args))
    args[0][2, 1] *= 10; args[1][2, 1] *= 10; args[2][2, 1] *= 10
    changed = model(*args)
    assert torch.equal(expected[:2], changed[:2])
    assert torch.equal(expected[:, 0], changed[:, 0])


@pytest.mark.parametrize('name', ['local_dim', 'forward_dim'])
def test_invalid_dimensions_rejected(name):
    with pytest.raises(ValueError): cls()(**{name: 0})
