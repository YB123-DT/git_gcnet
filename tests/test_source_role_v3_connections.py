"""Contracts for three isolated V3-to-Flat connection experiments."""
import importlib.util
import io

import pytest
import torch
from torch import nn


VARIANTS = ('head-preserving', 'source-slots', 'flat-hidden')


def block_type():
    assert importlib.util.find_spec('gcnet_missing_m3.source_role_v3_connections'), (
        'The dedicated V3 connection module has not been implemented')
    from gcnet_missing_m3.source_role_v3_connections import V3ConnectionBlock
    return V3ConnectionBlock


def inputs(t=4, b=3, local_dim=5, forward_dim=16):
    trajectory = torch.ones(t, b, 3, 4)
    trajectory[..., 0] = 3
    trajectory[..., 1] = .6
    trajectory[..., 2] = 2
    return [torch.randn(t, b, local_dim), torch.randn(t, b, 3, forward_dim),
            torch.randn(t, b, 3, 3, forward_dim), trajectory,
            torch.tensor([1., 0., 0.]).expand(t, b, 3).clone(), torch.ones(b, t)]


def make(variant, open_output=True):
    torch.manual_seed(31)
    model = block_type()(local_dim=5, forward_dim=16, width=8, pair_hidden=12,
                         num_memory_heads=4, flat_hidden_dim=13, variant=variant)
    if open_output and variant != 'flat-hidden':
        with torch.no_grad():
            for layer in model.decoders.modules():
                if isinstance(layer, nn.Linear):
                    layer.weight.normal_(std=.08)
                    layer.bias.fill_(.3)
    return model


@pytest.mark.parametrize('variant,count', [
    ('head-preserving', 335518), ('source-slots', 514718), ('flat-hidden', 611998)])
def test_default_parameter_count_shapes_and_initialization(variant, count):
    torch.manual_seed(7)
    model = block_type()(variant=variant)
    output = model(*inputs(2, 1, local_dim=256, forward_dim=512))
    assert sum(p.numel() for p in model.parameters()) == count
    assert model.pair_mlp[0].normalized_shape == (343,)
    assert model.source_attention.num_heads == 4
    if variant == 'flat-hidden':
        assert output.shape == (2, 1, 1600)
        assert output.abs().sum() > 0
        assert not hasattr(model, 'decoders')
        assert model.flat_projection.bias is None
        assert model.flat_norm.normalized_shape == (256,)
    else:
        assert output.shape == (2, 1, 4, 512)
        assert torch.equal(output, torch.zeros_like(output))
    if variant == 'source-slots':
        assert len(model.decoders) == 4
        assert all(layer.in_features == 192 for layer in model.decoders)
        assert not hasattr(model, 'role_queries')
        assert not hasattr(model, 'output_attention')
    if variant == 'head-preserving':
        assert not hasattr(model, 'memory_projection')
        assert len(model.head_projections) == 8
        assert all(layer.in_features == layer.out_features == 64
                   and layer.bias is None for layer in model.head_projections)
        assert model.head_embedding.shape == (8, 64)
        assert not model.head_embedding.any()
        assert len(model.decoders) == 4
        assert all(len(row) == 8 for row in model.decoders)
        assert all(layer.in_features == layer.out_features == 64
                   for row in model.decoders for layer in row)
    role_key = 'per_role_context_norm' if variant == 'flat-hidden' else 'per_role_delta_norm'
    for key in ('delta_norm', role_key, 'active_source_mean', 'active_pair_mean'):
        assert torch.isfinite(model.last_diagnostics[key]).all()
        assert model.last_diagnostics[key].numel() <= 4
        assert not model.last_diagnostics[key].requires_grad
    assert model.last_diagnostics[role_key].shape == (4,)


@pytest.mark.parametrize('variant', VARIANTS)
def test_missing_history_atv_padding_and_inactive_gap_cannot_open_from_bias(variant):
    model = make(variant)
    args = inputs()
    args[3][0, ..., 3] = 0
    args[4][1] = 1
    args[5][1, 3] = 0
    output = model(*args)
    assert not output[0].any()
    assert not output[1].any()
    assert not output[3, 1].any()
    assert output[2].abs().sum() > 0
    if variant != 'flat-hidden':
        assert not output[..., 1, :].any()


@pytest.mark.parametrize('variant', VARIANTS)
def test_inactive_nan_sources_gaps_and_rows_do_not_affect_outputs_or_gradients(variant):
    model = make(variant)
    args = inputs()
    args[3][:, :, 2, 3] = 0
    args[3][0, ..., 3] = 0
    args[4][1] = 1
    args[5][1, 3] = 0
    expected = model(*args)
    args[1][:, :, 2] = float('nan')
    args[2][:, :, :, 2] = float('nan')
    args[2][:, :, 0] = float('nan')
    args[3][:, :, 2, :3] = float('nan')
    for index in (0, 1, 2):
        args[index][0] = float('nan')
        args[index][1] = float('nan')
        args[index][3, 1] = float('nan')
        args[index].requires_grad_()
    output = model(*args)
    torch.testing.assert_close(expected, output, rtol=0, atol=0)
    output.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    for value in args[:3]:
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert not value.grad[0].any()
        assert not value.grad[1].any()
        assert not value.grad[3, 1].any()
    assert not args[1].grad[:, :, 2].any()
    assert not args[2].grad[:, :, :, 2].any()
    assert not args[2].grad[:, :, 0].any()
    assert all(torch.isfinite(value).all() for value in model.last_diagnostics.values())


@pytest.mark.parametrize('variant', VARIANTS)
def test_pair_inputs_retain_separate_gap_values_in_shared_coordinates(variant):
    model = make(variant)
    args = inputs(1, 1)
    captured = []
    handle = model.pair_mlp.register_forward_pre_hook(
        lambda _, values: captured.append(values[0].detach().clone()))
    try:
        model(*args)
        args[2][:, :, 1] += 1
        args[2][:, :, 2] -= 1
        model(*args)
    finally:
        handle.remove()
    first, second = captured
    assert first.shape[-3:] == (3, 3, 63)
    assert not first[..., 0, :, :].any()
    assert not torch.equal(first[..., 1:, :, 8:16], second[..., 1:, :, 8:16])
    torch.testing.assert_close(first[..., 1:, :, 8:16].mean(-3),
                               second[..., 1:, :, 8:16].mean(-3))
    if variant == 'head-preserving':
        assert first.shape == (1, 1, 4, 3, 3, 63)
        source_heads = args[1].reshape(1, 1, 3, 4, 4)
        base = torch.stack([layer(source_heads[..., head, :])
                            for head, layer in enumerate(model.head_projections)], dim=2)
    else:
        assert first.shape == (1, 1, 3, 3, 63)
        base = model.memory_projection(args[1])
    for role in (1, 2):
        torch.testing.assert_close(first[..., role, :, :8], base)
        projected_gap = first[..., role, :, 8:16]
        torch.testing.assert_close(first[..., role, :, 16:24], base * projected_gap)
        torch.testing.assert_close(first[..., role, :, 24:32], (base - projected_gap).abs())


@pytest.mark.parametrize('variant', VARIANTS)
def test_layer2_zero_single_source_passthrough_and_role_isolation(variant):
    model = make(variant)
    prefix = (2, 1, 4) if variant == 'head-preserving' else (2, 1)
    tokens = torch.randn(*prefix, 4, 3, 8)
    active = torch.zeros(*prefix, 4, 3, dtype=torch.bool)
    active[0, ..., 1] = True
    out = model._mix_sources(tokens, active)
    assert torch.equal(out[0, ..., 1, :], tokens[0, ..., 1, :])
    assert not out[0, ..., [0, 2], :].any()
    assert not out[1].any()
    active[:] = True
    out = model._mix_sources(tokens, active)
    changed = tokens.clone()
    changed[..., 2, :, :] *= 3
    new = model._mix_sources(changed, active)
    assert torch.equal(out[..., [0, 1, 3], :, :], new[..., [0, 1, 3], :, :])
    assert not torch.equal(out[..., 2, :, :], new[..., 2, :, :])


def test_head_outputs_do_not_mix_memory_heads_when_local_is_fixed():
    model = make('head-preserving')
    args = inputs()
    attention_sizes = []
    hooks = [layer.register_forward_pre_hook(
        lambda _, values: attention_sizes.append(values[1].shape[-2]))
        for layer in (model.source_attention, model.output_attention)]
    try:
        expected = model(*args)
        args[1][..., :4] *= 9
        args[2][..., :4] *= 9
        changed = model(*args)
    finally:
        for handle in hooks:
            handle.remove()
    assert torch.equal(expected[..., 4:], changed[..., 4:])
    assert not torch.equal(expected[..., :4], changed[..., :4])
    assert attention_sizes and set(attention_sizes) == {3}


def test_head_identity_conditions_pairs_and_shared_final_role_queries():
    model = make('head-preserving')
    args = inputs(1, 1)
    with torch.no_grad():
        model.head_embedding.copy_(torch.arange(4.)[:, None].expand(4, 8))
    pair_inputs, queries = [], []
    handles = [model.pair_mlp.register_forward_pre_hook(
        lambda _, values: pair_inputs.append(values[0])),
        model.output_attention.register_forward_pre_hook(
            lambda _, values: queries.append(values[0]))]
    try:
        model(*args)
    finally:
        for handle in handles:
            handle.remove()
    current = model.local_projection(args[0])[0, 0]
    for head in range(4):
        expected = current + model.head_embedding[head]
        torch.testing.assert_close(pair_inputs[0][0, 0, head, 1, :, 32:40],
                                   expected.expand(3, 8))
    expected_queries = torch.stack([layer(args[0])[0, 0]
                                    for layer in model.role_queries])
    expected_queries = expected_queries[None] + model.head_embedding[:, None]
    torch.testing.assert_close(queries[0].squeeze(1), expected_queries[:, [0, 2, 3]].reshape(-1, 8))


def test_source_slots_keep_a_t_v_positions_before_four_role_decoders():
    model = make('source-slots')
    args = inputs(1, 1)
    for source in range(3):
        args[3][..., 3] = 0
        args[3][:, :, source, 3] = 1
        decoder_inputs = []
        handles = [layer.register_forward_pre_hook(
            lambda _, values: decoder_inputs.append(values[0].detach().clone()))
            for layer in model.decoders]
        try:
            model(*args)
        finally:
            for handle in handles:
                handle.remove()
        assert len(decoder_inputs) == 4
        for role in (0, 2, 3):
            slots = decoder_inputs[role].reshape(1, 1, 3, 8)
            assert slots[:, :, source].abs().sum() > 0
            assert not slots[:, :, [i for i in range(3) if i != source]].any()
        assert not decoder_inputs[1].any()


def test_flat_hidden_masks_roles_again_after_affine_layernorm():
    model = make('flat-hidden')
    args = inputs()
    args[3][0, ..., 3] = 0
    args[4][1] = 1
    args[5][1, 3] = 0
    with torch.no_grad():
        model.flat_norm.bias.fill_(3)
    captured = []
    handle = model.flat_projection.register_forward_pre_hook(
        lambda _, values: captured.append(values[0].detach().clone()))
    try:
        output = model(*args)
    finally:
        handle.remove()
    latent = captured[0].reshape(4, 3, 4, 8)
    assert not latent[:, :, 1].any()
    assert not latent[0].any() and not latent[1].any() and not latent[3, 1].any()
    assert not output[0].any() and not output[1].any() and not output[3, 1].any()
    assert latent[2, :, [0, 2, 3]].abs().sum() > 0


@pytest.mark.parametrize('variant', VARIANTS)
def test_all_parameters_connect_and_upstream_values_learn_after_zero_outputs_open(variant):
    model = make(variant, open_output=False)
    args = inputs(2, 3)
    args[4][:] = torch.eye(3)[None]
    args[1].requires_grad_()
    args[2].requires_grad_()
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    opt = torch.optim.Adam(model.parameters(), lr=.01)
    target = torch.randn_like(model(*args))
    for _ in range(4):
        opt.zero_grad()
        args[1].grad = args[2].grad = None
        (model(*args) - target).square().mean().backward()
        opt.step()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
        assert not torch.equal(before[name], parameter), name
    assert (args[1].grad.abs().sum((0, 1, 3)) > 0).all()
    assert (args[2].grad.abs().sum((0, 1, 4)) > 0).all()


@pytest.mark.parametrize('variant', VARIANTS)
def test_expected_gradient_groups_cover_only_present_trainable_parameters(variant):
    model = make(variant)
    groups = model.expected_gradient_groups()
    names = tuple(name for name, _ in model.named_parameters())
    assert isinstance(groups, tuple) and groups
    assert len(groups) == len(set(groups))
    assert all(any(name.startswith(prefix) for name in names) for prefix in groups)
    assert all(any(name.startswith(prefix) for prefix in groups) for name in names)


@pytest.mark.parametrize('variant', VARIANTS)
def test_checkpoint_restore_and_time_conversation_isolation(variant):
    model = make(variant)
    args = inputs()
    expected = model(*args)
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored = make(variant, open_output=False)
    restored.load_state_dict(torch.load(stream, weights_only=True))
    assert torch.equal(expected, restored(*args))
    args[0][2, 1] *= 10
    args[1][2, 1] *= 10
    args[2][2, 1] *= 10
    changed = model(*args)
    assert not torch.equal(expected[2, 1], changed[2, 1])
    assert torch.equal(expected[:2], changed[:2])
    assert torch.equal(expected[3:], changed[3:])
    assert torch.equal(expected[:, [0, 2]], changed[:, [0, 2]])


@pytest.mark.parametrize('name,value', [
    ('local_dim', 0), ('forward_dim', 0), ('width', 0), ('pair_hidden', 0),
    ('num_memory_heads', 0), ('flat_hidden_dim', 0), ('width', 6),
    ('forward_dim', 15), ('num_memory_heads', True), ('width', 8.),
    ('variant', 'direct'), ('variant', '')])
def test_invalid_dimensions_or_variants_are_rejected(name, value):
    kwargs = dict(local_dim=5, forward_dim=16, width=8, pair_hidden=12,
                  num_memory_heads=4, flat_hidden_dim=13, variant='head-preserving')
    kwargs[name] = value
    with pytest.raises(ValueError):
        block_type()(**kwargs)


@pytest.mark.parametrize('index', range(6))
def test_input_shapes_are_checked(index):
    model = make('source-slots')
    args = inputs()
    args[index] = args[index][..., :0]
    with pytest.raises(ValueError):
        model(*args)
