"""Role/source slots must remain independently accessible before attention."""
import io

import pytest
import torch


def cls():
    from gcnet_missing_m3 import source_role_tokens
    return source_role_tokens.SourceRoleTokenAttention


def inputs(t=3, b=2):
    trajectory = torch.ones(t, b, 3, 4)
    trajectory[..., 0] = 3.
    trajectory[..., 1] = .6
    trajectory[..., 2] = 2.
    return [torch.randn(t, b, 5), torch.randn(t, b, 3, 7),
            torch.randn(t, b, 3, 3, 7), trajectory,
            torch.zeros(t, b, 3), torch.ones(b, t)]


def make(open_output=True):
    torch.manual_seed(31)
    model = cls()(local_dim=5, forward_dim=7, output_dim=11)
    if open_output:
        with torch.no_grad():
            model.output_projection.weight.normal_(std=.08)
            model.output_projection.bias.fill_(.3)
    return model


def test_twelve_independent_role_major_tokens_reach_self_and_cross_attention():
    model = make()
    args = inputs(t=1, b=1)
    args[4][:] = torch.tensor([1., 0., 0.])
    captured, cross, encoded = [], [], []
    handles = [
        model.source_self_attention.register_forward_pre_hook(
            lambda _, v: captured.append(v[0].detach().clone())),
        model.local_cross_attention.register_forward_pre_hook(
            lambda _, v: cross.append(v[1].detach().clone())),
        model.relation_mlp.register_forward_pre_hook(
            lambda _, v: encoded.append(v[0].detach().clone())),
    ]
    model(*args)
    for h in handles: h.remove()
    assert captured[0].shape == (1, 12, 128)
    assert cross[0].shape == (1, 12, 128)
    f = encoded[0].reshape(1, 1, 4, 3, 87)
    projected = torch.stack([model.base_projection(args[1]),
        *(p(args[2][:, :, i]) for i, p in enumerate(model.gap_projections))], dim=2)
    # Role1=Gap-A is inactive, other nine slots are independent.
    for role in (0, 2, 3):
        torch.testing.assert_close(f[:, :, role, :, :64], projected[:, :, role])
    assert not f[:, :, 1].any()
    for role in (0, 2, 3):
        torch.testing.assert_close(f[0, 0, role, :, 64:72],
                                  model.role_embedding[role].expand(3, 8))
        torch.testing.assert_close(f[0, 0, role, :, 72:80], model.source_embedding)
    weights = model.last_diagnostics['cross_attention_weights']
    assert weights.shape == (1, 1, 4, 3)
    assert not weights[:, :, 1].any()
    torch.testing.assert_close(weights.sum((-1, -2)), torch.ones(1, 1))


def test_opposing_gap_changes_cannot_disappear_before_attention():
    model = make()
    args = inputs(t=1, b=1)
    args[4][:] = torch.tensor([1., 0., 0.])
    with torch.no_grad():
        model.gap_projections[1].weight.fill_(.2)
        model.gap_projections[2].weight.fill_(.2)
    encoded = []
    handle = model.relation_mlp.register_forward_pre_hook(
        lambda _, v: encoded.append(v[0].detach().clone()))
    model(*args)
    args[2][:, :, 1] += 1.
    args[2][:, :, 2] -= 1.
    model(*args)
    handle.remove()
    a, b = [v.reshape(1, 1, 4, 3, 87) for v in encoded]
    # These exact countervailing changes would vanish under the old Gap mean.
    torch.testing.assert_close(a[:, :, 2:4, :, :64].mean(2),
                              b[:, :, 2:4, :, :64].mean(2))
    assert not torch.equal(a[:, :, 2, :, :64], b[:, :, 2, :, :64])
    assert not torch.equal(a[:, :, 3, :, :64], b[:, :, 3, :, :64])


def test_zero_init_shape_and_no_history_padding_after_output_bias():
    model = make(open_output=False)
    args = inputs()
    out = model(*args)
    assert out.shape == (3, 2, 11) and not out.any()
    with torch.no_grad(): model.output_projection.bias.fill_(1.)
    args[3][0, ..., 3] = 0
    args[5][1, 2] = 0
    out = model(*args)
    assert not out[0].any() and not out[2, 1].any()
    assert out[1].any()


def test_masked_nan_slots_metadata_and_identity_do_not_leak():
    model = make()
    args = inputs()
    args[3][0, ..., 3] = 0
    args[3][:, :, 1, 3] = 0
    args[4][:, :, 0] = 1
    args[5][1, 2] = 0
    expected = model(*args).detach()
    args[2][:, :, 0] = float('nan')
    args[1][:, :, 1] = float('nan')
    args[2][:, :, :, 1] = float('nan')
    args[3][:, :, 1, :3] = float('nan')
    for v in args[:3]: v[0] = float('nan')
    for v in args[:5]: v[2, 1] = float('nan')
    for v in args[:4]: v.requires_grad_()
    out = model(*args)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    out.square().sum().backward()
    for v in args[:4]: assert torch.isfinite(v.grad).all()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    with torch.no_grad(): model.source_embedding[1].fill_(1.e6)
    torch.testing.assert_close(model(*args), expected, rtol=0, atol=0)


def test_atv_history_uses_base_only_and_all_masked_is_finite():
    model = make()
    args = inputs()
    args[4].fill_(1.)
    args[2].fill_(float('nan'))
    assert torch.isfinite(model(*args)).all()
    weights = model.last_diagnostics['cross_attention_weights']
    assert not weights[:, :, 1:].any()
    assert weights[:, :, 0].any()
    args[3][..., 3] = 0
    for v in args[:3]: v.fill_(float('nan'))
    out = model(*args)
    assert torch.isfinite(out).all() and not out.any()


def test_all_twelve_read_slots_and_parameters_receive_finite_task_gradients():
    model = make(open_output=False)
    args = inputs()
    args[1].requires_grad_(); args[2].requires_grad_()
    opt = torch.optim.SGD(model.parameters(), lr=.1)
    for step in range(3):
        opt.zero_grad(set_to_none=True)
        args[1].grad = None; args[2].grad = None
        (model(*args) - .3).square().mean().backward()
        for name, p in model.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
        if step:
            assert (args[1].grad.abs().sum((0, 1, 3)) > 0).all()
            assert (args[2].grad.abs().sum((0, 1, 4)) > 0).all()
            assert (model.role_embedding.grad.abs().sum(-1) > 0).all()
            assert (model.source_embedding.grad.abs().sum(-1) > 0).all()
            for p in [model.base_projection.weight, *(p.weight for p in model.gap_projections),
                      model.source_self_attention.in_proj_weight,
                      model.local_cross_attention.in_proj_weight]:
                assert p.grad.abs().sum() > 0
        opt.step()


def test_checkpoint_reload_and_default_shapes():
    model = make().eval(); args = inputs()
    expected = model(*args)
    buf = io.BytesIO(); torch.save(model.state_dict(), buf); buf.seek(0)
    restored = make().eval(); restored.load_state_dict(torch.load(buf, weights_only=True), strict=True)
    torch.testing.assert_close(restored(*args), expected, rtol=0, atol=0)
    default = cls()()
    assert default.output_projection.out_features == 1600
    with pytest.raises(ValueError): cls()(local_dim=0)


def test_opt_in_layout_keeps_original_parameters_rng_and_zero_init_prediction():
    from gcnet_missing_m3.osram import OSRAMBackbone
    from test_osram import _inputs
    config = dict(latent_dim=8, output_dim=10, num_heads=2, key_dim=3,
        value_dim=4, n_speakers=2, dropout=.2, bidirectional=False, write_step=.6)
    torch.manual_seed(66); base = OSRAMBackbone(**config)
    init_rng = torch.get_rng_state().clone()
    torch.manual_seed(66)
    model = OSRAMBackbone(**config, osram_source_relation=True,
                          osram_source_relation_layout='role-tokens')
    assert isinstance(model.source_relation, cls())
    assert torch.equal(init_rng, torch.get_rng_state())
    for n, v in base.state_dict().items(): assert torch.equal(v, model.state_dict()[n])
    before = torch.get_rng_state().clone(); expected, ec = base(*_inputs())
    after = torch.get_rng_state().clone()
    torch.set_rng_state(before); actual, ac = model(*_inputs())
    assert torch.equal(expected, actual) and torch.equal(after, torch.get_rng_state())
    for n in ec: assert torch.equal(ec[n], ac[n])
    disabled = OSRAMBackbone(**config, osram_source_relation_layout='role-tokens')
    disabled.load_state_dict(base.state_dict(), strict=True)
    assert not hasattr(disabled, 'source_relation')


def test_trainconfig_factory_and_cli_layout_route():
    from dataclasses import replace
    from gcnet_missing_m3.train_gcnet import TrainConfig, build_parser
    from experiments.osram_meld_flat_source_combinations_20261013.evaluate import _build_model
    cfg = replace(TrainConfig(), dataset='CMUMOSI', backbone_type='osram',
        training_objective='emotion-only', disable_unused_aux_modules=True,
        latent_dim=8, osram_output_dim=10, osram_num_heads=2, osram_key_dim=3,
        osram_value_dim=4, osram_bidirectional=False, osram_source_relation=True,
        osram_source_relation_layout='role-tokens')
    model = _build_model(cfg, (4, 5, 6), torch.device('cpu'))
    assert isinstance(model.osram.source_relation, cls())
    parser = build_parser()
    required = ['--audio-feature', 'a', '--text-feature', 't',
                '--video-feature', 'v', '--output-dir', '/tmp/unused-parser-test']
    assert parser.parse_args(required).osram_source_relation_layout == 'source-mean'
    args = parser.parse_args(required + ['--osram-source-relation',
                                         '--osram-source-relation-layout', 'role-tokens'])
    assert args.osram_source_relation and args.osram_source_relation_layout == 'role-tokens'
    from gcnet_missing_m3.osram import OSRAMBackbone
    with pytest.raises(ValueError): OSRAMBackbone(osram_source_relation_layout='unknown')


def test_role_tokens_real_memory_source_gradients_and_strict_causality():
    from gcnet_missing_m3.osram import OSRAMBackbone
    from test_osram import _inputs
    model = OSRAMBackbone(latent_dim=8, output_dim=10, num_heads=2, key_dim=3,
        value_dim=4, n_speakers=2, dropout=0., bidirectional=False, write_step=.6,
        osram_source_relation=True, osram_source_relation_layout='role-tokens').eval()
    with torch.no_grad(): model.source_relation.output_projection.weight.normal_(std=.1)
    args = _inputs()
    expected, _ = model(*args)
    trace = model.last_source_trace
    for key in ('source_base', 'source_gap'): trace[key].retain_grad()
    expected[..., 0].sum().backward()
    for key in ('source_base', 'source_gap'):
        assert trace[key].grad is not None and torch.isfinite(trace[key].grad).all()
        assert trace[key].grad.abs().sum() > 0
    altered = list(args)
    altered[0] = args[0].clone(); altered[0][2] += 5
    altered[1] = {name: v.clone() for name, v in args[1].items()}
    for value in altered[1].values(): value[2] += 8
    actual, _ = model(*altered)
    torch.testing.assert_close(actual[:2], expected[:2], rtol=0, atol=0)
    assert not actual[3].any()
