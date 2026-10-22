"""Capacity knobs keep the original V3 default graph and checkpoint contract."""
import io
import subprocess
from dataclasses import replace

import pytest
import torch

from gcnet_missing_m3.source_role_v3 import HierarchicalSourceRoleV3
from gcnet_missing_m3.osram import OSRAMBackbone
from test_osram import _inputs
from test_source_role_v3 import inputs


def test_default_matches_original_parameters_rng_outputs_and_gradients():
    source = subprocess.check_output(
        ['git', 'show', '98bdb5f:gcnet_missing_m3/source_role_v3.py'], text=True)
    namespace = {'__name__': '_original_source_role_v3'}
    exec(compile(source, 'original_source_role_v3.py', 'exec'), namespace)
    torch.manual_seed(66)
    original = namespace['HierarchicalSourceRoleV3'](local_dim=5, forward_dim=7)
    original_rng = torch.get_rng_state().clone()
    torch.manual_seed(66)
    current = HierarchicalSourceRoleV3(local_dim=5, forward_dim=7)
    assert torch.equal(original_rng, torch.get_rng_state())
    assert original.state_dict().keys() == current.state_dict().keys()
    for name, value in original.state_dict().items():
        assert torch.equal(value, current.state_dict()[name]), name
    with torch.no_grad():
        for decoder in original.decoders:
            decoder.weight.normal_(std=.01)
    current.load_state_dict(original.state_dict(), strict=True)
    args = inputs()
    expected, actual = original(*args), current(*args)
    assert torch.equal(expected, actual)
    expected.square().sum().backward()
    actual.square().sum().backward()
    for name, parameter in original.named_parameters():
        assert torch.equal(parameter.grad, current.get_parameter(name).grad), name


def test_wide_zero_init_masks_learning_and_checkpoint_roundtrip():
    model = HierarchicalSourceRoleV3(local_dim=5, forward_dim=7,
                                    width=256, pair_hidden=512)
    assert model.pair_mlp[0].normalized_shape == (1303,)
    assert model.pair_mlp[1].out_features == 512
    assert model.source_attention.embed_dim == 256
    assert model.source_attention.num_heads == 4
    args = inputs(4, 2)
    args[3][0, ..., 3] = 0
    args[4][1] = 1
    args[5][1, 3] = 0
    initial = model(*args)
    assert not initial.any()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    target = torch.randn_like(initial)
    for _ in range(3):
        optimizer.zero_grad()
        (model(*args) - target).square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.parameters())
        optimizer.step()
    assert model.memory_projection.weight.grad.abs().sum() > 0
    output = model(*args)
    assert not output[0].any() and not output[1].any()
    assert not output[3, 1].any() and not output[:, :, 1].any()
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored = HierarchicalSourceRoleV3(local_dim=5, forward_dim=7,
                                       width=256, pair_hidden=512)
    restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    assert torch.equal(output, restored(*args))


@pytest.mark.parametrize('name,value', [
    ('width', 0), ('width', 3), ('width', True), ('width', 64.),
    ('pair_hidden', 0), ('pair_hidden', True), ('pair_hidden', 128.),
])
def test_invalid_capacities_rejected(name, value):
    with pytest.raises(ValueError):
        HierarchicalSourceRoleV3(**{name: value})


@pytest.mark.parametrize('name,value', [
    ('osram_source_role_v3_dim', 0), ('osram_source_role_v3_dim', 3),
    ('osram_source_role_v3_dim', True), ('osram_source_role_v3_dim', 64.),
    ('osram_source_role_v3_pair_hidden', 0),
    ('osram_source_role_v3_pair_hidden', True),
    ('osram_source_role_v3_pair_hidden', 128.),
])
def test_train_config_rejects_invalid_capacities_even_when_v3_disabled(name, value):
    from gcnet_missing_m3.train_gcnet import TrainConfig
    with pytest.raises(ValueError, match=name):
        TrainConfig(**{name: value})


def test_config_cli_and_factory_wire_capacity_without_changing_flat_width():
    from gcnet_missing_m3.train_gcnet import TrainConfig, build_parser
    from experiments.osram_meld_flat_source_combinations_20261013.evaluate import _build_model
    defaults = TrainConfig()
    assert defaults.osram_source_role_v3_dim == 64
    assert defaults.osram_source_role_v3_pair_hidden == 128
    required = ['--audio-feature', 'a', '--text-feature', 't',
                '--video-feature', 'v', '--output-dir', '/tmp/unused-v3-capacity']
    parsed = build_parser().parse_args(required + [
        '--osram-source-role-v3', '--osram-source-role-v3-dim', '256',
        '--osram-source-role-v3-pair-hidden', '512'])
    assert parsed.osram_source_role_v3_dim == 256
    assert parsed.osram_source_role_v3_pair_hidden == 512
    config = replace(defaults, dataset='CMUMOSI', backbone_type='osram',
        training_objective='emotion-only', disable_unused_aux_modules=True,
        latent_dim=8, osram_output_dim=10, osram_num_heads=2,
        osram_key_dim=3, osram_value_dim=4, osram_bidirectional=False,
        osram_adapter_hidden_dim=6, osram_source_role_v3=True,
        osram_source_role_v3_dim=256, osram_source_role_v3_pair_hidden=512)
    model = _build_model(config, (4, 5, 6), torch.device('cpu'))
    assert model.osram.source_role_v3.source_attention.embed_dim == 256
    assert model.osram.source_role_v3.pair_mlp[1].out_features == 512
    assert model.osram.emotion_adapter[1].out_features == 6
    assert model.osram.output_dim == 10


def test_disabled_capacity_has_no_parameter_or_rng_effect():
    kwargs = dict(latent_dim=8, output_dim=10, num_heads=2, key_dim=3,
                  value_dim=4, n_speakers=2, dropout=.2, bidirectional=False)
    torch.manual_seed(66)
    original = OSRAMBackbone(**kwargs)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(66)
    disabled = OSRAMBackbone(**kwargs, osram_source_role_v3_dim=256,
                            osram_source_role_v3_pair_hidden=512)
    assert torch.equal(rng, torch.get_rng_state())
    assert original.state_dict().keys() == disabled.state_dict().keys()
    for name, value in original.state_dict().items():
        assert torch.equal(value, disabled.state_dict()[name])


def test_wide_v3_zero_init_preserves_smaller_flat_forward_rng_and_single_scan():
    kwargs = dict(latent_dim=8, output_dim=10, num_heads=2, key_dim=3,
                  value_dim=4, n_speakers=2, dropout=.2, bidirectional=False,
                  osram_adapter_hidden_dim=6)
    torch.manual_seed(66)
    flat = OSRAMBackbone(**kwargs)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(66)
    wide = OSRAMBackbone(**kwargs, osram_source_role_v3=True,
        osram_source_role_v3_dim=256, osram_source_role_v3_pair_hidden=512)
    assert torch.equal(rng, torch.get_rng_state())
    with torch.no_grad():
        flat.emotion_adapter[-1].weight.normal_(std=.03)
    state = wide.state_dict()
    state.update(flat.state_dict())
    wide.load_state_dict(state, strict=True)
    calls = []
    original_scan = wide._scan
    def scan(*args, **kwargs):
        calls.append(1)
        return original_scan(*args, **kwargs)
    wide._scan = scan
    args = _inputs()
    before = torch.get_rng_state().clone()
    expected, _ = flat(*args)
    after = torch.get_rng_state().clone()
    torch.set_rng_state(before)
    actual, _ = wide(*args)
    assert torch.equal(expected, actual)
    assert torch.equal(after, torch.get_rng_state())
    assert len(calls) == 1
