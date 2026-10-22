"""Direct V3 replaces active read evidence, never the original Flat/Local Skip."""
import io
from dataclasses import replace

import pytest
import torch

from gcnet_missing_m3.osram import OSRAMBackbone
from gcnet_missing_m3.source_role_v3 import HierarchicalSourceRoleV3
from test_osram import _inputs
from test_source_role_v3 import inputs


def make(**kwargs):
    return OSRAMBackbone(latent_dim=8, output_dim=10, num_heads=2,
        key_dim=3, value_dim=4, n_speakers=2, dropout=.2,
        bidirectional=False, write_step=.6, **kwargs)


def test_residual_default_rng_state_and_disabled_forward_remain_exact():
    torch.manual_seed(66)
    old = make()
    rng = torch.get_rng_state().clone()
    torch.manual_seed(66)
    explicit = make(osram_source_role_v3_mode='residual')
    assert torch.equal(rng, torch.get_rng_state())
    assert old.state_dict().keys() == explicit.state_dict().keys()
    explicit.load_state_dict(old.state_dict(), strict=True)
    args = _inputs()
    state = torch.get_rng_state().clone()
    expected, _ = old(*args)
    after = torch.get_rng_state().clone()
    torch.set_rng_state(state)
    actual, _ = explicit(*args)
    assert torch.equal(expected, actual)
    assert torch.equal(after, torch.get_rng_state())


def test_direct_decoder_is_random_nonzero_without_extra_parameters_or_rng_effect():
    torch.manual_seed(66)
    residual = make(osram_source_role_v3=True)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(66)
    direct = make(osram_source_role_v3=True, osram_source_role_v3_mode='direct')
    assert torch.equal(rng, torch.get_rng_state())
    assert residual.state_dict().keys() == direct.state_dict().keys()
    for name, value in residual.state_dict().items():
        if name.startswith('source_role_v3.decoders.'):
            assert torch.count_nonzero(direct.state_dict()[name]) > 0
        else:
            assert torch.equal(value, direct.state_dict()[name]), name


def test_direct_replaces_only_guarded_forward_evidence_and_keeps_raw_contexts():
    torch.manual_seed(66)
    model = make(osram_source_role_v3=True, osram_source_role_v3_mode='direct').eval()
    for role, decoder in enumerate(model.source_role_v3.decoders):
        with torch.no_grad():
            decoder.weight.zero_()
            decoder.bias.fill_(role + 2.)
    features, skip_inputs, scans = [], [], []
    hook = model.emotion_adapter.register_forward_pre_hook(
        lambda _, args: features.append(args[0].detach().clone()))
    skip_hook = model.local_skip.register_forward_pre_hook(
        lambda _, args: skip_inputs.append(args[0].detach().clone()))
    original_scan = model._scan
    def scan(*args, **kwargs):
        scans.append(1)
        return original_scan(*args, **kwargs)
    model._scan = scan
    args = _inputs()
    hidden, raw = model(*args)
    hook.remove()
    skip_hook.remove()
    assert len(scans) == 1
    valid = args[4].T.bool()
    enabled = valid & model.last_source_trace['trajectory'][..., 3].bool().any(-1)
    enabled &= ~args[2].bool().all(-1)
    base = features[0][..., 8:24]
    gap = features[0][..., 24:].reshape(4, 2, 3, 16)
    expected_base = torch.where(enabled[..., None], torch.full_like(base[..., :8], 2.), raw['base'][..., :8])
    torch.testing.assert_close(base[..., :8], expected_base, rtol=0, atol=0)
    for role in range(3):
        active = enabled & ~args[2][..., role].bool()
        expected = torch.where(active[..., None], torch.full_like(gap[..., role, :8], role + 3.), raw['gap'][..., role, :8])
        torch.testing.assert_close(gap[..., role, :8], expected, rtol=0, atol=0)
    assert not base[..., 8:].any() and not gap[..., 8:].any()
    assert not hidden[3].any()
    assert torch.equal(skip_inputs[0], raw['local'])
    assert torch.equal(features[0][..., :8], raw['local'])
    # The contexts returned by OSRAM remain the raw shared-memory reads.
    assert not torch.equal(raw['base'][enabled, :8], expected_base[enabled])


@pytest.mark.parametrize('ablation', ['local-only', 'local-base', 'local-gap'])
def test_direct_does_not_bypass_emotion_readout_ablations(ablation):
    model = make(osram_source_role_v3=True, osram_source_role_v3_mode='direct').eval()
    model.osram_emotion_ablation = ablation
    features = []
    h = model.emotion_adapter.register_forward_pre_hook(lambda _, args: features.append(args[0]))
    model(*_inputs())
    h.remove()
    if ablation in ('local-only', 'local-gap'):
        assert not features[0][..., 8:24].any()
    if ablation in ('local-only', 'local-base'):
        assert not features[0][..., 24:].any()


def test_direct_finite_gradients_causality_and_checkpoint_roundtrip():
    model = make(osram_source_role_v3=True, osram_source_role_v3_mode='direct').eval()
    with torch.no_grad():
        model.emotion_adapter[-1].weight.normal_(std=.03)
    args = _inputs()
    output, _ = model(*args)
    output[..., 0].square().sum().backward()
    for name, parameter in model.source_role_v3.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert model.source_role_v3.memory_projection.weight.grad.abs().sum() > 0
    changed = list(args)
    changed[0] = args[0].clone()
    changed[0][2] += 5
    changed[1] = {name: value.clone() for name, value in args[1].items()}
    for value in changed[1].values():
        value[2] += 7
    altered, _ = model(*changed)
    assert torch.equal(output[:2], altered[:2])
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored = make(osram_source_role_v3=True, osram_source_role_v3_mode='direct').eval()
    restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    assert torch.equal(output, restored(*args)[0])


def test_direct_mode_validation_and_config_cli_model_factory():
    from gcnet_missing_m3.train_gcnet import TrainConfig, build_parser
    from experiments.osram_meld_flat_source_combinations_20261013.evaluate import _build_model
    assert TrainConfig().osram_source_role_v3_mode == 'residual'
    for bad in ('invalid', None):
        with pytest.raises(ValueError, match='mode'):
            make(osram_source_role_v3_mode=bad)
        with pytest.raises(ValueError, match='mode'):
            HierarchicalSourceRoleV3(mode=bad)
        with pytest.raises(ValueError, match='mode'):
            replace(TrainConfig(), osram_source_role_v3_mode=bad)
    with pytest.raises(ValueError, match='requires'):
        make(osram_source_role_v3_mode='direct')
    with pytest.raises(ValueError, match='requires'):
        replace(TrainConfig(), osram_source_role_v3_mode='direct')
    required = ['--audio-feature', 'a', '--text-feature', 't', '--video-feature', 'v', '--output-dir', '/tmp/not-used-v3-direct']
    parsed = build_parser().parse_args(required + ['--osram-source-role-v3', '--osram-source-role-v3-mode', 'direct'])
    assert parsed.osram_source_role_v3_mode == 'direct'
    config = replace(TrainConfig(), dataset='CMUMOSI', backbone_type='osram',
        training_objective='emotion-only', disable_unused_aux_modules=True,
        latent_dim=8, osram_output_dim=10, osram_num_heads=2,
        osram_key_dim=3, osram_value_dim=4, osram_bidirectional=False,
        osram_source_role_v3=True, osram_source_role_v3_mode='direct')
    full_model = _build_model(config, (4, 5, 6), torch.device('cpu'))
    assert full_model.osram.osram_source_role_v3_mode == 'direct'
    assert full_model.osram.source_role_v3.mode == 'direct'


def test_direct_standalone_masks_poisoned_inactive_evidence_and_padding():
    model = HierarchicalSourceRoleV3(local_dim=5, forward_dim=7, mode='direct')
    args = inputs(4, 2)
    args[3][0, ..., 3] = 0
    args[4][1] = 1
    args[5][1, 3] = 0
    args[2][:, :, 0] = float('nan')  # observed A -> inactive Gap-A
    output = model(*args)
    assert torch.isfinite(output).all()
    assert output[2].abs().sum() > 0
    assert not output[0].any() and not output[1].any()
    assert not output[3, 1].any() and not output[:, :, 1].any()
