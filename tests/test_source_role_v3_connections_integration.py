"""Connection-only V3 changes retain CHM, raw contexts and original Local Skip."""
from dataclasses import replace
import subprocess

import pytest
import torch

from gcnet_missing_m3.osram import OSRAMBackbone


def _inputs():
    torch.manual_seed(7)
    local = torch.randn(4, 2, 8)
    latents = {name: torch.randn(4, 2, 8) for name in ('audio', 'text', 'visual')}
    availability = torch.tensor([[[1, 0, 1], [0, 1, 1]],
        [[1, 1, 0], [0, 0, 1]], [[1, 1, 1], [1, 0, 0]], [[0, 0, 0], [0, 0, 0]]]).float()
    qmask = torch.tensor([[0, 1, 0, 0], [1, 0, 0, 0]])
    umask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]]).float()
    return local, latents, availability, qmask, umask, [3, 3]


VARIANTS = ('head-preserving', 'source-slots', 'flat-hidden')


def make(variant='standard', enabled=False, **kwargs):
    return OSRAMBackbone(latent_dim=8, output_dim=10, num_heads=2,
        key_dim=3, value_dim=4, n_speakers=2, dropout=.2,
        bidirectional=False, write_step=.6, osram_source_role_v3=enabled,
        osram_source_role_v3_variant=variant, **kwargs)


def test_default_disabled_and_standard_v3_preserve_previous_commit():
    source = subprocess.check_output(
        ['git', 'show', '614799a:gcnet_missing_m3/osram.py'], text=True)
    ns = {'__name__': 'gcnet_missing_m3._pre_connection', '__package__': 'gcnet_missing_m3'}
    exec(compile(source, 'pre_connection.py', 'exec'), ns)
    for enabled in (False, True):
        torch.manual_seed(66)
        old = ns['OSRAMBackbone'](latent_dim=8, output_dim=10, num_heads=2,
            key_dim=3, value_dim=4, n_speakers=2, dropout=.2,
            bidirectional=False, write_step=.6, osram_source_role_v3=enabled)
        init_rng = torch.get_rng_state().clone()
        torch.manual_seed(66)
        new = make(enabled=enabled)
        assert torch.equal(init_rng, torch.get_rng_state())
        new.load_state_dict(old.state_dict(), strict=True)
        args = _inputs()
        state = torch.get_rng_state().clone()
        a, ac = old(*args)
        a[..., 0].sum().backward()
        after = torch.get_rng_state().clone()
        torch.set_rng_state(state)
        b, bc = new(*args)
        b[..., 0].sum().backward()
        assert torch.equal(a, b) and torch.equal(after, torch.get_rng_state())
        for name in ac:
            assert torch.equal(ac[name], bc[name])
        for name, p in old.named_parameters():
            q = dict(new.named_parameters())[name]
            assert (p.grad is None) == (q.grad is None)
            if p.grad is not None:
                assert torch.equal(p.grad, q.grad), name


@pytest.mark.parametrize('variant', VARIANTS)
def test_fresh_initial_predictions_rng_one_scan_and_parameter_update(variant):
    torch.manual_seed(66)
    baseline = make()
    init_rng = torch.get_rng_state().clone()
    torch.manual_seed(66)
    model = make(variant, True)
    assert torch.equal(init_rng, torch.get_rng_state())
    for name, value in baseline.state_dict().items():
        assert torch.equal(value, model.state_dict()[name]), name
    args = _inputs()
    state = torch.get_rng_state().clone()
    expected, ec = baseline(*args)
    after = torch.get_rng_state().clone()
    scan = model._scan
    calls = []
    def counted(*a, **kw):
        calls.append(1)
        return scan(*a, **kw)
    model._scan = counted
    torch.set_rng_state(state)
    actual, ac = model(*args)
    assert len(calls) == 1
    assert torch.equal(actual, expected)
    assert torch.equal(after, torch.get_rng_state())
    for name in ec:
        assert torch.equal(ec[name], ac[name])
    assert not actual[3].any()
    initial = {n: p.detach().clone() for n, p in model.source_role_v3.named_parameters()}
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    observed = set()
    for _ in range(4):
        opt.zero_grad(set_to_none=True)
        hidden, _ = model(*args)
        (hidden[..., 0] - .7).square().mean().backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), name
                if name.startswith('source_role_v3.') and p.grad.abs().sum() > 0:
                    observed.add(name)
        opt.step()
    assert observed
    assert any(not torch.equal(initial[n], p) for n, p in model.source_role_v3.named_parameters())
    model.eval()
    rebuilt = make(variant, True).eval()
    rebuilt.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        a, _ = model(*args)
        b, _ = rebuilt(*args)
    assert torch.equal(a, b)


@pytest.mark.parametrize('variant', VARIANTS)
def test_future_change_cannot_affect_past_or_padding(variant):
    model = make(variant, True).eval()
    with torch.no_grad():
        model.emotion_adapter[-1].weight.normal_(std=.03)
        # Exercise the nonzero correction path, not merely the zero-init skip.
        for name, p in model.source_role_v3.named_parameters():
            if 'decoder' in name:
                p.normal_(std=.01)
    args = _inputs()
    expected, ec = model(*args)
    changed = list(args)
    changed[0] = args[0].clone()
    changed[0][2] += 5
    changed[1] = {k: v.clone() for k, v in args[1].items()}
    for value in changed[1].values():
        value[2] += 7
    actual, ac = model(*changed)
    assert torch.equal(actual[:2], expected[:2])
    for key in ec:
        assert torch.equal(ac[key][:2], ec[key][:2])
    assert not actual[3].any()


def test_hidden_variant_uses_original_adapter_preactivation_and_raw_reads():
    model = make('flat-hidden', True).eval()
    with torch.no_grad():
        model.emotion_adapter[-1].weight.normal_(std=.03)
    captured = {}
    hooks = [
        model.emotion_adapter[0].register_forward_pre_hook(
            lambda _, a: captured.update(flat_input=a[0].detach().clone())),
        model.emotion_adapter[1].register_forward_hook(
            lambda _, a, v: captured.update(linear=v.detach().clone())),
        model.emotion_adapter[2].register_forward_pre_hook(
            lambda _, a: captured.update(gelu_input=a[0].detach().clone())),
        model.source_role_v3.register_forward_hook(
            lambda _, a, v: captured.update(branch=v.detach().clone())),
        model.local_skip.register_forward_pre_hook(
            lambda _, a: captured.update(skip=a[0].detach().clone())),
    ]
    output, contexts = model(*_inputs())
    for hook in hooks:
        hook.remove()
    assert torch.equal(captured['gelu_input'], captured['linear'] + captured['branch'])
    assert captured['branch'][1:3].abs().sum() > 0
    assert torch.equal(captured['skip'], contexts['local'])
    expected_input = torch.cat((contexts['local'], contexts['base'], contexts['gap'].flatten(2)), -1)
    assert torch.equal(captured['flat_input'], expected_input)
    assert torch.isfinite(output).all()


def test_config_cli_factory_and_rejection_of_incompatible_variants():
    from gcnet_missing_m3.train_gcnet import TrainConfig, build_parser
    from experiments.osram_meld_flat_source_combinations_20261013.evaluate import _build_model
    assert TrainConfig().osram_source_role_v3_variant == 'standard'
    required = ['--audio-feature', 'a', '--text-feature', 't', '--video-feature', 'v',
                '--output-dir', '/tmp/unused-v3-connections']
    for variant in VARIANTS:
        args = build_parser().parse_args(required + ['--osram-source-role-v3',
            '--osram-source-role-v3-variant', variant])
        assert args.osram_source_role_v3_variant == variant
        cfg = replace(TrainConfig(), dataset='CMUMOSI', backbone_type='osram',
            training_objective='emotion-only', disable_unused_aux_modules=True,
            latent_dim=8, osram_output_dim=10, osram_num_heads=2, osram_key_dim=3,
            osram_value_dim=4, osram_bidirectional=False,
            osram_source_role_v3=True, osram_source_role_v3_variant=variant)
        model = _build_model(cfg, (4, 5, 6), torch.device('cpu'))
        assert model.osram.source_role_v3.variant == variant
        with pytest.raises(ValueError):
            make(variant, False)
        with pytest.raises(ValueError):
            make(variant, True, osram_source_role_v3_mode='direct')
        with pytest.raises(ValueError):
            make(variant, True, osram_disable_local_skip=True)
    with pytest.raises(ValueError):
        make('invalid', True)
