import torch

from gcnet_missing_m3.source_moe import SourceRoleMixtureOfExperts


def args():
    return [torch.randn(3, 2, 5), torch.randn(3, 2, 3, 7),
            torch.randn(3, 2, 3, 3, 7), torch.ones(3, 2, 3, 4),
            torch.tensor([1., 0., 0.]).expand(3, 2, 3).clone(), torch.ones(2, 3)]


def test_zero_init_independent_experts_and_gradients():
    m = SourceRoleMixtureOfExperts(5, 7, 11)
    x = args()
    assert torch.equal(m(*x), torch.zeros(3, 2, 11))
    assert len(m.experts) == 4
    assert len({id(e[1].weight) for e in m.experts}) == 4
    opt = torch.optim.Adam(m.parameters(), lr=.01)
    for _ in range(2):
        opt.zero_grad()
        (m(*x) - 1).square().mean().backward()
        opt.step()
    assert m.router[1].weight.grad.abs().sum() > 0
    assert all(e[1].weight.grad.abs().sum() > 0 for e in m.experts)


def test_mask_slots_atv_and_no_history():
    m = SourceRoleMixtureOfExperts(5, 7, 11)
    torch.nn.init.normal_(m.output_projection.weight)
    x = args()
    x[3][:, :, 2, 3] = 0
    clean = m(*x)
    x[1][:, :, 2] = float('nan')
    x[2][:, :, :, 2] = float('nan')
    x[2][:, :, 0] = float('nan')
    torch.testing.assert_close(clean, m(*x), rtol=0, atol=0)
    x = args()
    x[4][:] = 1
    m(*x)
    assert m.last_active[:, :, 0].all()
    assert not m.last_active[:, :, 1:].any()
    x[3][0, ..., 3] = 0
    x[5][1, 2] = 0
    y = m(*x)
    assert not m.last_active[0].any()
    assert not y[2, 1].any()
    torch.testing.assert_close(m.last_weights.sum(-1)[x[5].T.bool()], torch.ones(5))


def test_slots_not_pooled_and_checkpoint_roundtrip():
    m = SourceRoleMixtureOfExperts(5, 7, 11)
    assert m.experts[0][0].normalized_shape == (859,)
    seen = []
    h = m.experts[0].register_forward_pre_hook(lambda _, x: seen.append(x[0].detach()))
    x = args()
    m(*x)
    x[2][:, :, 1] += 2
    x[2][:, :, 2] -= 2
    m(*x)
    h.remove()
    assert not torch.equal(seen[0][..., :768], seen[1][..., :768])
    n = SourceRoleMixtureOfExperts(5, 7, 11)
    n.load_state_dict(m.state_dict(), strict=True)
    assert torch.equal(m(*x), n(*x))


def backbone(mode='off'):
    from gcnet_missing_m3.osram import OSRAMBackbone
    return OSRAMBackbone(latent_dim=8, output_dim=10, num_heads=2,
                         key_dim=3, value_dim=4, n_speakers=2,
                         dropout=.2, bidirectional=False, write_step=.6,
                         osram_source_moe_mode=mode)


def backbone_inputs():
    torch.manual_seed(7)
    return (torch.randn(4, 2, 8),
            {m: torch.randn(4, 2, 8) for m in ('audio', 'text', 'visual')},
            torch.tensor([[[1., 0, 1], [0, 1, 1]], [[1., 1, 0], [0, 0, 1]],
                          [[1., 1, 1], [1, 0, 0]], [[0., 0, 0], [0, 0, 0]]]),
            torch.tensor([[0, 1, 0, 0], [1, 0, 0, 0]]),
            torch.tensor([[1., 1, 1, 0], [1., 1, 1, 0]]), [3, 3])


def test_initial_baseline_equality_rng_one_scan_and_adapter_freeze():
    torch.manual_seed(66)
    original = backbone()
    initial = torch.get_rng_state().clone()
    x = backbone_inputs()
    for mode in ('replace', 'residual'):
        torch.manual_seed(66)
        model = backbone(mode)
        assert torch.equal(initial, torch.get_rng_state())
        for name, value in original.state_dict().items():
            assert torch.equal(value, model.state_dict()[name]), name
        assert all(p.requires_grad == (mode == 'residual') for p in model.emotion_adapter.parameters())
        original.eval(); model.eval()
        calls = []
        scan = model._scan
        def counted(*args, **kwargs):
            calls.append(1)
            return scan(*args, **kwargs)
        model._scan = counted
        expected, ec = original(*x)
        actual, ac = model(*x)
        assert len(calls) == 1
        assert torch.equal(expected, actual)
        for key in ec:
            assert torch.equal(ec[key], ac[key])
        assert model.last_source_trace['source_base'].requires_grad
        assert model.last_source_trace['source_gap'].requires_grad
        clone = backbone(mode).eval()
        clone.load_state_dict(model.state_dict(), strict=True)
        assert torch.equal(actual, clone(*x)[0])


def test_open_output_is_causal_has_writer_gradient_and_padding_zero():
    for mode in ('replace', 'residual'):
        m = backbone(mode).eval()
        torch.nn.init.normal_(m.source_moe.output_projection.weight, std=.03)
        x = backbone_inputs()
        y, _ = m(*x)
        other = list(x)
        other[0] = x[0].clone(); other[0][2] += 7
        other[1] = {k: v.clone() for k, v in x[1].items()}
        for v in other[1].values():
            v[2] += 9
        new, _ = m(*other)
        assert torch.equal(y[:2], new[:2])
        assert not y[3].any()
        y[..., 0].sum().backward()
        assert m.source_moe.router[1].weight.grad.abs().sum() > 0
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in m.value_projectors.parameters())
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())


def test_default_off_matches_prechange_source_training_and_rng():
    import subprocess
    source = subprocess.check_output(['git', 'show', '6400940:gcnet_missing_m3/osram.py'], text=True)
    namespace = {'__name__': 'gcnet_missing_m3._moe_legacy', '__package__': 'gcnet_missing_m3'}
    exec(compile(source, 'before_source_moe.py', 'exec'), namespace)
    torch.manual_seed(66)
    old = namespace['OSRAMBackbone'](latent_dim=8, output_dim=10, num_heads=2,
        key_dim=3, value_dim=4, n_speakers=2, dropout=.2, bidirectional=False, write_step=.6)
    init_rng = torch.get_rng_state().clone()
    torch.manual_seed(66); new = backbone()
    assert torch.equal(init_rng, torch.get_rng_state())
    new.load_state_dict(old.state_dict(), strict=True)
    x = backbone_inputs(); rng = torch.get_rng_state().clone()
    a, _ = old(*x); a[..., 0].sum().backward(); end = torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    b, _ = new(*x); b[..., 0].sum().backward()
    assert torch.equal(a, b) and torch.equal(end, torch.get_rng_state())
    for name, p in old.named_parameters():
        q = dict(new.named_parameters())[name]
        assert (p.grad is None) == (q.grad is None)
        if p.grad is not None:
            assert torch.equal(p.grad, q.grad), name


def test_source_moe_rejects_stacked_or_incompatible_readout():
    from gcnet_missing_m3.osram import OSRAMBackbone
    for extra in ({'osram_source_role_v3': True},
                  {'osram_meaningful_block': 'some-other-block'},
                  {'osram_readout_fusion': 'not-flat'},
                  {'osram_disable_local_skip': True},
                  {'osram_ablation': 'local-only'}):
        try:
            OSRAMBackbone(osram_source_moe_mode='replace', **extra)
        except ValueError:
            pass
        else:
            raise AssertionError(f'incompatible source MoE accepted: {extra}')
