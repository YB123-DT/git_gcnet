import torch

from gcnet_missing_m3.nested_concat import NestedRootedConcat, capacity_control_width
from gcnet_missing_m3.osram import OSRAMBackbone


def test_slots_masks_and_gradients():
    torch.set_num_threads(1)
    block = NestedRootedConcat(256, 8, 64)
    local = torch.randn(3, 2, 256)
    base = torch.randn(3, 2, 1024)
    gap = torch.randn(3, 2, 3, 1024)
    av = torch.tensor([[[1, 0, 1], [1, 1, 1]]] * 3)
    umask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    out = block(local, base, gap, av, umask)
    assert out.shape == (3, 2, 320)
    assert not out[0].count_nonzero()  # No history, matching original Nested.
    assert not out[2, 1].count_nonzero()
    slots = out.reshape(3, 2, 5, 64)
    assert not slots[:, :, 2].count_nonzero()
    assert not slots[:, :, 4].count_nonzero()
    poisoned = gap.clone()
    poisoned[av.bool()] = float('nan')
    poisoned[2, 1] = float('nan')
    local[2, 1] = float('nan')
    base[2, 1] = float('nan')
    assert torch.equal(out, block(local, base, poisoned, av, umask))
    out.square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters())
    assert not any('decoder' in n for n, _ in block.named_parameters())


def test_initialization_and_capacity():
    kwargs = dict(latent_dim=256, output_dim=1600, num_heads=8, key_dim=64,
                  value_dim=64, bidirectional=False, dropout=0.)
    torch.manual_seed(66)
    flat = OSRAMBackbone(**kwargs)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(66)
    concat = OSRAMBackbone(**kwargs, osram_meaningful_block='nested_rooted_concat')
    assert torch.equal(torch.get_rng_state(), rng)
    for name, value in flat.state_dict().items():
        if not name.startswith('emotion_adapter.'):
            assert torch.equal(value, concat.state_dict()[name]), name
    raw = torch.randn(2, 4352)
    z = torch.randn(2, 320)
    assert torch.equal(flat.emotion_adapter(raw), concat.emotion_adapter(torch.cat((raw, z), -1)))
    assert all(p.requires_grad for p in concat.parameters())
    torch.manual_seed(66)
    control = OSRAMBackbone(**kwargs, osram_meaningful_block='nested_concat_capacity_control')
    assert torch.equal(torch.get_rng_state(), rng)
    counts = [sum(p.numel() for p in x.parameters()) for x in (flat, concat, control)]
    assert abs(counts[1] - counts[2]) <= (4352 + 1600 + 1) / 2
    assert control.emotion_adapter[1].out_features == capacity_control_width(256, 1024, 1600)


def test_original_predecoder_output_exact():
    from gcnet_missing_m3.meaningful_input_new40 import TokenAdapter
    from gcnet_missing_m3.meaningful_new40_structure import NestedGNN
    from gcnet_missing_m3.meaningful_blocks_common import active_groups, typed_means
    torch.manual_seed(90)
    old = TokenAdapter(NestedGNN())
    torch.manual_seed(90)
    new = NestedRootedConcat()
    for name, p in new.state_dict().items():
        assert torch.equal(p, old.state_dict()[name]), name
    local = torch.randn(2, 1, 256)
    base = torch.randn(2, 1, 1024)
    gap = torch.randn(2, 1, 3, 1024)
    av = torch.tensor([[[1., 0., 1.]]] * 2)
    expected_evidence = torch.cat((base[1, :, None, :512], gap[1, :, :, :512]), 1)
    active = torch.tensor([[True, False, True, False]])
    tokens, mask = old.tokenizer(local[1], expected_evidence, active)
    output = torch.zeros_like(tokens)
    for rows, columns, packed in active_groups(tokens, mask):
        output[rows[:, None], columns[None, :]] = old.core(packed, columns, 8)
    expected = typed_means(output, mask, 8).flatten(1)
    assert torch.equal(expected, new(local, base, gap, av, torch.ones(1, 2))[1])


def test_full_forward_initial_equivalence_and_learning():
    from gcnet_missing_m3.osram import MODALITIES
    torch.set_num_threads(1)
    kwargs = dict(latent_dim=256, output_dim=1600, num_heads=8, key_dim=64,
                  value_dim=64, bidirectional=False, dropout=0.)
    torch.manual_seed(66)
    flat = OSRAMBackbone(**kwargs).eval()
    torch.manual_seed(66)
    new = OSRAMBackbone(**kwargs, osram_meaningful_block='nested_rooted_concat').eval()
    node = torch.randn(3, 2, 256)
    latents = {name: torch.randn_like(node) for name in MODALITIES}
    av = torch.tensor([[[1., 0., 1.], [0., 1., 0.]]] * 3)
    mask = torch.tensor([[1., 1., 1.], [1., 1., 0.]])
    av[~mask.T.bool()] = 0
    args = (node, latents, av, torch.zeros(2, 3, dtype=torch.long), mask)
    writes = [[], []]
    out_flat, ctx_flat = flat(*args, post_write_observer=lambda t, m, a: writes[0].append(m.detach()))
    out_new, ctx_new = new(*args, post_write_observer=lambda t, m, a: writes[1].append(m.detach()))
    assert torch.equal(out_flat, out_new)
    assert all(torch.equal(a, b) for a, b in zip(writes[0], writes[1]))
    assert not out_new[2, 1].count_nonzero()
    # Training dropout has identical RNG consumption despite changed adapter
    # input dimension; only hidden width controls its sampled mask shape.
    for model in (flat, new):
        model.train()
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = .2
    torch.manual_seed(51)
    train_flat, _ = flat(*args)
    after_flat = torch.get_rng_state().clone()
    torch.manual_seed(51)
    train_new, _ = new(*args)
    assert torch.equal(train_flat, train_new)
    assert torch.equal(after_flat, torch.get_rng_state())
    old_core = new.meaningful_block.core.layers[0][0].weight.detach().clone()
    opt = torch.optim.Adam(new.parameters(), lr=1e-3)
    target = torch.randn_like(out_new)
    for _ in range(3):
        opt.zero_grad()
        out, _ = new(*args)
        (out - target).square().mean().backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in new.parameters())
        opt.step()
    assert not torch.equal(old_core, new.meaningful_block.core.layers[0][0].weight)
    assert torch.isfinite(out).all()
