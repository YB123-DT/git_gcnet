from types import SimpleNamespace

import torch

from gcnet_missing_m3.r02_delta_product import DeltaProductStorage


def storage(nh=2):
    torch.manual_seed(82)
    config = SimpleNamespace(osram_num_heads=2, osram_key_dim=3,
                             osram_value_dim=4, latent_dim=16,
                             osram_r02_num_householder=nh)
    return DeltaProductStorage(config)


def inputs():
    torch.manual_seed(81)
    keys = {m: torch.randn(7, 2, 2, 3, requires_grad=True)
            for m in ('audio', 'text', 'visual')}
    values = {m: torch.randn(7, 2, 2, 4, requires_grad=True) for m in keys}
    queries = torch.randn(7, 2, 4, 2, 3, requires_grad=True)
    observed = torch.ones(7, 2, 3, dtype=torch.bool)
    observed[::2, :, 0] = False
    observed[4] = False
    valid = torch.ones(7, 2, dtype=torch.bool)
    valid[2, 0] = False
    return keys, values, queries, observed, valid


def test_product_matches_ordered_formula_and_is_not_commutative():
    model = storage()
    x = torch.randn(2, 2, 7)
    keys, values, betas = model.controls(x)
    torch.testing.assert_close(keys.norm(dim=-1), torch.ones(2, 2, 2))
    assert bool(((betas > 0) & (betas < 2)).all())
    assert not torch.allclose(keys[0], keys[1])
    state = torch.randn(2, 2, 3, 4)
    actual = model.update(state, keys, values, betas)
    expected = state
    identity = torch.eye(3)
    for k, v, beta in zip(keys, values, betas):
        transition = identity - beta[..., None, None] * k[..., :, None] * k[..., None, :]
        expected = transition @ expected + beta[..., None, None] * k[..., :, None] * v[..., None, :]
    torch.testing.assert_close(actual, expected)
    reversed_state = model.update(state, keys.flip(0), values.flip(0), betas.flip(0))
    assert not torch.allclose(actual, reversed_state)
    one = storage(1)
    k, v, beta = one.controls(x)
    delta = state + beta[0, ..., None, None] * k[0, ..., :, None] * (
        v[0] - torch.einsum('bhkv,bhk->bhv', state, k[0]))[..., None, :]
    torch.testing.assert_close(one.update(state, k, v, beta), delta)


def test_write_masks_and_modality_permutation():
    model = storage()
    k, v = torch.randn(2, 2, 3, 3), torch.randn(2, 2, 4, 3)
    observed = torch.tensor([[True, False, True], [True, True, True]])
    active = torch.tensor([True, False])
    state = torch.randn(2, 2, 3, 4)
    q = torch.randn(2, 4, 2, 3)
    result = model.write(state, k, v, observed, active, q)
    permutation = torch.tensor([2, 0, 1])
    reordered = model.write(state, k[..., permutation], v[..., permutation],
                            observed[:, permutation], active, q)
    torch.testing.assert_close(result, reordered)
    torch.testing.assert_close(result[1], state[1], rtol=0, atol=0)
    torch.testing.assert_close(model.write(state, k, v, torch.zeros_like(observed),
                                          active, q), state, rtol=0, atol=0)
    k[0, ..., 1], v[0, ..., 1] = float('nan'), float('inf')
    k[1], v[1] = float('nan'), float('nan')
    torch.testing.assert_close(model.write(state, k, v, observed, active, q), result)


def test_scan_causality_padding_poisoned_missing_and_parameter_updates():
    model = storage()
    args = inputs()
    base, gap, _ = model.scan(*args)
    assert base.shape == (7, 2, 8) and gap.shape == (7, 2, 3, 8)
    assert torch.count_nonzero(base[0]) == 0
    assert torch.count_nonzero(gap[0]) == 0
    assert torch.count_nonzero(base[2, 0]) == 0
    dirty = tuple({m: t.detach().clone() for m, t in group.items()} for group in args[:2])
    for group in dirty:
        for i, tensor in enumerate(group.values()):
            tensor[~(args[3][:, :, i] & args[4])] = float('nan')
    changed = model.scan(*dirty, *args[2:])
    torch.testing.assert_close(changed[0], base)
    torch.testing.assert_close(changed[1], gap)
    for group in dirty:
        for tensor in group.values():
            tensor[3:] = torch.randn_like(tensor[3:]) * 50
    changed = model.scan(*dirty, *args[2:])
    torch.testing.assert_close(changed[0][:4], base[:4])
    torch.testing.assert_close(changed[1][:4], gap[:4])
    # Removing padding and an observation-free time step preserves later reads.
    selected = torch.tensor([True, True, False, True, False, True, True])
    compact = tuple({m: t[selected, :1] for m, t in group.items()} for group in args[:2])
    out = model.scan(*compact, args[2][selected, :1], args[3][selected, :1],
                     args[4][selected, :1])
    torch.testing.assert_close(out[0], base[selected, :1])
    base.square().mean().add(gap.square().mean()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               and p.grad.abs().sum() > 0 for p in model.parameters())
    for group in args[:2]:
        for i, tensor in enumerate(group.values()):
            assert torch.isfinite(tensor.grad).all()
            assert torch.count_nonzero(tensor.grad[~(args[3][:, :, i] & args[4])]) == 0
    before = [p.detach().clone() for p in model.parameters()]
    torch.optim.SGD(model.parameters(), lr=0.01).step()
    assert all(not torch.equal(old, p) for old, p in zip(before, model.parameters()))
