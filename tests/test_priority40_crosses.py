import itertools

import pytest
import torch


METHODS = ['m21_nfm', 'm22_dcnv2', 'm23_cin', 'm24_afm', 'm25_kan']


@pytest.mark.parametrize('method', METHODS)
def test_shapes_all_masks_inactive_nan_gradients(method):
    from gcnet_missing_m3.priority40_crosses import build
    torch.manual_seed(42)
    model = build(method)
    mask = torch.tensor(list(itertools.product([False, True], repeat=5)))
    tokens = torch.randn(32, 5, 64, requires_grad=True)
    poisoned = torch.where(mask[..., None], tokens, torch.full_like(tokens, float('nan')))
    expected = model(tokens, mask)
    actual = model(poisoned, mask)
    assert actual.shape == (32, 64)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)
    assert torch.count_nonzero(actual[0]) == 0
    actual.square().mean().backward()
    assert torch.count_nonzero(tokens.grad[~mask]) == 0
    assert tokens.grad[mask].abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_nfm_exact_pair_pooling_before_mlp():
    from gcnet_missing_m3.priority40_crosses import build
    model = build('m21_nfm', dim=4)
    model.mlp = torch.nn.Identity()
    x = torch.randn(2, 5, 4)
    expected = sum(x[:, i] * x[:, j] for i in range(5) for j in range(i + 1, 5))
    torch.testing.assert_close(model(x, torch.ones(2, 5, dtype=torch.bool)), expected)


def test_dcn_dense_exact_two_cross_layers():
    from gcnet_missing_m3.priority40_crosses import build
    model = build('m22_dcnv2', dim=4)
    assert len(model.cross) == 2
    assert all(layer.weight.shape == (20, 20) for layer in model.cross)
    x = torch.randn(2, 5, 4)
    x0 = x.flatten(1)
    hidden = x0
    for layer in model.cross:
        hidden = x0 * (hidden @ layer.weight.T + layer.bias) + hidden
    torch.testing.assert_close(model(x, torch.ones(2, 5, dtype=torch.bool)), model.out(hidden))


def test_cin_reuses_matching_fixed_architecture():
    from gcnet_missing_m3.priority40_crosses import build
    from gcnet_missing_m3.readout_candidates_interactions import CIN
    model = build('m23_cin')
    assert isinstance(model, CIN)
    assert [tuple(w.shape) for w in model.weights] == [(16, 5, 5), (16, 16, 5)]


def test_afm_attention_is_over_valid_pairs():
    from gcnet_missing_m3.priority40_crosses import build
    model = build('m24_afm', dim=4)
    model.out = torch.nn.Identity()
    for parameter in model.attention.parameters():
        torch.nn.init.zeros_(parameter)
    x = torch.randn(2, 5, 4)
    mask = torch.tensor([[True, False, True, True, False], [True, False, False, False, False]])
    result = model(x, mask)
    expected = (x[0, 0] * x[0, 2] + x[0, 0] * x[0, 3] + x[0, 2] * x[0, 3]) / 3
    torch.testing.assert_close(result[0], expected)
    assert torch.count_nonzero(result[1]) == 0


def test_kan_cubic_grid_and_edge_coefficients():
    from gcnet_missing_m3.priority40_crosses import build
    model = build('m25_kan')
    assert model.first.coefficients.shape == (32, 64, 8)
    assert model.second.coefficients.shape == (64, 32, 8)
    x = torch.linspace(-.99, .99, 64)[None]
    basis = model.first.basis(x)
    assert basis.shape == (1, 64, 8)
    assert (basis >= 0).all()
    torch.testing.assert_close(basis.sum(-1), torch.ones_like(x))
