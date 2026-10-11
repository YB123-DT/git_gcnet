"""Readout-only evidence interventions over the original Flat adapter."""

import importlib

import pytest
import torch
from torch import nn


def adapter(input_dim=11, output_dim=5, dropout=0.0):
    return nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, 7),
                         nn.GELU(), nn.Dropout(dropout), nn.Linear(7, output_dim))


def inputs():
    torch.manual_seed(19)
    local = torch.randn(3, 1, 3)
    base = torch.randn(3, 1, 2)
    gap = torch.randn(3, 1, 3, 2)
    availability = torch.tensor([[[1, 1, 1]], [[1, 0, 1]], [[0, 0, 1]]])
    umask = torch.ones(1, 3)
    return local, base, gap, availability, umask


def module():
    return importlib.import_module('gcnet_missing_m3.counterfactual_readout')


def test_equal_views_share_training_dropout_and_one_call_rng_advance():
    flat = adapter(dropout=.5).train()
    values = torch.randn(3, 2, 11)
    views = values.unsqueeze(2).expand(-1, -1, 5, -1).clone()
    state = torch.get_rng_state()
    expected = flat(values)
    expected_rng = torch.get_rng_state()
    torch.set_rng_state(state)
    actual = module().evaluate_flat_views(flat, views)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    torch.testing.assert_close(actual[..., 0, :], expected)
    for index in range(1, 5):
        torch.testing.assert_close(actual[..., index, :], actual[..., 0, :])


@pytest.mark.parametrize('mode,view_count,token_count',
                         [('marginal', 5, 4), ('coalition', 12, 10)])
def test_zero_init_preserves_flat_anchor_and_masks_inactive_slots(mode, view_count, token_count):
    local, base, gap, availability, umask = inputs()
    skip, flat = nn.Linear(3, 5), adapter()
    block = module().CounterfactualEvidenceReadout(mode, 2, 5, encoder_dropout=0).eval()
    anchor, residual = block(local, base, gap, availability, umask,
                             local_skip=skip, adapter=flat)
    valid = umask.T.bool()
    history = valid & ((valid.long().cumsum(0) - valid.long()) > 0)
    safe_base = torch.where(history[..., None], base, 0.)
    active_gap = history[..., None] & ~availability.bool()
    safe_gap = torch.where(active_gap[..., None], gap, 0.)
    full_input = torch.cat((local, safe_base, safe_gap.flatten(2)), -1)
    torch.testing.assert_close(anchor, skip(local) + flat(full_input))
    assert torch.count_nonzero(residual) == 0
    assert block.last_view_count == view_count
    effects, active = block.last_effects, block.last_active
    assert effects.shape == (3, 1, token_count, 5)
    assert torch.count_nonzero(effects[~active]) == 0
    assert torch.count_nonzero(active[0]) == 0  # no history on the first utterance
    assert torch.count_nonzero(residual[~umask.T.bool()]) == 0


def test_marginals_equal_full_minus_designated_omit_view():
    local, base, gap, availability, umask = inputs()
    block = module().CounterfactualEvidenceReadout('marginal', 2, 5, encoder_dropout=0).eval()
    block(local, base, gap, availability, umask,
          local_skip=nn.Linear(3, 5), adapter=adapter())
    views = block.last_view_outputs
    expected = views[..., :1, :] - views[..., 1:, :]
    expected = torch.where(block.last_active[..., None], expected, 0.)
    torch.testing.assert_close(block.last_effects, expected)


def test_coalition_singleton_and_pair_effects_match_mobius_formula():
    local, base, gap, availability, umask = inputs()
    block = module().CounterfactualEvidenceReadout('coalition', 2, 5, encoder_dropout=0).eval()
    block(local, base, gap, availability, umask,
          local_skip=nn.Linear(3, 5), adapter=adapter())
    views = block.last_view_outputs
    empty, singles, pairs = views[..., 0, :], views[..., 1:5, :], views[..., 5:11, :]
    expected_singles = singles - empty.unsqueeze(-2)
    pair_indices = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    expected_pairs = torch.stack([
        pairs[..., n, :] - singles[..., i, :] - singles[..., j, :] + empty
        for n, (i, j) in enumerate(pair_indices)
    ], -2)
    expected = torch.cat((expected_singles, expected_pairs), -2)
    expected = torch.where(block.last_active[..., None], expected, 0.)
    torch.testing.assert_close(block.last_effects, expected)


@pytest.mark.parametrize('mode', ['marginal', 'coalition'])
def test_nan_in_inactive_evidence_cannot_leak_and_gradients_are_finite(mode):
    local, base, gap, availability, umask = inputs()
    clean = (local.clone(), base.clone(), gap.clone(), availability, umask)
    dirty = (local.clone(), base.clone(), gap.clone(), availability, umask)
    valid = umask.T.bool()
    history = valid & ((valid.long().cumsum(0) - valid.long()) > 0)
    active_gap = history[..., None] & ~availability.bool()
    dirty[1][~history] = float('nan')
    dirty[2][~active_gap] = float('nan')
    skip, flat = nn.Linear(3, 5), adapter()
    block = module().CounterfactualEvidenceReadout(mode, 2, 5, encoder_dropout=0)
    nn.init.normal_(block.residual.weight)
    nn.init.normal_(block.residual.bias)
    expected = block(*clean, local_skip=skip, adapter=flat)
    actual = block(*dirty, local_skip=skip, adapter=flat)
    torch.testing.assert_close(actual, expected)
    loss = (actual[0] + actual[1]).square().sum()
    loss.backward()
    for parameter in list(block.parameters()) + list(flat.parameters()) + list(skip.parameters()):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_invalid_mode_is_rejected():
    with pytest.raises(ValueError, match='counterfactual readout'):
        module().CounterfactualEvidenceReadout('unknown', 2, 5)
