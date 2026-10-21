"""Tests use tiny real Torch modules; no dataset or CHM simulation required."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from experiments.osram_mosei_direct_role_readout_20261021.readout import replay_roles
from experiments.osram_mosei_direct_role_readout_20261021.analysis import analyze


def fixture():
    torch.manual_seed(66)
    osram = SimpleNamespace(training=False, num_heads=2, value_dim=2,
        local_skip=nn.Linear(3, 6).eval(),
        emotion_adapter=nn.Sequential(nn.Linear(35, 6), nn.GELU()).eval(),
        emotion_norm=nn.LayerNorm(6).eval())
    head = nn.Linear(6, 1).eval()
    local = torch.randn(3, 2, 3)
    availability = torch.tensor([[[1, 0, 1], [0, 1, 1]]] * 3)
    umask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    contexts = torch.randn(3, 2, 4, 8)
    contexts[..., 4:] = 0
    active = torch.cat((torch.ones(3, 2, 1, dtype=torch.bool), ~availability.bool()), -1)
    active &= umask.T[..., None].bool()
    contexts = torch.where(active[..., None], contexts, torch.zeros_like(contexts))
    x = torch.cat((local, contexts.flatten(-2)), -1)
    full = head(osram.emotion_norm(osram.local_skip(local) + osram.emotion_adapter(x)) * umask.T[..., None])
    return osram, head, local, x, availability, umask, full


def test_roles_preserve_local_skip_contexts_rng_and_full():
    osram, head, local, x, av, mask, full = fixture()
    old = x.clone(); rng = torch.get_rng_state().clone()
    result = replay_roles(osram, head, local, x, av, mask, full)
    assert torch.equal(result['Full'], full)
    assert torch.equal(x, old) and torch.equal(rng, torch.get_rng_state())
    for name, keep in [('L', (0, 0, 0, 0)), ('LB', (1, 0, 0, 0)), ('LG', (0, 1, 1, 1))]:
        value = x.clone(); ctx = value[..., 3:].reshape(3, 2, 4, 8)
        ctx.copy_(torch.where(torch.tensor(keep).bool()[None, None, :, None], ctx, torch.zeros_like(ctx)))
        expected = head(osram.emotion_norm(osram.local_skip(local) + osram.emotion_adapter(value)) * mask.T[..., None])
        assert torch.equal(result[name], expected)
        assert not result[name].requires_grad


def test_reject_observed_gap_and_changed_local():
    osram, head, local, x, av, mask, full = fixture()
    changed = x.clone(); changed[0, 0, 3 + 8] = 1
    with pytest.raises(ValueError, match='Inactive'):
        replay_roles(osram, head, local, changed, av, mask, full)
    with pytest.raises(AssertionError):
        replay_roles(osram, head, local + 1, x, av, mask, full)


def test_metric_corrections_filter_zero_labels():
    arrays = dict(labels=np.array([1., -1., 0., -1.]),
        prediction_L=np.array([-1., -1., 8., 1.]),
        prediction_LB=np.array([1., -1., -8., 1.]),
        prediction_LG=np.array([-1., 1., 8., -1.]),
        prediction_Full=np.array([1., -1., -8., -1.]))
    result = analyze(arrays)
    assert result['models']['Full']['ACC'] == 100
    assert result['comparisons']['L_to_LB']['corrections'] == 1
    assert result['comparisons']['L_to_LB']['harms'] == 0
    assert result['comparisons']['L_to_LG']['corrections'] == 1
    assert result['comparisons']['L_to_LG']['harms'] == 1
    assert result['groups']['all']['learned'] == result['models']['Full']
