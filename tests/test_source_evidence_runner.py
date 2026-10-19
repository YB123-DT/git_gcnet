"""Readout-only intervention contract, before the audit runner exists."""
import importlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn


def _api():
    name = 'experiments.osram_12_source_evidence_audit_20261019.readout'
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        pytest.fail('Missing readout-only source evidence implementation: ' + str(exc))


def _fixture():
    torch.manual_seed(12)
    local = torch.randn(4, 2, 8)
    sources = torch.randn(4, 2, 4, 3, 6)
    umask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0.]])
    availability = torch.tensor([[[1, 0, 1], [1, 1, 1]],
                                 [[0, 1, 0], [0, 0, 1]],
                                 [[1, 0, 0], [0, 0, 0]],
                                 [[0, 0, 0], [0, 0, 0]]])
    active = torch.cat((umask.T.bool()[..., None],
                        umask.T.bool()[..., None] & ~availability.bool()), -1)
    sources = torch.where(active[..., None, None], sources, 0.)
    reads = sources.sum(-2)
    contexts = torch.cat((reads, torch.zeros_like(reads)), -1)
    emotion_input = torch.cat((local, contexts.flatten(2)), -1)
    osram = SimpleNamespace(latent_dim=8, num_heads=2, value_dim=3,
        local_skip=nn.Linear(8, 10), emotion_adapter=nn.Sequential(
            nn.LayerNorm(56), nn.Linear(56, 10), nn.GELU(), nn.Dropout(.4), nn.Linear(10, 10)),
        emotion_norm=nn.LayerNorm(10))
    for layer in (osram.local_skip, osram.emotion_adapter, osram.emotion_norm):
        layer.eval().requires_grad_(False)
    head = nn.Linear(10, 4).eval().requires_grad_(False)
    return osram, head, local, emotion_input, sources, umask, availability


def test_readout_full_is_exact_and_no_rng_or_input_changes():
    api = _api()
    osram, head, local, inputs, sources, umask, av = _fixture()
    before = [v.clone() for v in (local, inputs, sources)]
    rng = torch.get_rng_state().clone()
    with torch.no_grad():
        full = head(osram.emotion_norm(osram.local_skip(local) + osram.emotion_adapter(inputs))
                    * umask.T[..., None])
        result = api.replay_readouts(osram, head, local, inputs, sources, umask, av, full)
    assert torch.equal(result['logits_coalitions'][..., 7, :], full)
    assert torch.equal(rng, torch.get_rng_state())
    for prior, value in zip(before, (local, inputs, sources)):
        assert torch.equal(prior, value)
    assert result['logits_removed'].shape == (4, 2, 4, 3, 4)
    assert result['logits_coalitions'].shape == (4, 2, 8, 4)


def test_empty_sources_and_inactive_gaps_have_no_decision_effect():
    api = _api()
    osram, head, local, inputs, sources, umask, av = _fixture()
    with torch.no_grad():
        full = head(osram.emotion_norm(osram.local_skip(local) + osram.emotion_adapter(inputs))
                    * umask.T[..., None])
        result = api.replay_readouts(osram, head, local, inputs, sources, umask, av, full)
        removed = result['logits_removed']
        for target in range(3):
            inactive = av[..., target].bool() & umask.T.bool()
            assert torch.equal(removed[..., target + 1, :, :][inactive],
                               full[inactive][:, None].expand(-1, 3, -1))
        empty = inputs.clone()
        empty[..., 8:] = 0
        expected = head(osram.emotion_norm(osram.local_skip(local) + osram.emotion_adapter(empty))
                        * umask.T[..., None])
        assert torch.equal(result['logits_coalitions'][..., 0, :], expected)


def test_reject_nonzero_backward_half_and_training_readout():
    api = _api()
    osram, head, local, inputs, sources, umask, av = _fixture()
    inputs[..., 8 + 6] = .5
    with pytest.raises(ValueError, match='backward|forward'):
        api.replay_readouts(osram, head, local, inputs, sources, umask, av, torch.zeros(4, 2, 4))
