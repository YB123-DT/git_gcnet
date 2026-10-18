"""Single-trajectory, exact-mask frozen B1/B2 intervention contracts."""
import copy
import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn


def setup(monkeypatch, kind='B2'):
    evaluator = importlib.import_module(
        'experiments.osram_meld_related_history_diagnostic_20261018.evaluate')
    from tests.test_recent_history_attachment import setup_model
    from experiments.osram_meld_related_history_20261018.integration import attach_related_history
    model, _ = setup_model()
    model.smax_fc = nn.Linear(model.osram.output_dim, 7)
    branch = attach_related_history(model, kind=kind)
    with torch.no_grad():
        branch.output.weight.normal_(std=.02)
        branch.output.bias.fill_(.1)
    model.eval().requires_grad_(False)
    raw = [torch.randn(4, 2, width) for width in (3, 4, 5, 3, 4, 5)]
    raw += [torch.zeros(2, 4, dtype=torch.long),
            torch.tensor([[1., 1., 1., 1.], [1., 1., 0., 0.]]),
            torch.tensor([[0, 1, 2, 1], [3, 4, 0, 0]]), ['a', 'b']]
    availability = torch.tensor([[[1., 0, 0], [0, 1, 1]],
                                 [[1, 1, 0], [0, 0, 1]],
                                 [[1, 1, 1], [0, 0, 0]],
                                 [[0, 1, 0], [0, 0, 0]]])

    def view(data, availability, dimensions):
        host = torch.cat(data[:3], -1)
        mask = torch.cat([availability[..., i:i+1].expand(-1, -1, width)
                          for i, width in enumerate(dimensions)], -1)
        return dict(incomplete=host * mask, availability=availability,
                    qmask=data[6], umask=data[7], lengths=[4, 2])

    monkeypatch.setattr(evaluator, 'replay_view_from_availability', view)
    prepared = view(raw, availability, (3, 4, 5))
    logits = model([prepared['incomplete']], availability, raw[6], raw[7], [4, 2],
                   predict_missing=False)[0]
    valid = raw[7].bool().numpy()
    saved = dict(logits=logits.transpose(0, 1).numpy()[valid],
                 labels=raw[8].numpy()[valid], speakers=raw[6].numpy()[valid],
                 availability=availability.transpose(0, 1).numpy()[valid],
                 conversation_ids=np.broadcast_to(np.array(raw[-1])[:, None], valid.shape)[valid],
                 utterance_indices=np.broadcast_to(np.arange(4), valid.shape)[valid])
    saved['predictions'] = saved['logits'].argmax(1)
    cfg = SimpleNamespace(dataset='MELD', latent_dim=256)
    trainer = SimpleNamespace(_move_batch=lambda data, device: data)
    return evaluator, model, [raw], cfg, (3, 4, 5), saved, trainer


@pytest.mark.parametrize('kind', ['B1', 'B2'])
def test_one_scan_saved_parity_weights_rng_and_relation_alignment(monkeypatch, kind):
    ev, model, loader, cfg, dims, saved, trainer = setup(monkeypatch, kind)
    before = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()
    scans = []
    hook = model.osram.register_forward_pre_hook(lambda module, args: scans.append(1))
    result, artifacts, pairs = ev.evaluate_rate(model, loader, cfg, dims, saved, trainer)
    hook.remove()
    assert len(scans) == result['checks']['memory_scan_count'] == 1
    assert result['checks']['saved_predictions_exact']
    assert model.osram.related_history_enabled
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())
    np.testing.assert_allclose(artifacts['logits_on'], saved['logits'], rtol=2e-5, atol=2e-5)
    assert set(result['metrics']) == ({'on', 'off', 'half'} if kind == 'B2' else {'on', 'off'})
    # Both conversation-first positions have no history, even with a nonzero output bias.
    np.testing.assert_array_equal(artifacts['residual_norm'][[0, 4]], 0.)
    np.testing.assert_array_equal(artifacts['logits_on'][[0, 4]], artifacts['logits_off'][[0, 4]])
    for name in result['metrics']:
        assert result['metrics'][name]['N'] == 6
        assert len(result['metrics'][name]['recall']) == 7
    if kind == 'B2':
        assert len(pairs['labels']) == 7  # 4*(4-1)/2 + 2*(2-1)/2
        assert np.all(pairs['past_indices'] < pairs['current_indices'])
        assert np.all(pairs['lag'] == pairs['current_indices'] - pairs['past_indices'])
        assert np.all((pairs['probabilities'] >= 0) & (pairs['probabilities'] <= 1))
        labels = dict(a=[0, 1, 2, 1], b=[3, 4])
        expected = [labels[c][t] == labels[c][j] for c, t, j in zip(
            pairs['conversation_ids'], pairs['current_indices'], pairs['past_indices'])]
        np.testing.assert_array_equal(pairs['labels'], expected)
        assert model.osram.related_history._cache is None
    else:
        assert pairs == {}


def test_half_replay_changes_only_d_and_rho_and_hard_masks_empty_history(monkeypatch):
    ev, model, loader, cfg, dims, saved, trainer = setup(monkeypatch)
    branch = model.osram.related_history
    captured = []
    handle = branch.fusion[0].register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().clone()))
    _, artifacts, _ = ev.evaluate_rate(model, loader, cfg, dims, saved, trainer)
    handle.remove()
    assert len(captured) == 2
    torch.testing.assert_close(captured[0][..., :-129], captured[1][..., :-129], rtol=0, atol=0)
    assert torch.count_nonzero(captured[1][..., -129:-1]) == 0
    assert captured[1][1, 0, -1] == .5
    assert captured[1][0, 0, -1] == 0
    np.testing.assert_array_equal(artifacts['half_residual_norm'][[0, 4]], 0.)


@pytest.mark.parametrize('field', ['labels', 'conversation_ids', 'utterance_indices', 'speakers'])
def test_metadata_mismatch_fails_before_scan_and_restores_hooks(monkeypatch, field):
    ev, model, loader, cfg, dims, saved, trainer = setup(monkeypatch)
    saved[field] = saved[field].copy()
    saved[field][0] = 'wrong' if field == 'conversation_ids' else 9
    handles = len(model.osram.emotion_norm._forward_pre_hooks)
    with pytest.raises((AssertionError, ValueError)):
        ev.evaluate_rate(model, loader, cfg, dims, saved, trainer)
    assert handles == len(model.osram.emotion_norm._forward_pre_hooks)
    assert model.osram.related_history_enabled


def test_same_epoch_replay_does_not_require_other_models_saved_logits(monkeypatch):
    ev, model, loader, cfg, dims, saved, trainer = setup(monkeypatch)
    saved['logits'] = saved['logits'] + np.arange(7)[None]
    saved['predictions'] = saved['logits'].argmax(1)
    result, _, _ = ev.evaluate_rate(model, loader, cfg, dims, saved, trainer, verify_saved=False)
    assert result['checks']['saved_prediction_verification_requested'] is False
    assert result['checks']['on_replay_class_exact']


def test_invalid_mask_and_nonfrozen_model_rejected(monkeypatch):
    ev, model, loader, cfg, dims, saved, trainer = setup(monkeypatch)
    model.train()
    with pytest.raises(ValueError, match='frozen'):
        ev.evaluate_rate(model, loader, cfg, dims, saved, trainer)
    model.eval()
    saved['availability'][0] = 0
    with pytest.raises(ValueError):
        ev.evaluate_rate(model, loader, cfg, dims, saved, trainer)


def test_padding_and_inactive_gap_are_zero_in_all_readouts(monkeypatch):
    ev, model, loader, cfg, dims, saved, trainer = setup(monkeypatch)
    fusion_inputs, head_inputs = [], []
    fusion_hook = model.osram.related_history.fusion[0].register_forward_pre_hook(
        lambda module, args: fusion_inputs.append(args[0].detach().clone()))
    head_hook = model.smax_fc.register_forward_pre_hook(
        lambda module, args: head_inputs.append(args[0].detach().clone()))
    ev.evaluate_rate(model, loader, cfg, dims, saved, trainer)
    fusion_hook.remove(); head_hook.remove()
    valid = loader[0][7].T.bool()
    for hidden in head_inputs:
        assert torch.count_nonzero(hidden[~valid]) == 0
    availability = torch.zeros((*valid.shape, 3))
    availability.transpose(0, 1)[loader[0][7].bool()] = torch.as_tensor(saved['availability'])
    width = model.osram.context_dim // 2
    for inputs in fusion_inputs:
        gaps = inputs[..., 256 + width:256 + 4 * width].reshape(*valid.shape, 3, width)
        assert torch.count_nonzero(gaps[availability.bool()]) == 0
        assert torch.count_nonzero(gaps[~valid]) == 0


def test_forward_failure_removes_temporary_hooks_and_preserves_enabled_state(monkeypatch):
    ev, model, loader, cfg, dims, saved, trainer = setup(monkeypatch)
    hooks_before = tuple(model.osram.emotion_norm._forward_pre_hooks)
    branch_hooks_before = tuple(model.osram.related_history._forward_hooks)
    saved['logits'][0, 0] += 100
    saved['predictions'] = saved['logits'].argmax(1)
    with pytest.raises(AssertionError):
        ev.evaluate_rate(model, loader, cfg, dims, saved, trainer)
    assert tuple(model.osram.emotion_norm._forward_pre_hooks) == hooks_before
    assert tuple(model.osram.related_history._forward_hooks) == branch_hooks_before
    assert model.osram.related_history_enabled
