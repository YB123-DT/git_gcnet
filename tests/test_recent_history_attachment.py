"""The optional sidecar must leave the original causal backbone untouched."""
import copy
import importlib

import pytest
import torch


def attach():
    try:
        return importlib.import_module(
            'experiments.osram_mosi_recent_history_20261018.integration').attach_recent_history
    except ModuleNotFoundError as exc:
        pytest.fail('Recent-history Flat attachment is missing: ' + str(exc))


def setup_model():
    from tests.test_meaningful_block_integration import config
    from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model
    torch.set_num_threads(1)
    model = _build_model(config(), (3, 4, 5))
    with torch.no_grad():
        model.osram.emotion_adapter[-1].weight.normal_(std=.005)
    x = torch.randn(5, 2, 12)
    u = torch.tensor([[1.] * 5, [1., 1., 0., 0., 0.]])
    a = torch.tensor([[[1., 0., 1.], [0., 1., 0.]]] * 5)
    a[~u.T.bool()] = 0
    return model, ([x], a, torch.zeros(2, 5, dtype=torch.long), u, [5, 2])


def test_disabled_is_exact_noop_and_legacy_strict_load():
    model, args = setup_model()
    before = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()
    assert attach()(model, enabled=False) is None
    assert torch.equal(rng, torch.get_rng_state())
    model.load_state_dict(before, strict=True)
    assert set(model.state_dict()) == set(before)


def test_zero_init_preserves_original_state_predictions_rng_and_memory():
    model, args = setup_model()
    reference = copy.deepcopy(model)
    rng = torch.get_rng_state().clone()
    branch = attach()(model)
    assert torch.equal(rng, torch.get_rng_state())
    for name, value in reference.state_dict().items():
        assert torch.equal(value, model.state_dict()[name]), name
    for training in (False, True):
        reference.train(training); model.train(training)
        torch.manual_seed(42)
        expected = reference(*args, predict_missing=False)[0]
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(42)
        actual = model(*args, predict_missing=False)[0]
        assert torch.equal(expected, actual)
        assert torch.equal(expected_rng, torch.get_rng_state())
    assert branch is model.osram.recent_history
    assert model.osram._recent_history_inputs is None
    with pytest.raises(ValueError, match='already'):
        attach()(model)


def test_optimizer_updates_sidecar_without_replacing_flat_and_reloads_strictly():
    model, args = setup_model()
    attach()(model)
    original = model.osram.emotion_adapter
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    for _ in range(3):
        optimizer.zero_grad()
        prediction = model(*args, predict_missing=False)[0]
        prediction.square().mean().backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
    assert model.osram.emotion_adapter is original
    assert model.osram.recent_history.output.weight.abs().sum() > 0
    other, _ = setup_model()
    attach()(other)
    other.load_state_dict(model.state_dict(), strict=True)


def test_nonzero_branch_bypass_restores_flat_and_does_not_change_memory_reads():
    model, args = setup_model()
    model.eval()
    expected = model(*args, predict_missing=False)[0]
    memory_inputs = []
    handle = model.osram.emotion_adapter.register_forward_pre_hook(
        lambda module, inputs: memory_inputs.append(inputs[0].detach().clone()))
    branch = attach()(model)
    with torch.no_grad():
        branch.output.weight.normal_(std=.01)
        branch.output.bias.fill_(.1)
    full = model(*args, predict_missing=False)[0]
    integration = importlib.import_module('experiments.osram_mosi_recent_history_20261018.integration')
    integration.set_enabled(model, False)
    base = model(*args, predict_missing=False)[0]
    handle.remove()
    assert torch.equal(expected, base)
    assert not torch.equal(full, base)
    assert torch.equal(memory_inputs[0], memory_inputs[1])
    assert model.osram._recent_history_inputs is None
