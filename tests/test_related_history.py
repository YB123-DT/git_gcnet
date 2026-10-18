"""B1/B2 contracts: strictly prior Local history and isolated relation supervision."""

import importlib
import importlib.util

import pytest
import torch


def _class():
    name = "gcnet_missing_m3.related_history"
    assert importlib.util.find_spec(name) is not None, "B1/B2 module is missing"
    return importlib.import_module(name).RelatedHistoryResidual


def _inputs():
    generator = torch.Generator().manual_seed(133)
    local = torch.randn(6, 2, 6, generator=generator)
    base = torch.randn(6, 2, 8, generator=generator)
    gap = torch.randn(6, 2, 3, 8, generator=generator)
    availability = torch.tensor([1., 0., 1.]).expand(6, 2, 3).clone()
    umask = torch.ones(6, 2)
    speakers = torch.arange(6).remainder(2)[:, None].expand(6, 2)
    return local, base, gap, availability, umask, speakers


def _model(kind="B2", random_output=True, window=2):
    torch.manual_seed(49)
    model = _class()(kind=kind, local_dim=6, context_dim=8, output_dim=10,
                     window=window, attention_dim=8)
    if random_output:
        torch.nn.init.normal_(model.output.weight)
        torch.nn.init.normal_(model.output.bias)
    return model


@pytest.mark.parametrize("kind,width", [("B1", 2435), ("B2", 2564)])
def test_default_dimensions_keep_all_four_forward_memory_slots(kind, width):
    model = _class()(kind=kind)
    assert model.fusion[0].in_features == width
    assert model.output.in_features == 128
    assert model.output.out_features == 1600
    assert model.window == 8
    if kind == "B2":
        assert model.relation[0].in_features == 520


@pytest.mark.parametrize("kind", ["B1", "B2"])
def test_zero_init_and_hard_nohistory_padding(kind):
    inputs = _inputs()
    model = _model(kind, random_output=False)
    result = model(*inputs)
    assert result.shape == (6, 2, 10)
    assert torch.count_nonzero(result) == 0
    torch.nn.init.normal_(model.output.weight)
    torch.nn.init.ones_(model.output.bias)
    inputs[4][3:, 1] = 0
    inputs[0][3:, 1] = float("nan")
    inputs[1][3:, 1] = float("nan")
    inputs[2][3:, 1] = float("nan")
    result = model(*inputs)
    assert torch.isfinite(result).all()
    assert torch.count_nonzero(result[0]) == 0
    assert torch.count_nonzero(result[3:, 1]) == 0


@pytest.mark.parametrize("kind", ["B1", "B2"])
def test_causal_window_and_batch_isolation(kind):
    model = _model(kind).eval()
    original = _inputs()
    expected = model(*original)[4, 0].detach().clone()
    changed = _inputs()
    changed[0][:2, 0] += 500  # t=4 may see only t=2,3.
    changed[0][5:, 0] -= 500
    for tensor in changed[:3]:
        tensor[:, 1] += 500
    torch.testing.assert_close(model(*changed)[4, 0], expected, rtol=0, atol=0)
    changed = _inputs()
    changed[0][3, 0] += 500
    assert not torch.equal(model(*changed)[4, 0], expected)


def test_attention_is_strictly_past_normalized_and_no_padding_history():
    model = _model()
    inputs = _inputs()
    inputs[4][2:4, 1] = 0
    result = model(*inputs)
    weights = model.last_attention
    assert weights.shape == (6, 2, 2)
    assert torch.count_nonzero(weights[0]) == 0
    torch.testing.assert_close(weights[1:, 0].sum(-1), torch.ones(5))
    assert torch.count_nonzero(result[4, 1]) == 0


def test_inactive_gap_nan_safe_and_gradients_zero():
    model = _model()
    inputs = _inputs()
    reference = model(*inputs)
    inputs[2][:, :, [0, 2]] = float("nan")
    for tensor in inputs[:3]:
        tensor.requires_grad_(True)
    actual = model(*inputs)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    actual.square().mean().backward()
    assert all(torch.isfinite(tensor.grad).all() for tensor in inputs[:3])
    assert torch.count_nonzero(inputs[2].grad[:, :, [0, 2]]) == 0
    assert inputs[2].grad[:, :, 1].abs().sum() > 0


def test_auxiliary_supervision_updates_only_relation_predictor():
    model = _model()
    inputs = _inputs()
    inputs[0].requires_grad_(True)
    inputs[1].requires_grad_(True)
    inputs[2].requires_grad_(True)
    model(*inputs)
    labels = torch.arange(12).reshape(6, 2).remainder(3)
    loss = model.auxiliary_relation_loss(labels, dataset="MELD")
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert all(tensor.grad is None for tensor in inputs[:3])
    for name, parameter in model.named_parameters():
        if name.startswith("relation."):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        else:
            assert parameter.grad is None, name
    assert model.relation[-1].weight.grad.abs().sum() > 0


def test_main_task_relation_path_remains_differentiable():
    model = _model()
    inputs = _inputs()
    inputs[0].requires_grad_(True)
    model(*inputs).square().mean().backward()
    assert model.relation[-1].weight.grad.abs().sum() > 0
    assert model.query.weight.grad.abs().sum() > 0
    assert model.key.weight.grad.abs().sum() > 0
    assert model.value.weight.grad.abs().sum() > 0
    assert inputs[0].grad.abs().sum() > 0


def test_first_zero_init_main_gradient_is_not_mistaken_for_upstream_learning():
    model = _model(random_output=False)
    inputs = _inputs()
    model(*inputs).sum().backward()
    assert model.output.weight.grad.abs().sum() > 0
    assert model.query.weight.grad.abs().sum() == 0
    assert model.relation[-1].weight.grad.abs().sum() == 0


def test_relation_loss_is_conversation_mean_not_pair_count_mean():
    model = _model()
    inputs = _inputs()
    inputs[4][3:, 1] = 0
    labels = torch.arange(12).reshape(6, 2).remainder(3)
    model(*inputs)
    full = model.auxiliary_relation_loss(labels, "MELD")
    separate = []
    for batch in range(2):
        subset = tuple(t[:, batch:batch+1] for t in inputs)
        model(*subset)
        separate.append(model.auxiliary_relation_loss(labels[:, batch:batch+1], "MELD"))
    torch.testing.assert_close(full, torch.stack(separate).mean())


def test_neutral_regression_pairs_have_zero_finite_aux_loss():
    model = _model()
    model(*_inputs())
    loss = model.auxiliary_relation_loss(torch.zeros(6, 2), "MOSI")
    assert loss.item() == 0 and torch.isfinite(loss)
    loss.backward()
    assert model.relation[-1].weight.grad is not None
    assert torch.count_nonzero(model.relation[-1].weight.grad) == 0
    model.clear_cache()
    with pytest.raises(RuntimeError, match="forward"):
        model.auxiliary_relation_loss(torch.zeros(6, 2), "MOSI")


def test_speaker_indices_and_onehot_have_identical_behavior():
    model = _model()
    inputs = _inputs()
    expected = model(*inputs)
    onehot = torch.nn.functional.one_hot(inputs[-1], num_classes=2).float()
    torch.testing.assert_close(model(*inputs[:-1], onehot), expected, rtol=0, atol=0)


def test_no_relation_targets_are_needed_during_inference_and_rng_is_unchanged():
    model = _model().eval()
    state = torch.get_rng_state().clone()
    with torch.no_grad():
        result = model(*_inputs())
    assert torch.isfinite(result).all()
    assert torch.equal(state, torch.get_rng_state())
    assert all(not value.requires_grad for value in model.last_diagnostics.values())
