"""Contract tests for the strictly causal observed-latent side branch."""

import importlib
import importlib.util

import pytest
import torch


MODULE = "gcnet_missing_m3.recent_history"


def _class():
    assert importlib.util.find_spec(MODULE) is not None, "recent-history branch is not implemented"
    return importlib.import_module(MODULE).RecentHistoryAttentionResidual


def _inputs():
    generator = torch.Generator().manual_seed(212)
    local = torch.randn(7, 2, 6, generator=generator)
    base = torch.randn(7, 2, 16, generator=generator)
    latents = {key: torch.randn(7, 2, 6, generator=generator)
               for key in ("audio", "text", "visual")}
    availability = torch.ones(7, 2, 3)
    umask = torch.ones(7, 2)
    return local, base, latents, availability, umask


def _model(window=2, random_output=True):
    torch.manual_seed(311)
    model = _class()(local_dim=6, latent_dim=6, context_dim=8,
                     output_dim=10, window=window, attention_dim=8, num_heads=2)
    if random_output:
        torch.nn.init.normal_(model.output.weight)
        torch.nn.init.normal_(model.output.bias)
    return model.eval()


def test_zero_initialized_output_is_exact_zero():
    model = _model(random_output=False)
    output = model(*_inputs())
    assert output.shape == (7, 2, 10)
    assert torch.equal(output, torch.zeros_like(output))


def test_current_future_and_outside_window_latents_cannot_affect_target():
    model = _model()
    original = _inputs()
    expected = model(*original)[4].clone()
    local, base, latents, availability, umask = _inputs()
    for value in latents.values():
        value[:2] += 100  # t=4 reads only t=2,3.
        value[4:] -= 100
    actual = model(local, base, latents, availability, umask)[4]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_recent_observed_token_can_change_readout():
    model = _model()
    original = _inputs()
    expected = model(*original)[4].clone()
    local, base, latents, availability, umask = _inputs()
    latents["text"][3] += 100
    assert not torch.equal(model(local, base, latents, availability, umask)[4], expected)


def test_conversations_do_not_share_tokens():
    model = _model()
    original = _inputs()
    expected = model(*original)[:, 0].clone()
    local, base, latents, availability, umask = _inputs()
    for value in latents.values():
        value[:, 1] += 1000
    torch.testing.assert_close(model(local, base, latents, availability, umask)[:, 0],
                               expected, rtol=0, atol=0)


def test_unobserved_nan_tokens_do_not_leak_and_gradients_are_finite():
    model = _model()
    local, base, latents, availability, umask = _inputs()
    availability[:, :, 1] = 0
    reference = model(local, base, latents, availability, umask)
    latents["text"][:] = float("nan")
    for value in latents.values():
        value.requires_grad_(True)
    output = model(local, base, latents, availability, umask)
    torch.testing.assert_close(output, reference, rtol=0, atol=0)
    output.square().mean().backward()
    assert all(torch.isfinite(value.grad).all() for value in latents.values())
    assert torch.count_nonzero(latents["text"].grad) == 0


def test_padding_and_no_history_are_zero_even_with_trained_output_bias():
    model = _model()
    local, base, latents, availability, umask = _inputs()
    umask[4:, 1] = 0
    local[4:, 1] = float("nan")
    base[4:, 1] = float("nan")
    for value in latents.values():
        value[4:, 1] = float("nan")
    output = model(local, base, latents, availability, umask)
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output[0]) == 0
    assert torch.count_nonzero(output[4:, 1]) == 0
    availability.zero_()
    assert torch.count_nonzero(model(local, base, latents, availability, umask)) == 0


def test_padding_does_not_create_history_for_a_later_valid_position():
    model = _model(window=1)
    local, base, latents, availability, umask = _inputs()
    umask[2] = 0
    output = model(local, base, latents, availability, umask)
    assert torch.count_nonzero(output[3]) == 0


def test_query_uses_current_local_and_forward_base_but_not_zero_backward_half():
    model = _model()
    original = _inputs()
    expected = model(*original).clone()
    local, base, latents, availability, umask = _inputs()
    base[..., 8:] = float("nan")
    torch.testing.assert_close(model(local, base, latents, availability, umask),
                               expected, rtol=0, atol=0)
    base[4, :, :8] += 10
    assert not torch.equal(model(local, base, latents, availability, umask)[4], expected[4])
    local, base, latents, availability, umask = _inputs()
    local[4] += 10
    assert not torch.equal(model(local, base, latents, availability, umask)[4], expected[4])


def test_zero_init_then_training_yields_finite_connected_gradients():
    model = _model(random_output=False).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    inputs = _inputs()
    target = torch.randn(7, 2, 10)
    for step in range(3):
        optimizer.zero_grad()
        loss = (model(*inputs) - target).square().mean()
        loss.backward()
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                   for parameter in model.parameters())
        if step == 0:
            assert model.query.weight.grad.abs().sum() == 0
        else:
            assert model.query.weight.grad.abs().sum() > 0
            assert model.key.weight.grad.abs().sum() > 0
            assert model.value.weight.grad.abs().sum() > 0
        optimizer.step()
    assert all(not value.requires_grad for value in model.last_diagnostics.values())


def test_default_branch_forward_does_not_consume_random_numbers():
    model = _model().train()
    inputs = _inputs()
    before = torch.get_rng_state().clone()
    model(*inputs)
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize("kwargs", [{"window": 0}, {"attention_dim": 7, "num_heads": 2}])
def test_invalid_configuration_fails(kwargs):
    with pytest.raises(ValueError):
        _class()(**kwargs)
