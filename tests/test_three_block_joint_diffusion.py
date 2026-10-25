import torch

from experiments.osram_three_block_joint_mosi_20261025.diffusion import DifferentiableLatentCSDI
from osram_four_block_wiring.csdi_latent import LatentCSDI


def test_checkpoint_matches_outputs_and_all_gradients():
    torch.manual_seed(7)
    model = DifferentiableLatentCSDI(steps=3, channels=8, layers=1)
    observed = torch.randn(2, 2, 3, 64, requires_grad=True)
    context = torch.randn(2, 2, 2304, requires_grad=True)
    a = torch.tensor([[[1, 0, 1], [1, 1, 1]], [[0, 1, 0], [0, 0, 0]]]).bool()
    valid = torch.tensor([[1, 1], [1, 0]]).bool()
    gradients = []
    outputs = []
    for enabled in (False, True):
        model.checkpointed = enabled
        model.zero_grad(set_to_none=True)
        observed.grad = context.grad = None
        rng_state = torch.random.get_rng_state().clone()
        out = model.rsample(observed, a, context, valid, generator=torch.Generator().manual_seed(9))
        assert torch.equal(rng_state, torch.random.get_rng_state())
        out.square().sum().backward()
        grads = [observed.grad.clone(), context.grad.clone()] + [p.grad.clone() for p in model.parameters()]
        assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
        gradients.append(grads)
        outputs.append(out.detach())
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    for first, second in zip(*gradients):
        torch.testing.assert_close(first, second)
    expanded = observed.detach().unsqueeze(2).expand_as(outputs[0])
    keep = (a & valid[..., None])[..., None, :, None].expand_as(outputs[0])
    assert torch.equal(outputs[0][keep], expanded[keep])
    assert outputs[0][1, 1].count_nonzero() == 0


def test_visible_queries_skip_denoiser_and_noise():
    model = DifferentiableLatentCSDI()
    assert sum(p.numel() for p in model.parameters()) == 310401
    observed = torch.randn(2, 1, 3, 64, requires_grad=True)
    rng = torch.Generator().manual_seed(4)
    state = rng.get_state().clone()
    model._predict = lambda *args: (_ for _ in ()).throw(AssertionError("unexpected denoising"))
    out = model.rsample(observed, torch.ones(2, 1, 3).bool(), torch.zeros(2, 1, 2304),
                        torch.tensor([[True], [False]]), generator=rng)
    assert torch.equal(state, rng.get_state())
    assert torch.equal(out[0], observed[0].unsqueeze(1).expand_as(out[0]))
    assert out[1].count_nonzero() == 0
    out.sum().backward()
    assert observed.grad[0].abs().sum() > 0
    assert observed.grad[1].count_nonzero() == 0


def test_checkpoint_trains_when_inputs_do_not_require_grad():
    model = DifferentiableLatentCSDI(steps=2, channels=8, layers=1)
    out = model.rsample(torch.zeros(1, 1, 3, 64), torch.zeros(1, 1, 3).bool(),
                        torch.zeros(1, 1, 2304), torch.ones(1, 1).bool())
    out.square().mean().backward()
    assert model.output.weight.grad.abs().sum() > 0


def test_original_sampler_matches_when_all_rows_need_denoising():
    model = DifferentiableLatentCSDI(steps=3, channels=8, layers=1)
    observed = torch.randn(2, 1, 3, 64)
    context = torch.randn(2, 1, 2304)
    a = torch.tensor([[[True, False, True]], [[False, True, False]]])
    valid = torch.ones(2, 1).bool()
    actual = model.rsample(observed, a, context, valid, generator=torch.Generator().manual_seed(42))
    expected = LatentCSDI.sample(model, observed, a, context, valid,
                                generator=torch.Generator().manual_seed(42))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_keyed_noise_is_stable_under_reordering_and_padding():
    model = DifferentiableLatentCSDI(steps=3, channels=8, layers=1)
    observed = torch.randn(2, 1, 3, 64)
    context = torch.randn(2, 1, 2304)
    a = torch.tensor([[[True, False, True]], [[False, True, False]]])
    valid = torch.ones(2, 1).bool()
    state = torch.random.get_rng_state().clone()
    first = model.rsample(observed, a, context, valid, query_keys=["conv7/u0", "conv7/u1"])
    assert torch.equal(state, torch.random.get_rng_state())
    order = torch.tensor([1, 0])
    second = model.rsample(observed[order], a[order], context[order], valid[order],
                           query_keys=["conv7/u1", "conv7/u0"])
    torch.testing.assert_close(first[order], second)
    padded = model.rsample(torch.cat((observed, observed[:1])), torch.cat((a, a[:1])),
                           torch.cat((context, context[:1])), torch.tensor([[1], [1], [0]]).bool(),
                           query_keys=["conv7/u0", "conv7/u1", "padding"])
    torch.testing.assert_close(first, padded[:2])
    assert padded[2].count_nonzero() == 0
    assert len(model._query_noise_cache) == 2
    assert all(value.device.type == "cpu" for value in model._query_noise_cache.values())
    restored = DifferentiableLatentCSDI(steps=3, channels=8, layers=1)
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(first, restored.rsample(observed, a, context, valid,
                               query_keys=["conv7/u0", "conv7/u1"]))
