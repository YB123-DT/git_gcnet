import torch

from osram_four_block_wiring.csdi_latent import FixedLatentProjection, HypothesisBridge, LatentCSDI


def _inputs():
    torch.manual_seed(4)
    z = torch.randn(2, 1, 3, 64)
    a = torch.tensor([[[True, False, True]], [[False, True, False]]])
    return z, a, torch.randn(2, 1, 2304), torch.tensor([[True], [False]])


def test_projection_masks_after_centering():
    p = FixedLatentProjection(torch.ones(3, 256), torch.randn(3, 64, 256))
    a = torch.tensor([[[True, False, True]]])
    result = p(torch.zeros(1, 1, 3, 256), a)
    assert torch.equal(result[:, :, 1], torch.zeros_like(result[:, :, 1]))
    assert not list(p.parameters())


def test_sampling_preserves_observed_and_generator():
    z, a, context, valid = _inputs()
    model = LatentCSDI(steps=3, channels=8, layers=2)
    first = model.sample(z, a, context, valid, generator=torch.Generator().manual_seed(10))
    second = model.sample(z, a, context, valid, generator=torch.Generator().manual_seed(10))
    assert first.shape == (2, 1, 4, 3, 64)
    assert torch.equal(first, second)
    observed_mask = (a & valid[..., None]).unsqueeze(2).unsqueeze(-1).expand_as(first)
    assert torch.equal(first[observed_mask], z.unsqueeze(2).expand_as(first)[observed_mask])
    assert torch.count_nonzero(first[1]) == 0


def test_loss_masks_and_zero_exit():
    z, a, context, valid = _inputs()
    model = LatentCSDI(steps=3, channels=8, layers=1)
    target = torch.randn_like(z)
    loss = model.noise_loss(z, target, a, context, valid, torch.Generator().manual_seed(2))
    changed = target.clone()
    changed[a | ~valid[..., None]] += 100
    other = model.noise_loss(z, changed, a, context, valid, torch.Generator().manual_seed(2))
    assert torch.equal(loss, other)
    loss.backward()
    assert model.output.weight.grad.abs().sum() > 0
    model.zero_grad()
    zero = model.noise_loss(z, target, torch.ones_like(a), context, valid)
    zero.backward()
    assert zero.item() == 0


def test_frozen_sample_bridge_gradient_and_hard_zero():
    z, a, context, valid = _inputs()
    model = LatentCSDI(steps=2, channels=8, layers=1).requires_grad_(False)
    samples = model.sample(z, a, context, valid)
    bridge = HypothesisBridge()
    local = torch.randn(2, 1, 256, requires_grad=True)
    output = bridge(local, samples, a, valid)
    assert output.shape == (2, 1, 4, 1600)
    assert torch.count_nonzero(output) == 0
    output.sum().backward()
    assert bridge.net[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.parameters())
    with torch.no_grad():
        bridge.net[-1].bias.fill_(1)
    all_available = torch.ones_like(a)
    assert torch.count_nonzero(bridge(local, samples, all_available, valid)) == 0
    assert torch.count_nonzero(bridge(local, samples, a, valid)[1]) == 0


def test_query_sampling_is_independent_of_permutation_and_padding():
    z, a, context, valid = _inputs()
    valid[:] = True
    model = LatentCSDI(steps=3, channels=8, layers=1)
    keys = ["parent1/rate.5/conv7/utt0", "parent1/rate.5/conv7/utt1"]
    baseline = model.sample(z, a, context, valid, query_keys=keys)
    order = torch.tensor([1, 0])
    reordered = model.sample(z[order], a[order], context[order], valid[order],
                             query_keys=[keys[1], keys[0]])
    torch.testing.assert_close(baseline[order], reordered, rtol=1e-5, atol=1e-6)
    padded = model.sample(torch.cat((z, z[:1])), torch.cat((a, a[:1])),
                          torch.cat((context, context[:1])),
                          torch.tensor([[True], [True], [False]]),
                          query_keys=[[keys[0]], [keys[1]], ["padding"]])
    torch.testing.assert_close(baseline, padded[:2], rtol=1e-5, atol=1e-6)
    assert torch.count_nonzero(padded[2]) == 0
