"""CPU checks of core mechanisms, masking, state lifetime, and source algebra."""
import torch

from gcnet_missing_m3.core20_probability import (
    MODALITIES, MMVAE, NaturalPosteriorNetwork, PFRNNMemory, RKNMemory, build_memory,
)


def memory_inputs():
    torch.manual_seed(17)
    keys = {m: torch.randn(5, 2, 2, 4) for m in MODALITIES}
    values = {m: torch.randn(5, 2, 2, 3) for m in MODALITIES}
    queries = torch.randn(5, 2, 4, 2, 4)
    av = torch.tensor([[[1, 1, 1], [1, 0, 1]], [[1, 0, 0], [0, 1, 0]],
                       [[0, 0, 0], [1, 1, 0]], [[0, 1, 1], [0, 0, 0]],
                       [[1, 1, 1], [1, 1, 1]]])
    valid = torch.tensor([[1, 1], [1, 1], [1, 1], [1, 0], [0, 0]]).bool()
    return keys, values, queries, av, valid


def _memory_finite_gradients_reset_and_read_before_write(method):
    model = build_memory(method, num_heads=2, key_dim=4, value_dim=3, latent_dim=8)
    args = memory_inputs()
    torch.manual_seed(29)
    base, gap, diag = model.scan(*args)
    assert base.shape == (5, 2, 6) and gap.shape == (5, 2, 3, 6)
    assert torch.equal(base[0], torch.zeros_like(base[0]))
    assert not base[1:].eq(0).all()
    assert base[~args[-1]].eq(0).all() and gap[~args[-1]].eq(0).all()
    assert gap[args[3].bool()].eq(0).all()
    assert all(len(diag[m]["rho"]) == 14 for m in MODALITIES)
    (base.square().mean() + gap.square().mean()).backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)
    torch.manual_seed(29)
    again = model.scan(*args)
    torch.testing.assert_close(base, again[0])
    torch.testing.assert_close(gap, again[1])


def _memory_nan_masking_padding_and_batch_isolation(method):
    model = build_memory(method, num_heads=2, key_dim=4, value_dim=3, latent_dim=8).eval()
    keys, values, queries, av, valid = memory_inputs()
    torch.manual_seed(31)
    baseline = model.scan(keys, values, queries, av, valid)
    corrupted_k, corrupted_v = {}, {}
    active = av.bool() & valid[..., None]
    for i, m in enumerate(MODALITIES):
        corrupted_k[m] = keys[m].masked_fill(~active[..., i, None, None], float("nan"))
        corrupted_v[m] = values[m].masked_fill(~active[..., i, None, None], float("nan"))
    corrupted_q = queries.masked_fill(~valid[..., None, None, None], float("nan"))
    torch.manual_seed(31)
    masked = model.scan(corrupted_k, corrupted_v, corrupted_q, av, valid)
    torch.testing.assert_close(baseline[0], masked[0])
    torch.testing.assert_close(baseline[1], masked[1])
    # Change only the other conversation, using the same random draws for PF-RNN.
    changed_k = {m: k.clone() for m, k in keys.items()}
    changed_v = {m: v.clone() for m, v in values.items()}
    for m in MODALITIES:
        changed_k[m][:, 1] += 500
        changed_v[m][:, 1] -= 100
    torch.manual_seed(31)
    other = model.scan(changed_k, changed_v, queries, av, valid)
    torch.testing.assert_close(baseline[0][:, 0], other[0][:, 0])
    torch.testing.assert_close(baseline[1][:, 0], other[1][:, 0])


def test_memory_finite_gradients_reset_and_read_before_write():
    for method in ("rkn", "pfrnn"):
        _memory_finite_gradients_reset_and_read_before_write(method)


def test_memory_nan_masking_padding_and_batch_isolation():
    for method in ("rkn", "pfrnn"):
        _memory_nan_masking_padding_and_batch_isolation(method)


def test_source_kalman_algebra_matches_dense_conditioning():
    mean = torch.tensor([.2, -.3], dtype=torch.double)
    cov = tuple(torch.tensor([x], dtype=torch.double) for x in (2., 3., .4))
    observed, variance = torch.tensor([1.1], dtype=torch.double), torch.tensor([.7], dtype=torch.double)
    actual_mean, actual_cov = RKNMemory.kalman_update(mean, cov, observed, variance)
    dense = torch.tensor([[2., .4], [.4, 3.]], dtype=torch.double)
    gain = dense[:, 0] / (dense[0, 0] + variance)
    expected_mean = mean + gain * (observed - mean[0])
    expected_cov = dense - gain[:, None] * dense[0:1]
    torch.testing.assert_close(actual_mean, expected_mean)
    torch.testing.assert_close(torch.stack(actual_cov).squeeze(),
                               torch.stack([expected_cov[0, 0], expected_cov[1, 1], expected_cov[0, 1]]))


def test_source_soft_resampling_importance_correction():
    weights = torch.tensor([[.1, .3, .6]])
    idx = torch.tensor([[2, 2, 0]])
    actual = PFRNNMemory.resampling_correction(weights.log(), idx, .5).exp()
    proposal = .5 * weights + .5 / 3
    expected = (weights / proposal).gather(-1, idx)
    expected = expected / expected.sum(-1, keepdim=True)
    torch.testing.assert_close(actual, expected)


def latent_inputs():
    latents = {m: torch.randn(3, 2, 8) for m in MODALITIES}
    av = torch.tensor([[[1, 1, 1], [1, 0, 0]], [[0, 1, 0], [0, 0, 0]], [[1, 1, 1], [1, 1, 1]]])
    umask = torch.tensor([[1, 1, 0], [1, 1, 0]])
    return latents, av, umask


def test_mmvae_mixture_generative_objective_and_masking():
    dimensions = dict(audio=5, text=9, visual=7)
    model = MMVAE(8, dimensions)
    latents, av, umask = latent_inputs()
    targets = {m: torch.randn(3, 2, d, requires_grad=True) for m, d in dimensions.items()}
    torch.manual_seed(41)
    rep, loss, meta = model(latents, av, umask, reconstruction_targets=targets)
    torch.testing.assert_close(loss, meta["mixture_kl_mc"] + meta["reconstruction_nll"])
    assert meta["mixture_weights"][~meta["available"]].eq(0).all()
    assert rep[~umask.T.bool()].eq(0).all()
    loss.backward()
    for i, m in enumerate(MODALITIES):
        assert targets[m].grad[~meta["available"][..., i]].eq(0).all()
        assert torch.isfinite(targets[m].grad).all()
        assert targets[m].grad[meta["available"][..., i]].abs().sum() > 0
    for family in (model.encoders, model.decoders):
        for module in family.values():
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
    poisoned_targets = {m: x.masked_fill(~meta["available"][..., i, None], float("nan"))
                        for i, (m, x) in enumerate(targets.items())}
    torch.manual_seed(41)
    poisoned = model(latents, av, umask, reconstruction_targets=poisoned_targets)
    torch.testing.assert_close(loss, poisoned[1])
    changed = {m: x + 10 for m, x in targets.items()}
    torch.manual_seed(41)
    changed_result = model(latents, av, umask, reconstruction_targets=changed)
    torch.testing.assert_close(rep, changed_result[0])
    assert changed_result[1] > loss
    try:
        model(latents, av, umask)
    except ValueError as error:
        assert "reconstruction_targets" in str(error)
    else:
        raise AssertionError("training must require original feature targets")
    train_empty = model(
        {m: torch.full_like(x, float("nan")) for m, x in latents.items()}, av * 0, umask,
        reconstruction_targets={m: torch.full_like(x, float("nan")) for m, x in targets.items()})
    assert train_empty[0].eq(0).all() and train_empty[1].item() == 0
    model.eval()
    reference = model(latents, av, umask)
    corrupt = {m: x.masked_fill(~meta["available"][..., i, None], float("nan"))
               for i, (m, x) in enumerate(latents.items())}
    result = model(corrupt, av, umask)
    torch.testing.assert_close(reference[0], result[0])
    torch.testing.assert_close(reference[1], result[1])
    # Evaluation does not call likelihood decoders or consult targets.
    handles = [decoder.register_forward_pre_hook(lambda *_: (_ for _ in ()).throw(
        AssertionError("evaluation must skip generative decoder"))) for decoder in model.decoders.values()]
    try:
        eval_result = model(latents, av, umask)
        torch.testing.assert_close(eval_result[0], rep)
        assert eval_result[1].item() == 0
    finally:
        for handle in handles:
            handle.remove()
    # All unavailable: no posterior expert, no likelihood target, zero objective.
    empty = model({m: torch.full_like(x, float("nan")) for m, x in latents.items()}, av * 0, umask)
    assert empty[0].eq(0).all() and empty[1].item() == 0


def test_natpn_source_conjugate_update_loss_and_density_gradient():
    model = NaturalPosteriorNetwork(hidden_dim=12, latent_dim=4)
    hidden = torch.randn(3, 2, 12)
    umask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    mean, params = model(hidden, umask)
    assert mean.shape == (3, 2, 1) and mean[~umask.T.bool()].eq(0).all()
    loss = model.loss(torch.randn(2, 3), umask, params)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.flow.reference.grad.abs().sum() > 0
    reference = model(hidden, umask)
    corrupted = hidden.masked_fill(~umask.T.bool()[..., None], float("nan"))
    torch.testing.assert_close(reference[0], model(corrupted, umask)[0])
    update = model.conjugate_update(torch.tensor(2.), torch.tensor(4.), torch.tensor(3.))
    assert update["mu"].item() == 1.5 and update["lambda"].item() == 4
    assert update["alpha"].item() == 2
    torch.testing.assert_close(update["beta"], torch.tensor(51.875))
