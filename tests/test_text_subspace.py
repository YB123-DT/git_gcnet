"""Gradient boundaries for task-relevant AND predictable Text targets."""
import importlib.util
from types import SimpleNamespace

import pytest
import torch


def test_feature_available():
    assert importlib.util.find_spec('gcnet_missing_m3.text_subspace') is not None


def test_stage1_separate_gradients_and_teacher_boundary():
    from gcnet_missing_m3.text_subspace import TextSubspacePretrainer
    torch.manual_seed(5)
    model = TextSubspacePretrainer()
    z = {m: torch.randn(12, 256, requires_grad=True) for m in ('audio', 'text', 'visual')}
    losses = model.loss(z, torch.randn(12))
    for name in ('sentiment', 'predictability'):
        grad = torch.autograd.grad(losses[name], tuple(model.projector.parameters()), retain_graph=True)
        assert sum(g.abs().sum() for g in grad) > 0
    assert sum(g.abs().sum() for g in torch.autograd.grad(
        losses['predictability'], tuple(model.predictor.parameters()), retain_graph=True)) > 0
    losses['total'].backward()
    assert all(value.grad is None for value in z.values())


def test_stage1_predictor_cannot_access_text_or_hidden():
    from gcnet_missing_m3.text_subspace import TextSubspacePretrainer
    model = TextSubspacePretrainer()
    a, v = torch.randn(4, 256), torch.randn(4, 256)
    assert torch.equal(model.predict(a, v, 'A'), model.predict(a, v * 99, 'A'))
    assert torch.equal(model.predict(a, v, 'V'), model.predict(a * 99, v, 'V'))
    assert model.predict(a, v, 'AV').shape == (4, 32)
    with pytest.raises(ValueError):
        model.predict(a, v, 'T')


def test_safeguard_statistics_do_not_claim_full_rank():
    from gcnet_missing_m3.text_subspace import representation_stats, variance_covariance
    z = torch.ones(8, 32, requires_grad=True)
    var, cov = variance_covariance(z)
    assert var > 0 and cov == 0
    stats = representation_stats(z)
    assert stats['effective_rank'] == 0
    assert stats['per_dim_std'] == [0.] * 32
    assert stats['covariance_off_diagonal'] == 0
    assert stats['collapsed']
    v, c = variance_covariance(z[:1])
    assert v == 0 and c == 0


def predictions():
    reg = torch.randn(4, 2, 3, 256, requires_grad=True)
    cl = torch.randn_like(reg, requires_grad=True)
    mask = torch.ones(4, 2, 3, dtype=torch.bool)
    return SimpleNamespace(reg_predictions=reg, cl_predictions=cl, target_mask=mask)


@pytest.mark.parametrize('project', [False, True])
def test_text_loss_ignores_audio_visual_and_frozen_projection_passes_gradient(project):
    from gcnet_missing_m3.text_subspace import text_jepa_loss
    p = predictions()
    targets = {m: torch.randn(4, 2, 256, requires_grad=True) for m in ('audio', 'text', 'visual')}
    r = torch.nn.Linear(256, 32).requires_grad_(False) if project else None
    loss = text_jepa_loss(p, targets, r)
    assert loss.target_count == 8
    torch.testing.assert_close(loss.total, .5 * loss.regression + .5 * loss.contrastive)
    loss.total.backward()
    for tensor in (p.reg_predictions, p.cl_predictions):
        assert tensor.shape[-1] == 256
        assert tensor.grad[..., 1, :].abs().sum() > 0
        assert torch.count_nonzero(tensor.grad[..., (0, 2), :]) == 0
    assert all(t.grad is None for t in targets.values())
    if r is not None:
        assert all(t.grad is None for t in r.parameters())


def test_empty_and_single_text_targets():
    from gcnet_missing_m3.text_subspace import text_jepa_loss
    p = predictions()
    p.target_mask[..., 1] = False
    targets = {'text': torch.randn(4, 2, 256)}
    loss = text_jepa_loss(p, targets)
    assert loss.total == 0 and loss.target_count == 0
    loss.total.backward()
    assert p.reg_predictions.grad is not None and p.cl_predictions.grad is not None
    p.target_mask[0, 0, 1] = True
    loss = text_jepa_loss(p, targets)
    assert loss.target_count == 1 and loss.contrastive == 0
    assert loss.total == .5 * loss.regression


def test_subspace_checkpoint_rejects_test_selection_and_wrong_teacher(tmp_path):
    from gcnet_missing_m3.text_subspace import TextSubspacePretrainer, save_subspace, load_frozen_subspace
    m = TextSubspacePretrainer()
    path = tmp_path / 'subspace.pt'
    save_subspace(path, m, teacher_sha256='abc', epoch=2, validation_loss=1., config={})
    r, provenance = load_frozen_subspace(path, 'abc', 256)
    assert r.weight.shape == (32, 256) and not r.training
    assert all(not p.requires_grad for p in r.parameters())
    assert provenance['teacher_projector_sha256'] == 'abc'
    assert torch.equal(r.weight, m.projector.weight)
    with pytest.raises(ValueError, match='Teacher'):
        load_frozen_subspace(path, 'wrong', 256)
    payload = torch.load(path, weights_only=False)
    payload['selection_split'] = 'test'
    torch.save(payload, path)
    with pytest.raises(ValueError, match='validation'):
        load_frozen_subspace(path, 'abc', 256)
