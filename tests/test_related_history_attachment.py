"""Actual-model tests for the opt-in B1/B2 Flat attachment."""
import copy
import importlib
import types

import pytest
import torch

from tests.test_recent_history_attachment import setup_model


def integration():
    return importlib.import_module('experiments.osram_meld_related_history_20261018.integration')


@pytest.mark.parametrize('kind', ['B1', 'B2'])
def test_zero_init_preserves_original_parameters_outputs_and_rng(kind):
    model, args = setup_model()
    reference = copy.deepcopy(model)
    rng = torch.get_rng_state().clone()
    integration().attach_related_history(model, kind=kind)
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in reference.state_dict().items():
        assert torch.equal(value, model.state_dict()[key]), key
    for training in (False, True):
        reference.train(training); model.train(training)
        torch.manual_seed(19)
        expected = reference(*args, predict_missing=False)[0]
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(19)
        actual = model(*args, predict_missing=False)[0]
        assert torch.equal(expected, actual)
        assert torch.equal(expected_rng, torch.get_rng_state())
    assert model.osram._related_history_inputs is None


def test_disabled_noop_and_legacy_strict_restore():
    model, _ = setup_model()
    before = copy.deepcopy(model.state_dict())
    assert integration().attach_related_history(model, enabled=False) is None
    model.load_state_dict(before, strict=True)
    assert set(model.state_dict()) == set(before)


class RecordingBranch(torch.nn.Module):
    def __init__(self, *, kind, local_dim, context_dim, output_dim, **kwargs):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(.01))
        self.output_dim = output_dim
        self.inputs = None
        self.calls = 0

    def forward(self, **kwargs):
        self.inputs = kwargs
        self.calls += 1
        valid = kwargs['umask'].bool().unsqueeze(-1)
        return torch.where(valid, self.scale.expand(*valid.shape[:2], self.output_dim), 0.)

    def auxiliary_relation_loss(self, labels, dataset):
        self.labels = labels
        return self.scale.square()

    def clear_cache(self):
        pass


def test_actual_masked_flat_inputs_forward_halves_and_skip_are_preserved():
    model, args = setup_model()
    model.eval()
    base_prediction = model(*args, predict_missing=False)[0]
    skip = model.osram.local_skip
    captured = []
    handle = model.osram.emotion_adapter.register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0].detach().clone()))
    branch = integration().attach_related_history(model, kind='B1', module_class=RecordingBranch)
    model(*args, predict_missing=False)
    assert model.osram.local_skip is skip
    actual = branch.inputs
    assert actual['local'].shape[-1] == model.osram.latent_dim
    assert actual['base'].shape[-1] == model.osram.context_dim // 2
    assert actual['gap'].shape[-2:] == (3, model.osram.context_dim // 2)
    missing = ~args[1].bool() & args[3].T.bool().unsqueeze(-1)
    assert torch.count_nonzero(actual['gap'][~missing]) == 0
    assert torch.equal(actual['speakers'], args[2].T)
    assert torch.equal(actual['umask'], args[3].T)
    integration().set_enabled(model, False)
    disabled_prediction = model(*args, predict_missing=False)[0]
    assert torch.equal(base_prediction, disabled_prediction)
    assert torch.equal(captured[0], captured[1])
    handle.remove()


@pytest.mark.parametrize('ablation,zero_base,zero_gap', [
    ('local-only', True, True), ('local-base', False, True), ('local-gap', True, False)])
def test_classification_ablation_cannot_be_bypassed(ablation, zero_base, zero_gap):
    model, args = setup_model()
    model.eval()
    model.osram.osram_emotion_ablation = ablation
    branch = integration().attach_related_history(model, kind='B1', module_class=RecordingBranch)
    model(*args, predict_missing=False)
    if zero_base:
        assert torch.count_nonzero(branch.inputs['base']) == 0
    if zero_gap:
        assert torch.count_nonzero(branch.inputs['gap']) == 0


def test_factory_attaches_before_optimizer_and_auxiliary_loss_registration():
    calls = []
    def construct():
        model, _ = setup_model()
        calls.append(model)
        return model
    trainer = types.SimpleNamespace(MissingM3GraphModel=construct,
        _task_loss=lambda dataset, logits, labels, umask: logits.square().mean())
    integration().install(trainer, kind='B2')
    model = trainer.MissingM3GraphModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    parameters = {id(p) for group in optimizer.param_groups for p in group['params']}
    assert all(id(p) in parameters for p in model.osram.related_history.parameters())
    assert model is calls[0]


def test_auxiliary_is_training_only_once_per_forward_and_detaches_backbone():
    model, args = setup_model()
    branch = integration().attach_related_history(model, kind='B2')
    trainer = types.SimpleNamespace(_task_loss=lambda dataset, logits, labels, umask: logits.sum() * 0.)
    integration().install_task_loss(trainer, lambda: model, relation_weight=.05)
    labels = torch.tensor([[0, 1, 0, 1, 0], [1, 0, 0, 0, 0]])
    model.train()
    prediction = model(*args, predict_missing=False)[0]
    loss = trainer._task_loss('MELD', prediction, labels, args[3])
    assert loss > 0
    loss.backward()
    grads = [(name, p.grad) for name, p in model.named_parameters() if p.grad is not None]
    assert any(name.startswith('osram.related_history.') and bool(grad.abs().sum()) for name, grad in grads)
    assert all(name.startswith('osram.related_history.') or not bool(grad.abs().sum()) for name, grad in grads)
    again = trainer._task_loss('MELD', prediction, labels, args[3])
    assert again.item() == 0
    model.eval()
    with torch.no_grad():
        prediction = model(*args, predict_missing=False)[0]
        assert trainer._task_loss('MELD', prediction, labels, args[3]).item() == 0


@pytest.mark.parametrize('kind', ['B1', 'B2'])
def test_joint_optimizer_updates_branch_and_strict_new_checkpoint_restore(kind):
    model, args = setup_model()
    integration().attach_related_history(model, kind=kind)
    trainer = types.SimpleNamespace(_task_loss=lambda dataset, logits, labels, umask: logits.square().mean())
    integration().install_task_loss(trainer, lambda: model)
    labels = torch.zeros_like(args[3]).long()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    original = copy.deepcopy(model.state_dict())
    for _ in range(3):
        optimizer.zero_grad()
        logits = model(*args, predict_missing=False)[0]
        trainer._task_loss('MELD', logits, labels, args[3]).backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
    assert any(not torch.equal(value, original[name]) for name, value in model.state_dict().items()
               if name.startswith('osram.related_history.'))
    assert any(not torch.equal(value, original[name]) for name, value in model.state_dict().items()
               if name.startswith('osram.emotion_adapter.'))
    other, _ = setup_model()
    integration().attach_related_history(other, kind=kind)
    other.load_state_dict(model.state_dict(), strict=True)
