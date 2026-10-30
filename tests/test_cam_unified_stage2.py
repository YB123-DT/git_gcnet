"""Frozen Nested compatibility and strict checkpoint ownership checks."""
import importlib
import importlib.util
from types import SimpleNamespace

import pytest
import torch
from torch import nn


def api():
    name = 'experiments.cam_unified_20261030.stage2'
    assert importlib.util.find_spec(name) is not None, 'unified stage2 entry is missing'
    return importlib.import_module(name)


class SmallAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(4352, 1)
        self.output = nn.Linear(1, 1600)

    def forward(self, value):
        return self.output(self.input(value))


class Parent(nn.Module):
    """Tiny computational substitute with the exact original Flat interface."""
    def __init__(self):
        super().__init__()
        self.osram = nn.Module()
        self.osram.local_skip = nn.Linear(256, 1600)
        self.osram.emotion_adapter = SmallAdapter()
        self.osram.emotion_norm = nn.LayerNorm(1600)
        self.smax_fc = nn.Linear(1600, 1)
        self.calls = 0

    def forward(self, inputs, availability, qmask, umask, lengths, predict_missing=False):
        self.calls += 1
        value = inputs[0]
        hidden = self.osram.emotion_norm(
            self.osram.local_skip(value[..., :256].contiguous())
            + self.osram.emotion_adapter(value)) * umask.T[..., None]
        return self.smax_fc(hidden), hidden, None, None


def view():
    availability = torch.tensor([[[1, 0, 1]], [[1, 0, 0]], [[0, 1, 1]]])
    local = torch.randn(3, 1, 256)
    base = torch.cat((torch.randn(3, 1, 512), torch.zeros(3, 1, 512)), -1)
    gap = torch.cat((torch.randn(3, 1, 3, 512), torch.zeros(3, 1, 3, 512)), -1)
    base[0] = 0
    gap[0] = 0
    gap *= (~availability.bool())[..., None]
    return dict(incomplete=torch.cat((local, base, gap.flatten(-2)), -1),
                availability=availability,
                umask=torch.ones(1, 3), qmask=torch.zeros(3, 1, 2), lengths=[3])


def test_original_nested_is_single_pass_frozen_and_initially_identical():
    torch.manual_seed(66)
    module = api()
    parent = Parent()
    batch = view()
    expected = parent([batch['incomplete']], batch['availability'], batch['qmask'],
                      batch['umask'], batch['lengths'])[0].detach()
    model = module.FrozenNestedModel(parent).eval()
    parent.calls = 0
    actual = model(batch)
    assert parent.calls == 1
    assert torch.equal(actual['prediction'], expected)
    assert torch.equal(actual['base_prediction'], expected)
    assert sum(p.numel() for p in model.block.parameters()) == 159235
    model.train()
    assert not model.parent.training
    result = model(batch)
    result['prediction'].square().mean().backward()
    assert all(p.grad is None for p in parent.parameters())
    assert any(p.grad is not None and bool(p.grad.ne(0).any()) for p in model.block.parameters())


def test_decode_keeps_original_local_skip():
    module = api()
    model = module.FrozenNestedModel(Parent()).eval()
    batch = view()
    raw = batch['incomplete']
    frame = SimpleNamespace(local=raw[..., :256], base=raw[..., 256:1280],
                            gap=raw[..., 1280:].reshape(3, 1, 3, 1024),
                            valid=batch['umask'].T.bool())
    changed = frame.local + 2
    adapter_input = torch.cat((changed, frame.base, frame.gap.flatten(-2)), -1)
    hidden = model.parent.osram.emotion_norm(
        model.parent.osram.local_skip(frame.local.contiguous())
        + model.parent.osram.emotion_adapter(adapter_input))
    assert torch.equal(model._decode(frame, (changed, frame.base, frame.gap)),
                       model.parent.smax_fc(hidden))


def test_parent_restore_is_strict_and_rejects_block_only(tmp_path):
    module = api()
    parent = Parent()
    path = tmp_path / 'flat.pt'
    torch.save(dict(model=parent.state_dict(), epoch=3), path)
    restored = Parent()
    identity = module.restore_parent(restored, path)
    assert len(identity['sha256']) == 64
    assert all(torch.equal(p, restored.state_dict()[k]) for k, p in parent.state_dict().items())
    with pytest.raises(ValueError, match='SHA256'):
        module.restore_parent(restored, path, expected_sha256='0' * 64)
    torch.save(dict(block={}), path)
    with pytest.raises(ValueError, match='block-only'):
        module.restore_parent(restored, path)
    state = parent.state_dict()
    state.pop('smax_fc.bias')
    torch.save(dict(model=state), path)
    with pytest.raises(RuntimeError, match='Missing key'):
        module.restore_parent(restored, path)


def test_nested_block_requires_exact_parent_identity_and_strict_state(tmp_path):
    module = api()
    model = module.FrozenNestedModel(Parent())
    parent_id = 'a' * 64
    path = tmp_path / 'nested.pt'
    torch.save(module.block_checkpoint(model, parent_id, epoch=4), path)
    restored = module.FrozenNestedModel(Parent())
    module.restore_block(restored, path, parent_id)
    with pytest.raises(ValueError, match='parent'):
        module.restore_block(restored, path, 'b' * 64)
    payload = torch.load(path, map_location='cpu', weights_only=False)
    payload['block'].pop(next(iter(payload['block'])))
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match='Missing key'):
        module.restore_block(restored, path, parent_id)


def test_inference_rejects_parameter_invisible_query_config_change():
    name = 'experiments.cam_unified_20261030.infer'
    assert importlib.util.find_spec(name) is not None, 'unified inference entry is missing'
    infer = importlib.import_module(name)
    expected = dict(osram_gap_read='residual', train_missing_rates=(0., .1))
    infer.validate_parent_config(expected, dict(expected, train_missing_rates=[0., .1]))
    with pytest.raises(ValueError, match='osram_gap_read'):
        infer.validate_parent_config(expected, dict(expected, osram_gap_read='raw'))
    with pytest.raises(ValueError, match='config'):
        infer.validate_parent_config(expected, dict(osram_gap_read='residual'))
