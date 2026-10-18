import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn


def module():
    try:
        return importlib.import_module('experiments.osram_persistent_patterns_20261018.evaluate')
    except ModuleNotFoundError:
        pytest.fail('Persistent pattern evaluator not implemented')


def test_all_patterns_and_checkpoint_rule():
    m = module()
    assert set(m.PATTERNS) == {'A', 'T', 'V', 'AT', 'AV', 'TV', 'ATV'}
    for pattern, bits in m.PATTERNS.items():
        a = m.fixed_availability(torch.tensor([[1., 1.], [1., 0.]]), pattern)
        assert torch.equal(a[torch.tensor([[1, 1], [1, 0]]).T.bool()], torch.tensor(bits).float().expand(3, -1))
        assert torch.count_nonzero(a[1, 1]) == 0
        assert m.CHECKPOINT_RATE[pattern] == {1: '0.7', 2: '0.3', 3: '0.0'}[sum(bits)]


@pytest.mark.parametrize('classes', [4, 6, 7])
def test_ua_uses_actual_class_count(classes):
    labels = np.array([0, 1, 2, 3])
    logits = np.zeros((4, classes)); logits[:, 0] = 1
    score = module().classification_metrics(labels, logits, classes)
    assert score['ACC'] == 25
    assert score['UA'] == pytest.approx(100/classes)
    assert len(score['recall']) == len(score['support']) == classes


def test_frozen_replay_and_fixed_history_use_identical_metadata_without_mutation():
    m = module()
    torch.manual_seed(3)
    class Model(nn.Module):
        def __init__(self):
            super().__init__(); self.head = nn.Linear(6, 4); self.calls = 0
        def forward(self, inputs, availability, qmask, umask, lengths, predict_missing=False):
            self.calls += 1
            return self.head(inputs[0]), None, None, None
    raw = [torch.randn(3, 2, 2) for _ in range(6)]
    raw += [torch.tensor([[0, 1, 0], [1, 0, 0]]), torch.tensor([[1., 1., 1.], [1., 1., 0.]]),
            torch.tensor([[0, 1, 2], [3, 0, 0]]), ['x', 'y']]
    def prepare(data, host, guest, dimensions):
        assert torch.equal(host, guest)
        return dict(incomplete=torch.cat(data[:3], -1)*host.repeat_interleave(2, -1),
            availability=host, qmask=data[6], umask=data[7], lengths=[3, 2])
    trainer = SimpleNamespace(_move_batch=lambda raw, device: raw, _prepare_view_from_primary_masks=prepare)
    model = Model().eval().requires_grad_(False)
    availability = m.fixed_availability(raw[7], 'ATV')
    valid = raw[7].bool().numpy()
    saved = dict(labels=raw[8].numpy()[valid], speakers=raw[6].numpy()[valid],
        conversation_ids=np.broadcast_to(np.array(['x', 'y'])[:, None], valid.shape)[valid],
        utterance_indices=np.broadcast_to(np.arange(3), valid.shape)[valid],
        availability=availability.transpose(0, 1).numpy()[valid])
    saved['logits'] = model([prepare(raw, availability, availability, (2, 2, 2))['incomplete']],
        availability, raw[6], raw[7], [3, 2])[0].transpose(0, 1).numpy()[valid]
    saved['predictions'] = saved['logits'].argmax(1)
    model.calls = 0; before = {k: v.clone() for k, v in model.state_dict().items()}
    result, _ = m.evaluate_condition(model, [raw], (2, 2, 2), saved, trainer, None, 4)
    assert result['checks']['selected_classes_exact'] and model.calls == 1
    result, values = m.evaluate_condition(model, [raw], (2, 2, 2), saved, trainer, 'A', 4)
    np.testing.assert_array_equal(values['availability'], np.array([[1, 0, 0]]*5))
    assert result['metrics']['N'] == 5 and model.calls == 2
    for k, v in model.state_dict().items(): assert torch.equal(v, before[k])
    saved['labels'] = saved['labels'][::-1]
    with pytest.raises((AssertionError, ValueError)):
        m.evaluate_condition(model, [raw], (2, 2, 2), saved, trainer, 'T', 4)
    assert model.calls == 2
