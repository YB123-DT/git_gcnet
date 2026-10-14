"""Frozen trace invariants using the real causal OSRAM with a tiny feature wrapper."""

import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gcnet_missing_m3.osram import OSRAMBackbone
from gcnet_modality_jepa.gcnet_legacy_random_mask import random_mask
from experiments.osram_meld_gcnet_random_no_skip_evidence_20261014.evaluate import (
    replay_view_from_availability, scatter_saved_availability,
)
from experiments.osram_meld_write_retention_20261014.run import WriteCapture, state_digest
from experiments.osram_meld_write_retention_20261014.task_replay import TaskCapture


def trace_module():
    name = 'experiments.osram_meld_query_decision_20261014.trace'
    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError:
        spec = None
    assert spec is not None, 'Frozen trace extraction is missing'
    return importlib.import_module(name)


class TinySource(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.osram = OSRAMBackbone(latent_dim=4, output_dim=4, num_heads=2,
                                  key_dim=2, value_dim=2, n_speakers=2, bidirectional=False,
                                  dropout=0)
        self.smax_fc = torch.nn.Linear(4, 7)
        torch.nn.init.normal_(self.osram.emotion_adapter[-1].weight, std=.2)

    def forward(self, inputs, availability, qmask, umask, lengths, predict_missing=False):
        assert not predict_missing
        latents = dict(zip(('audio', 'text', 'visual'), inputs[0].split(4, -1)))
        node = sum(latents.values()) / availability.sum(-1, keepdim=True).clamp_min(1)
        hidden, _ = self.osram(node, latents, availability, qmask, umask, lengths)
        return self.smax_fc(hidden), None, None, None


def fixture():
    torch.manual_seed(20)
    model = TinySource().eval()
    umask = torch.tensor([[1., 1., 1., 1.], [1., 1., 0., 0.]])
    qmask = torch.tensor([[0, 1, 0, 1], [1, 0, 0, 0]])
    labels = torch.tensor([[0, 1, 2, 3], [4, 5, -1, -1]])
    features = [torch.randn(4, 2, 4) for _ in range(6)]
    raw = [*features, qmask, umask, labels, ['a', 'b']]
    availability = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1],
                             [0, 1, 1], [1, 0, 0]], dtype=np.float32)
    tensor, _ = scatter_saved_availability(availability, 0, umask)
    view = replay_view_from_availability(raw, tensor, (4, 4, 4))
    with torch.no_grad(), TaskCapture(model.osram) as task, WriteCapture(model.osram, umask) as writes:
        logits = model([view['incomplete']], tensor, qmask, umask, [4, 2])[0]
    valid = umask.T.bool()
    saved = dict(availability=availability, labels=labels[umask.bool()].numpy(),
                 predictions=logits.transpose(0, 1)[umask.bool()].argmax(-1).numpy())
    return model, raw, saved, task, writes, logits, valid


def test_saved_trace_is_prewrite_batch_major_frozen_and_past_only():
    module = trace_module()
    model, raw, saved, task, writes, logits, _ = fixture()
    before = state_digest(model)
    arrays, checks = module.extract_split(model, [raw], SimpleNamespace(dataset='MELD', mosi_task_mode='regression'),
                                          (4, 4, 4), .7, 'test', saved=saved)
    assert arrays['conversation_ids'].tolist() == ['a'] * 4 + ['b'] * 2
    assert arrays['utterance_indices'].tolist() == [0, 1, 2, 3, 0, 1]
    assert arrays['has_history'].tolist() == [False, True, True, True, False, True]
    np.testing.assert_array_equal(arrays['history_seen'], [[0, 0, 0], [1, 0, 0], [1, 1, 0],
                                                         [1, 1, 1], [0, 0, 0], [0, 1, 1]])
    np.testing.assert_array_equal(arrays['labels'], saved['labels'])
    np.testing.assert_array_equal(arrays['availability'], saved['availability'])
    valid = raw[7].bool()
    for name, tensor in [('local', task.local), ('queries', task.queries), ('base', task.base),
                         ('gap', task.gap), ('logits', logits)]:
        np.testing.assert_allclose(arrays[name], tensor.transpose(0, 1)[valid].numpy(), atol=1e-6)
    for index, (conversation, t) in enumerate(zip(arrays['conversation_ids'], arrays['utterance_indices'])):
        batch = 0 if conversation == 'a' else 1
        snapshot = writes.snapshots[t]
        np.testing.assert_allclose(arrays['memory'][index], snapshot[1][batch].numpy(), atol=1e-6)
        np.testing.assert_allclose(arrays['keys'][index], snapshot[2][batch].numpy(), atol=1e-6)
    assert not arrays['memory'][[0, 4]].any()
    assert not np.allclose(arrays['memory'][1], writes.snapshots[1][-1][0].numpy())
    assert checks['saved_predictions_exact'] and checks['full_saved_coverage']
    assert checks['source_state_unchanged'] and state_digest(model) == before
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    for name in ('local', 'memory', 'queries', 'keys', 'availability', 'base', 'gap', 'logits'):
        assert arrays[name].dtype == np.float32


def test_offline_view_matches_exact_gcnet_sampler_and_selected_speaker_without_global_rng_changes():
    module = trace_module()
    model, raw, _, _, _, _, _ = fixture()
    rng = np.random.RandomState(66)
    sides = [random_mask(3, 8, .6, rng=rng).reshape(2, 4, 3) for _ in range(2)]
    expected = np.where(raw[6].numpy()[..., None] == 0, sides[0], sides[1])[raw[7].bool().numpy()]
    np.random.seed(89)
    before = np.random.get_state()
    arrays, checks = module.extract_split(model, [raw], SimpleNamespace(dataset='MELD', mosi_task_mode='regression'),
                                          (4, 4, 4), .6, 'train', mask_seed=66)
    np.testing.assert_array_equal(arrays['availability'], expected)
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    assert checks['mask_protocol'] == 'gcnet-random-mask-v1'
    assert 'offline' in checks['mask_scope']
    again, _ = module.extract_split(model, [raw], SimpleNamespace(dataset='MELD', mosi_task_mode='regression'),
                                   (4, 4, 4), .6, 'train', mask_seed=66)
    np.testing.assert_array_equal(again['availability'], arrays['availability'])


def test_saved_trace_rejects_mismatch_and_unconsumed_rows_but_marks_partial():
    module = trace_module()
    model, raw, saved, *_ = fixture()
    args = (model, [raw], SimpleNamespace(dataset='MELD', mosi_task_mode='regression'), (4, 4, 4), .7, 'test')
    with pytest.raises(AssertionError):
        module.extract_split(*args, saved={**saved, 'predictions': (saved['predictions'] + 1) % 7})
    extended = {name: np.concatenate((value, value)) for name, value in saved.items()}
    with pytest.raises(ValueError, match='fully consumed'):
        module.extract_split(*args, saved=extended)
    _, checks = module.extract_split(*args, saved=extended, max_batches=1)
    assert checks['partial'] and not checks['full_saved_coverage']


def test_donors_use_only_split_availability_history_and_different_conversations():
    module = trace_module()
    data = dict(conversation_ids=np.array(['a', 'a', 'b', 'b', 'c', 'c', 'solo']),
                availability=np.array([[1, 0, 0]] * 6 + [[0, 1, 0]]),
                has_history=np.array([False, True, False, True, False, True, False]),
                labels=np.arange(7), history_seen=np.eye(7, 3))
    donors, included = module.make_donor_indices(data, seed=66)
    assert included.tolist() == [True] * 6 + [False]
    assert donors[-1] == -1
    np.testing.assert_array_equal(data['availability'][donors[included]], data['availability'][included])
    np.testing.assert_array_equal(data['has_history'][donors[included]], data['has_history'][included])
    assert np.all(data['conversation_ids'][donors[included]] != data['conversation_ids'][included])
    changed = {**data, 'labels': data['labels'][::-1], 'history_seen': np.zeros((7, 3))}
    other, coverage = module.make_donor_indices(changed, seed=66)
    np.testing.assert_array_equal(other, donors)
    np.testing.assert_array_equal(coverage, included)


def test_single_conversation_and_empty_donor_population_never_fall_back_to_self():
    module = trace_module()
    for count in (0, 4):
        donors, included = module.make_donor_indices(dict(conversation_ids=np.array(['a'] * count),
                                                         availability=np.ones((count, 3)),
                                                         has_history=np.ones(count, dtype=bool)))
        assert donors.tolist() == [-1] * count
        assert not included.any()


def test_train_multibatch_rng_continues_and_histories_reset_for_each_dialogue():
    module = trace_module()
    model, raw, *_ = fixture()
    second = [*raw[:-1], ['c', 'd']]
    config = SimpleNamespace(dataset='MELD', mosi_task_mode='regression')
    arrays, checks = module.extract_split(model, [raw, second], config, (4, 4, 4), .7, 'train')
    rng = np.random.RandomState(66)
    expected = []
    for batch in (raw, second):
        sides = [random_mask(3, 8, .7, rng=rng).reshape(2, 4, 3) for _ in range(2)]
        expected.append(np.where(batch[6].numpy()[..., None] == 0, sides[0], sides[1])[batch[7].bool().numpy()])
    np.testing.assert_array_equal(arrays['availability'], np.concatenate(expected))
    assert checks['memory_scan_count'] == 2 and checks['utterances'] == 12
    assert arrays['conversation_ids'].tolist() == ['a'] * 4 + ['b'] * 2 + ['c'] * 4 + ['d'] * 2
    assert not arrays['memory'][[0, 4, 6, 10]].any()
    assert not arrays['history_seen'][[0, 4, 6, 10]].any()


def test_future_and_other_dialogue_features_cannot_change_target_prewrite_memory():
    module = trace_module()
    model, raw, saved, *_ = fixture()
    config = SimpleNamespace(dataset='MELD', mosi_task_mode='regression')
    baseline, _ = module.extract_split(model, [raw], config, (4, 4, 4), .7, 'test', saved=saved)
    changed = [value.clone() if torch.is_tensor(value) else value for value in raw]
    for features in changed[:6]:
        features[2:, 0] += 1000  # future relative to a/utterance 1
        features[:, 1] -= 1000  # all positions in a separate conversation
    # Regenerate expected predictions solely to permit saved-mask replay parity;
    # the intervention leaves availability, labels, and trace ordering unchanged.
    tensor, _ = scatter_saved_availability(saved['availability'], 0, changed[7])
    view = replay_view_from_availability(changed, tensor, (4, 4, 4))
    with torch.no_grad():
        logits = model([view['incomplete']], tensor, changed[6], changed[7], [4, 2])[0]
    changed_saved = {**saved, 'predictions': logits.transpose(0, 1)[changed[7].bool()].argmax(-1).numpy()}
    intervention, _ = module.extract_split(model, [changed], config, (4, 4, 4), .7, 'test', saved=changed_saved)
    np.testing.assert_array_equal(intervention['memory'][:2], baseline['memory'][:2])
    np.testing.assert_array_equal(intervention['queries'][:2], baseline['queries'][:2])
    assert not np.allclose(intervention['memory'][3], baseline['memory'][3])
