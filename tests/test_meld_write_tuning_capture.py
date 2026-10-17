import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def capture_class():
    path = Path(__file__).parents[1] / 'experiments/osram_meld_write_tuning_20261017/capture.py'
    assert path.exists(), 'TrainingCapture has not been implemented'
    return importlib.import_module('experiments.osram_meld_write_tuning_20261017.capture').TrainingCapture


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(3, 7)
        self.calls = 0

    def forward(self, inputs, availability, qmask, umask, lengths, predict_missing=False):
        self.calls += 1
        return self.layer(inputs[0]), None, None, None


def fake_trainer():
    trainer = SimpleNamespace()

    def prepare(data, schedule, epoch, dimensions):
        view = dict(data)
        a = torch.from_numpy(np.random.randint(0, 2, (3, 2, 3))).float()
        a[..., 0] = 1
        a[2, 1] = 0
        view['availability'] = a
        return view

    def evaluate(model, loader, schedule, dataset, dimensions, device, collect,
                 mosi_task_mode='regression', task_regression_loss='mse', task_smooth_l1_beta=1.):
        model.eval()
        for raw in loader:
            view = trainer._prepare_view(raw, schedule, 0, dimensions)
            model([view['incomplete']], view['availability'], view['qmask'], view['umask'],
                  view['lengths'], predict_missing=False)
        return trainer.return_value

    trainer._prepare_view = prepare
    trainer.evaluate_rate = evaluate
    trainer.return_value = ({'accuracy': .2}, None)
    return trainer


def batch():
    return dict(incomplete=torch.arange(18.).reshape(3, 2, 3),
                qmask=torch.tensor([[0, 1, 0], [1, 0, 0]]),
                umask=torch.tensor([[1, 1, 1], [1, 1, 0]]),
                labels=torch.tensor([[0, 1, 2], [3, 4, -99]]),
                lengths=[3, 2], conversation_ids=['first', 'second'])


def evaluate(trainer, model, collect=False, positional=False):
    schedule = SimpleNamespace(split='test', requested_missing_rate=.4)
    if positional:
        return trainer.evaluate_rate(model, [batch()], schedule, 'MELD', (1, 1, 1), torch.device('cpu'), collect)
    return trainer.evaluate_rate(model=model, loader=[batch()], schedule=schedule, dataset='MELD',
        dimensions=(1, 1, 1), device=torch.device('cpu'), collect=collect)


def test_capture_does_not_add_forward_random_draws_or_change_returns(tmp_path):
    Capture = capture_class()
    trainer, model = fake_trainer(), TinyModel()
    np.random.seed(51)
    expected = evaluate(trainer, model)
    random_after = np.random.get_state()
    torch_rng = torch.get_rng_state().clone()
    np.random.seed(51)
    with Capture(trainer, tmp_path) as capture:
        capture.bind(model)
        actual = evaluate(trainer, model, positional=True)
    assert actual is expected
    assert model.calls == 2
    now = np.random.get_state()
    assert now[0] == random_after[0] and np.array_equal(now[1], random_after[1]) and now[2:] == random_after[2:]
    assert torch.equal(torch_rng, torch.get_rng_state())
    assert len(capture.initial_state_sha256) == 64
    assert capture.modelparameters == sum(p.numel() for p in model.parameters())
    assert not model._forward_hooks


def test_selected_capture_has_exact_batch_first_ids_and_padding_filter(tmp_path):
    Capture = capture_class()
    trainer, model = fake_trainer(), TinyModel()
    with Capture(trainer, tmp_path) as capture:
        capture.bind(model)
        evaluate(trainer, model)
        ref = capture.selection_reference(1, '0.4')
        capture.write_trace()
    with np.load(tmp_path / ref['path']) as selected:
        assert selected['labels'].tolist() == [0, 1, 2, 3, 4]
        assert selected['conversation_ids'].tolist() == ['first'] * 3 + ['second'] * 2
        assert selected['utterance_indices'].tolist() == [0, 1, 2, 0, 1]
        assert selected['speakers'].tolist() == [0, 1, 0, 1, 0]
        assert selected['availability'].shape == (5, 3)
        assert selected['logits'].shape == (5, 7)
        assert selected['predictions'].tolist() == selected['logits'].argmax(1).tolist()
        assert int(selected['selected_epoch']) == 1
    assert len(ref['sha256']) == len(ref['availability_sha256']) == 64
    assert capture.selected_scores['0.4']['N'] == 5
    assert (tmp_path / 'MASK_TRACE.json').is_file()


def test_final_collect_is_not_a_new_selection_epoch(tmp_path):
    Capture = capture_class()
    trainer, model = fake_trainer(), TinyModel()
    with Capture(trainer, tmp_path) as capture:
        capture.bind(model)
        evaluate(trainer, model)
        before = capture.latest['0.4']
        evaluate(trainer, model, collect=True)
        assert capture.latest['0.4'] is before
        capture.selection_reference(1, '.4')
        with pytest.raises(ValueError, match='epoch'):
            capture.selection_reference(2, '.4')


def test_hooks_restore_after_evaluation_error_and_train_masks_recorded(tmp_path):
    Capture = capture_class()
    trainer, model = fake_trainer(), TinyModel()
    original_prepare, original_evaluate = trainer._prepare_view, trainer.evaluate_rate
    with pytest.raises(RuntimeError, match='deliberate'):
        with Capture(trainer, tmp_path) as capture:
            capture.bind(model)
            trainer._prepare_view(batch(), SimpleNamespace(split='train', requested_missing_rate=.7), 3, (1, 1, 1))
            assert capture.trace['train']['4']['batches'] == 1
            model.forward = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('deliberate'))
            evaluate(trainer, model)
    assert trainer._prepare_view is original_prepare
    assert trainer.evaluate_rate is original_evaluate
    assert not model._forward_hooks


def saved_trace(capture, epochs):
    return dict(train={str(epoch): {'batches': 1} for epoch in range(1, epochs + 1)},
                selection={str(epoch): {f'{i / 10:.1f}': {'batches': 1} for i in range(8)}
                           for epoch in range(1, epochs + 1)},
                final=[{'stale': True}],
                initial_state_sha256=capture.initial_state_sha256, parameters=capture.modelparameters)


def test_resume_discards_uncommitted_trace_and_continues_epoch_counter(tmp_path):
    Capture = capture_class()
    trainer, model = fake_trainer(), TinyModel()
    capture = Capture(trainer, tmp_path).bind(model)
    (tmp_path / 'MASK_TRACE.json').write_text(json.dumps(saved_trace(capture, 3)))
    with capture:
        capture.resume_trace(2)
        assert set(capture.trace['train']) == set(capture.trace['selection']) == {'1', '2'}
        assert capture.trace['final'] == []
        assert capture.latest == {} and capture._digests == {}
        evaluate(trainer, model)
        assert capture.latest['0.4']['epoch'] == 3


@pytest.mark.parametrize('broken', ['missing_train', 'missing_rate', 'initial_hash', 'parameters'])
def test_resume_rejects_incomplete_committed_evidence_or_changed_initialization(tmp_path, broken):
    Capture = capture_class()
    capture = Capture(fake_trainer(), tmp_path).bind(TinyModel())
    trace = saved_trace(capture, 2)
    if broken == 'missing_train':
        del trace['train']['1']
    elif broken == 'missing_rate':
        del trace['selection']['2']['0.7']
    elif broken == 'initial_hash':
        trace['initial_state_sha256'] = 'wrong'
    else:
        trace['parameters'] += 1
    (tmp_path / 'MASK_TRACE.json').write_text(json.dumps(trace))
    with pytest.raises(ValueError):
        capture.resume_trace(2)


def test_resume_zero_without_trace_is_safe_but_nonzero_requires_evidence(tmp_path):
    Capture = capture_class()
    capture = Capture(fake_trainer(), tmp_path).bind(TinyModel())
    capture.resume_trace(0)
    assert capture.trace == dict(train={}, selection={}, final=[])
    assert capture._counts == {f'{i / 10:.1f}': 0 for i in range(8)}
    with pytest.raises(ValueError, match='trace'):
        capture.resume_trace(1)
