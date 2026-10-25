"""Finite joint protocol and durable recovery, without GPU/data dependencies."""
import copy
import hashlib
from types import SimpleNamespace

import pytest
import torch


def runner():
    from experiments.osram_recent_noarea_serial_20261025 import stage1
    return stage1


def config():
    return SimpleNamespace(dataset='CMUMOSI', seed=66, fold=1, epochs=150,
        learning_rate=3e-4, weight_decay=1e-5, batch_size=32, optimizer='adam',
        missing_mask_protocol='gcnet-random-mask-v1', checkpoint_selection='test-oracle-per-rate',
        checkpoint_selection_metric='weighted_f1', train_rate_mode='cyclic',
        train_missing_rates=tuple(i / 10 for i in range(8)), mosi_task_mode='regression',
        task_regression_loss='mse', training_objective='emotion-only', evaluation_protocol='official')


def test_finite_scope_protocol_exact_lr_and_earliest_ties():
    run = runner()
    assert run.MODES == ('parallel', 'serial')
    assert run.epoch_budget(False) == 150 and run.epoch_budget(True) == 3
    assert run.epoch_lr(99) == 3e-4 and run.epoch_lr(100) == 3e-4 / 10
    run.validate_protocol(config())
    cfg = config()
    cfg.train_rate_mode = 'fixed'
    with pytest.raises(ValueError, match='train_rate_mode'):
        run.validate_protocol(cfg)
    previous = {'metrics': {'router': {'WF1': 75., 'ACC': 70.}}}
    assert not run.is_improvement({'router': {'WF1': 75., 'ACC': 99.}}, previous)
    assert run.is_improvement({'router': {'WF1': 75.1, 'ACC': 65.}}, previous)


def test_exact_mask_hash_and_order_cannot_be_substituted():
    run = runner()
    masks = {'test': {r: [dict(availability=torch.tensor([[[1., 0., 1.]]]))] for r in run.RATES}}
    expected = {r: hashlib.sha256(rows[0]['availability'].numpy().tobytes()).hexdigest()
                for r, rows in masks['test'].items()}
    assert run.validate_masks(masks, expected) == expected
    masks['test']['0.3'][0]['availability'][0, 0, 0] = 0
    with pytest.raises(ValueError, match='mask'):
        run.validate_masks(masks, expected)


def test_numerical_source_uses_its_own_root_and_detects_drift(tmp_path):
    run = runner()
    numeric, overlay = tmp_path / 'numeric', tmp_path / 'overlay'
    numeric.mkdir()
    overlay.mkdir()
    (numeric / 'core.py').write_text('original')
    (overlay / 'core.py').write_text('overlay')
    digest = run.source_digest({'numerical': (numeric, ['core.py']), 'overlay': (overlay, ['core.py'])})
    assert digest['numerical'] != digest['overlay']
    run.verify_source_digest(digest, numeric, overlay)
    (numeric / 'core.py').write_text('changed')
    with pytest.raises(ValueError, match='Source'):
        run.verify_source_digest(digest, numeric, overlay)


def test_full_resume_restores_optimizer_global_rng_and_noise_rng():
    run = runner()
    from experiments.osram_tasr_20261018 import runner_helpers as h
    torch.manual_seed(12)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    model(torch.ones(1, 2)).square().mean().backward()
    optimizer.step()
    noise = torch.Generator().manual_seed(6601)
    torch.rand(1, generator=noise)
    identity = {'stage': 'joint', 'smoke': True}
    payload = copy.deepcopy(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
        rng=h.rng_state(), noise_rng=noise.get_state(), identity=identity, epoch=1,
        history=[{'epoch': 1}], best_by_rate={'sentinel': 1}))
    expected_model = copy.deepcopy(model.state_dict())
    expected_draw, expected_noise = torch.rand(3), torch.rand(3, generator=noise)
    with torch.no_grad():
        model.weight.add_(5)
    start, history, best = run.restore_training(model, optimizer, noise, payload, identity)
    assert (start, history, best) == (1, [{'epoch': 1}], {'sentinel': 1})
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in expected_model.items())
    assert torch.equal(torch.rand(3), expected_draw)
    assert torch.equal(torch.rand(3, generator=noise), expected_noise)
    assert all('exp_avg' in state for state in optimizer.state.values())
    with pytest.raises(ValueError, match='identity'):
        run.restore_training(model, optimizer, noise, payload, {'stage': 'frozen'})
    incomplete = dict(payload)
    incomplete.pop('noise_rng')
    with pytest.raises(ValueError, match='checkpoint'):
        run.restore_training(model, optimizer, noise, incomplete, identity)


def test_actual_memory_scan_probe_counts_calls_and_removes_temporary_hooks():
    class Memory(torch.nn.Module):
        def _scan(self, value):
            return value + 1

    class Model(torch.nn.Module):
        def __init__(self, count):
            super().__init__()
            self.parent = torch.nn.Module()
            self.parent.osram = Memory()
            self.count = count

        def forward(self, value):
            for _ in range(self.count):
                value = self.parent.osram._scan(value)
            return value

    run = runner()
    model = Model(1)
    with run.memory_scan_probe(model) as probe:
        assert model(torch.tensor(0)).item() == 1
        assert model(torch.tensor(3)).item() == 4
    assert probe == {'forwards': 2, 'scans': 2}
    assert '_scan' not in model.parent.osram.__dict__
    assert not model._forward_hooks and not model._forward_pre_hooks
    model.count = 2
    with pytest.raises(RuntimeError, match='exactly one'):
        with run.memory_scan_probe(model):
            model(torch.tensor(0))
    assert '_scan' not in model.parent.osram.__dict__
    assert not model._forward_hooks and not model._forward_pre_hooks


def test_fresh_zero_flat_serial_opens_task_gradients_after_first_update():
    from experiments.osram_recent_noarea_serial_20261025.model import RecentNoAreaModel
    from tests.test_three_block_joint_model import parent_model, view
    torch.set_num_threads(1)
    torch.manual_seed(66)
    model = RecentNoAreaModel(parent_model(), mode='serial', stage='joint').train()
    assert not model.parent.osram.emotion_adapter[-1].weight.any()
    assert all(not decoder.weight.any() for decoder in model.block.decoders)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    data, history = view(2), []
    run = runner()
    before = [decoder.weight.detach().clone() for decoder in model.block.decoders]
    with torch.backends.mkldnn.flags(enabled=False):
        for epoch in range(3):
            optimizer.zero_grad(set_to_none=True)
            model(data)['prediction'].square().mean().backward()
            grads = [parameter.grad for parameter in model.block.parameters()]
            assert all(grad is not None and torch.isfinite(grad).all() for grad in grads)
            row = dict(epoch=epoch + 1, gradients={'block': dict(connected=len(grads),
                nonzero=sum(int(grad.ne(0).any()) for grad in grads))})
            run.validate_block_connection(row)
            history.append(row)
            optimizer.step()
    assert history[0]['gradients']['block']['nonzero'] == 0
    assert all(row['gradients']['block']['nonzero'] > 0 for row in history[1:])
    run.validate_block_progress(history)
    assert any(not torch.equal(previous, decoder.weight) for previous, decoder in zip(before, model.block.decoders))
    with pytest.raises(RuntimeError, match='nonzero'):
        run.validate_block_progress(history[:1])
