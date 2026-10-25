"""Frozen recent NoArea runs preserve the finite MOSI66 protocol."""
from types import SimpleNamespace

import pytest
import torch


def runner():
    from experiments.osram_recent_noarea_serial_20261025 import stage2
    return stage2


def test_scope_exact_original_lr_and_strict_wf1_selection():
    run = runner()
    for mode in ('parallel', 'serial'):
        for rate in ('0.0', '0.3', '0.7'):
            assert run.validate_scope(mode, rate) == rate
    for args in [('wrong', '0.1'), ('serial', '0.8'), ('parallel', 'nan')]:
        with pytest.raises(ValueError):
            run.validate_scope(*args)
    for seed, fold in [(67, 1), (66, 2), (True, 1)]:
        with pytest.raises(ValueError):
            run.validate_scope('serial', '0.1', seed, fold)
    assert [run.learning_rate(e) for e in (0, 99, 100, 149)] == [3e-4, 3e-4, 3e-4/10, 3e-4/10]
    assert run.learning_rate(100) != 3e-5
    with pytest.raises(ValueError):
        run.learning_rate(150)
    assert run.improves({'WF1': 70., 'ACC': 1.}, None)
    assert not run.improves({'WF1': 70., 'ACC': 99.}, {'WF1': 70.})


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.parent = torch.nn.Linear(1, 1).requires_grad_(False).eval()
        self.block = torch.nn.Linear(1, 1, bias=False)
        self.aux = 0.

    def train(self, mode=True):
        super().train(mode)
        self.parent.eval()
        return self

    def forward(self, view):
        prediction = self.block(view['x'])
        return dict(prediction=prediction, aux_loss=prediction.new_tensor(self.aux))


def toy_bundle():
    class Loader(list):
        sampler = SimpleNamespace(set_epoch=lambda epoch: None)
    class Trainer:
        @staticmethod
        def _schedules(cfg, split):
            return {r: r for r in cfg.train_missing_rates}
        @staticmethod
        def _move_batch(raw, device):
            return raw
        @staticmethod
        def _prepare_view(data, schedule, epoch, dimensions):
            return dict(x=torch.ones(2, 1), labels=torch.zeros(2, 1), umask=torch.ones(2),
                        availability=torch.full((2, 3), schedule), conversation_ids=['a', 'b'])
        @staticmethod
        def _task_loss(dataset, prediction, labels, umask, task, loss, beta):
            assert dataset == 'CMUMOSI' and task == 'regression' and loss == 'mse'
            return (prediction-labels).square().mean()
    cfg = SimpleNamespace(dataset='CMUMOSI', train_missing_rates=tuple(i/10 for i in range(8)),
        mosi_task_mode='regression', task_regression_loss='mse', task_smooth_l1_beta=1., gradient_clip_norm=1.)
    return dict(trainer=Trainer, cfg=cfg, train_loader=Loader(range(10)), dimensions=())


def test_optimizer_only_block_and_cyclic_training_original_mse_including_gold0():
    run, model = runner(), ToyModel()
    optimizer = run.build_optimizer(model)
    assert type(optimizer) is torch.optim.Adam
    assert optimizer.param_groups[0]['lr'] == 3e-4
    assert optimizer.param_groups[0]['weight_decay'] == 1e-5
    assert {id(p) for group in optimizer.param_groups for p in group['params']} == {id(p) for p in model.block.parameters()}
    before = model.block.weight.detach().clone()
    row = run.train_epoch(toy_bundle(), model, optimizer, 3, '0.7', False)
    assert row['training_rates'] == [.3, .4, .5, .6, .7, 0., .1, .2, .3, .4]
    assert row['N'] == 20 and row['optimizer_steps'] == 10
    assert row['gradients']['nonzero'] > 0 and row['updated_parameters'] > 0
    assert not torch.equal(before, model.block.weight)
    assert not model.parent.training and all(p.grad is None for p in model.parent.parameters())
    assert len(row['mask_trace_sha256']) == 64
    model.parent.requires_grad_(True)
    with pytest.raises(RuntimeError, match='Frozen parent'):
        run.build_optimizer(model)


def test_smoke_one_batch_per_epoch_and_nonzero_auxiliary_loss_rejected():
    run, model = runner(), ToyModel()
    optimizer = run.build_optimizer(model)
    row = run.train_epoch(toy_bundle(), model, optimizer, 0, '0.1', True)
    assert row['optimizer_steps'] == 1 and row['N'] == 2
    model.aux = 1.
    with pytest.raises(RuntimeError, match='auxiliary'):
        run.train_epoch(toy_bundle(), model, optimizer, 1, '0.1', True)


def test_history_trace_requires_contiguous_epochs_original_samples_and_exact_new_lr():
    run = runner()
    run.validate_history([dict(epoch=1), dict(epoch=2)], 2)
    with pytest.raises(ValueError):
        run.validate_history([dict(epoch=2)], 1)
    original = dict(epoch=101, training_rates=[.4], optimizer_steps=4, N=3,
                    mask_trace_sha256='a', learning_rate=3e-5)
    current = dict(original, learning_rate=3e-4/10)
    run.validate_epoch_trace(current, original)
    with pytest.raises(ValueError):
        run.validate_epoch_trace(dict(current, N=4), original)
    with pytest.raises(ValueError):
        run.validate_epoch_trace(original, original)


def test_resume_restores_only_block_adam_rng_history_and_checks_identity(tmp_path):
    from experiments.osram_tasr_20261018 import runner_helpers as h
    run, model = runner(), ToyModel()
    optimizer = run.build_optimizer(model)
    row = run.train_epoch(toy_bundle(), model, optimizer, 0, '0.1', True)
    identity = dict(stage='frozen', rate='0.1')
    saved = run.checkpoint_payload(model, optimizer, identity, 1, [row], {'epoch': 1})
    assert set(saved) == {'version', 'block', 'optimizer', 'identity', 'epoch', 'history', 'best', 'rng'}
    expected = model.block.weight.detach().clone()
    torch.manual_seed(4)
    saved['rng'] = h.rng_state()
    next_random = torch.rand(3)
    with torch.no_grad():
        model.block.weight.zero_()
    epoch, history, best = run.restore_checkpoint(model, optimizer, saved, identity, 2)
    assert epoch == 1 and history == [row] and best == {'epoch': 1}
    assert torch.equal(model.block.weight, expected)
    assert torch.equal(torch.rand(3), next_random)
    assert optimizer.state
    with pytest.raises(ValueError, match='identity'):
        run.restore_checkpoint(model, optimizer, saved, dict(stage='joint'), 2)
    with pytest.raises(ValueError, match='budget'):
        run.restore_checkpoint(model, optimizer, saved, identity, 0)


def test_cached_evaluation_uses_predict_frame_only_and_original_metrics():
    import numpy as np
    run = runner()
    class Frame:
        valid = torch.ones(2, 1, dtype=torch.bool)
        prediction = torch.tensor([[[-1.]], [[1.]]])
        def to(self, device):
            return self
    class Cached(ToyModel):
        def predict_frame(self, frame):
            return frame.prediction + .1
        def forward(self, view):
            raise AssertionError('Cached evaluation rescanned the parent')
    rows = [dict(frame=Frame(), metadata=dict(labels=np.array([-1., 1.]),
                conversation_ids=np.array(['a', 'a'])), availability=np.ones((2, 3)))]
    score, arrays = run.evaluate(Cached(), rows)
    assert score['ACC'] == score['WF1'] == 100.
    assert arrays['prediction'].shape == arrays['base_prediction'].shape == (2,)
    np.testing.assert_allclose(arrays['prediction'], [-.9, 1.1])


@pytest.mark.parametrize('mode', ['parallel', 'serial'])
def test_real_frozen_wrapper_step_counts_actual_scan_and_keeps_parent_unchanged(mode):
    from experiments.osram_recent_noarea_serial_20261025.model import RecentNoAreaModel
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from tests.test_three_block_joint_model import parent_model, view
    torch.set_num_threads(1)
    torch.manual_seed(44)
    parent = parent_model()
    torch.nn.init.normal_(parent.osram.emotion_adapter[-1].weight, std=.01)
    model = RecentNoAreaModel(parent, mode, 'frozen').eval()
    bundle = toy_bundle()
    bundle['trainer'] = SimpleNamespace(
        _schedules=lambda cfg, split: {r: r for r in cfg.train_missing_rates},
        _move_batch=lambda raw, device: raw,
        _prepare_view=lambda *args: dict(view(2), labels=torch.zeros(2, 3), conversation_ids=['a', 'b']),
        _task_loss=lambda dataset, prediction, labels, umask, *args: (
            prediction.transpose(0, 1).squeeze(-1) - labels).square().mean())
    before = h.backbone_digest(parent)
    run = runner()
    optimizer = run.build_optimizer(model)
    # Match the model tests' Torch1.13 CPU LSTM backward workaround. This
    # changes only the synthetic CPU test, never the protected CUDA protocol.
    with torch.backends.mkldnn.flags(enabled=False):
        row = run.train_epoch(bundle, model, optimizer, 0, '0.3', True)
    assert row['live_scan_calls'] == [1]
    assert '_scan' not in parent.osram.__dict__
    assert row['updated_parameters'] > 0 and row['gradients']['nonzero'] > 0
    assert before == h.backbone_digest(parent)
    assert all(parameter.grad is None for parameter in parent.parameters())


def test_reference_matches_own_rate_mask_parent_and_config_with_json_roundtrip():
    import json
    run = runner()
    parent, config = {'parent_id': 'a', 'config': {'features': ('a', 'v', 't')}}, {'learning_rate': 3e-4}
    identity = dict(mode='signature', dataset='CMUMOSI', seed=66, evaluation_rate='0.3',
        source_commit=run.NUMERICAL_SOURCE, smoke=False, stage2_epochs=150, cyclic_training=True,
        mask_protocol='gcnet-random-mask-v1', effective_stage2_config=config,
        test_mask_sha256='mask', parent=parent)
    reference = json.loads(json.dumps(dict(status='complete', epoch=150, total_parameters=281856,
        parent_id='a', identity=identity)))
    run.validate_reference(reference, parent, config, 'mask', '0.3')
    with pytest.raises(ValueError):
        run.validate_reference(reference, parent, config, 'wrong', '0.3')
    with pytest.raises(ValueError):
        run.validate_reference(reference, dict(parent_id='wrong'), config, 'mask', '0.3')
    with pytest.raises(ValueError):
        run.validate_reference(reference, parent, config, 'mask', '0.7')
