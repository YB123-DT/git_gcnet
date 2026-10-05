from dataclasses import replace
import subprocess
import sys
import types
import unittest

import torch

from gcnet_missing_m3.core20 import METHODS, attach, cagrad_backward, prepare_optimizers, prototype_phase
from gcnet_missing_m3.train_gcnet import TrainConfig, _task_loss, _schedules, train_epoch
from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model


def config(method='none'):
    return TrainConfig(dataset='CMUMOSI', device='cpu', backbone_type='osram',
        training_objective='emotion-only', disable_unused_aux_modules=True,
        osram_bidirectional=False, latent_dim=16, osram_output_dim=32,
        osram_num_heads=2, osram_key_dim=4, osram_value_dim=4,
        dropout=.1, projector_dropout=.1, core20_method=method,
        mosi_task_mode='binary' if method.startswith('C17') else 'regression',
        emotion_loss_mode='pattern-groupdro-author' if method == 'C18' else 'sample-mean')


def batch():
    x = torch.randn(4, 2, 12)
    availability = torch.tensor([[[1., 0, 1], [0, 1, 0]], [[1, 1, 0], [1, 1, 1]],
        [[0, 1, 1], [1, 0, 0]], [[1, 0, 0], [0, 0, 0]]])
    umask = torch.tensor([[1., 1, 1, 1], [1, 1, 1, 0]])
    incomplete = torch.cat([torch.where(availability[..., i:i+1].bool(), block, 0.)
                            for i, block in enumerate(x.split((3, 4, 5), -1))], -1)
    return dict(incomplete=incomplete, complete=x, availability=availability, umask=umask,
                qmask=torch.zeros(2, 4, dtype=torch.long), lengths=[4, 3],
                labels=torch.tensor([[1., -1, .5, -1], [-.5, 1, -1, 0]]))


class Core20IntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_disabled_preserves_state_rng_and_outputs(self):
        cfg = config()
        model = _build_model(cfg, (3, 4, 5)).eval()
        state = {k: v.clone() for k, v in model.state_dict().items()}
        before = torch.get_rng_state().clone()
        attach(model, cfg)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertFalse(hasattr(model, 'core20'))
        for k, v in state.items():
            self.assertTrue(torch.equal(v, model.state_dict()[k]))
        view = batch()
        args = ([view['incomplete']], view['availability'], view['qmask'], view['umask'], view['lengths'])
        torch.manual_seed(19); expected = model(*args)[0]
        torch.manual_seed(19); actual = model(*args)[0]
        self.assertTrue(torch.equal(expected, actual))

    def test_disabled_matches_prechange_commit(self):
        import gcnet_missing_m3.model as current
        originals = {}
        for name in ('osram', 'model'):
            source = subprocess.check_output(['git', 'show', f'073984d:gcnet_missing_m3/{name}.py'], text=True)
            module = types.ModuleType(f'gcnet_missing_m3._core20_prior_{name}')
            module.__package__ = 'gcnet_missing_m3'
            sys.modules[module.__name__] = module
            exec(compile(source, f'prior_{name}.py', 'exec'), module.__dict__)
            originals[name] = module
        originals['model'].OSRAMBackbone = originals['osram'].OSRAMBackbone
        new_class = current.MissingM3GraphModel
        try:
            current.MissingM3GraphModel = originals['model'].MissingM3GraphModel
            expected_model = _build_model(config(), (3, 4, 5))
        finally:
            current.MissingM3GraphModel = new_class
        actual_model = _build_model(config(), (3, 4, 5))
        for k, v in expected_model.state_dict().items():
            self.assertTrue(torch.equal(v, actual_model.state_dict()[k]), k)
        view = batch()
        args = ([view['incomplete']], view['availability'], view['qmask'], view['umask'], view['lengths'])
        for training in (True, False):
            expected_model.train(training); actual_model.train(training)
            torch.manual_seed(17); expected = expected_model(*args)[0]; rng = torch.get_rng_state()
            torch.manual_seed(17); actual = actual_model(*args)[0]
            self.assertTrue(torch.equal(expected, actual))
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_all_methods_reach_real_task_backward(self):
        for method in METHODS:
            with self.subTest(method=method):
                cfg = config(method)
                model = _build_model(cfg, (3, 4, 5))
                if method == 'C17':
                    model.smax_fc = torch.nn.Linear(32, 2)
                before = torch.get_rng_state().clone()
                attach(model, cfg)
                self.assertTrue(torch.equal(before, torch.get_rng_state()), method)
                view = batch()
                logits = model([view['incomplete']], view['availability'], view['qmask'],
                               view['umask'], view['lengths'])[0]
                criterion = lambda output: _task_loss(cfg.dataset, output, view['labels'], view['umask'], cfg.mosi_task_mode)
                loss = model.core20.loss(model, cfg, view, logits, criterion(logits), criterion)
                self.assertTrue(torch.isfinite(loss), method)
                if method == 'C20':
                    cagrad_backward(model.core20.multilevel_losses, model)
                else:
                    loss.backward()
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads), method)
                optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad])
                optimizer.step()
                model.eval()
                with torch.no_grad():
                    pred = model([view['incomplete']], view['availability'], view['qmask'],
                                 view['umask'], view['lengths'])[0]
                self.assertTrue(torch.isfinite(pred).all(), method)

    def test_actual_epoch_loop_and_extra_optimizer(self):
        view = batch()
        blocks = list(view['complete'].split((3, 4, 5), -1))
        raw = blocks + blocks + [view['qmask'], view['umask'], view['labels'], ['train-a', 'train-b']]
        for method in ('C15', 'C17', 'C18', 'C20'):
            with self.subTest(method=method):
                cfg = config(method)
                model = _build_model(cfg, (3, 4, 5))
                if method == 'C17':
                    model.smax_fc = torch.nn.Linear(32, 2)
                attach(model, cfg)
                groups = prepare_optimizers(model, cfg, [{'params': [p for p in model.parameters() if p.requires_grad]}])
                optimizer = torch.optim.Adam(groups, lr=cfg.learning_rate)
                prototype_phase(model, 0)
                weights = torch.ones(7, dtype=torch.double) if method == 'C18' else None
                metrics = train_epoch(model, [raw], optimizer, cfg, _schedules(cfg, 'train'),
                                      0, (3, 4, 5), torch.device('cpu'), weights)
                self.assertEqual(metrics['optimizer_steps'], 1)
                self.assertEqual(metrics['core20']['method'], method)
                if method == 'C15':
                    self.assertTrue(model.core20_critic_optimizer.state)
                if method == 'C18':
                    torch.testing.assert_close(weights.sum(), weights.new_tensor(1.))


if __name__ == '__main__':
    unittest.main()
