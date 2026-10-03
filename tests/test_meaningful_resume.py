import importlib
import importlib.util
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


class TrainingResumeTests(unittest.TestCase):
    def state(self, path, identity=None):
        name = 'gcnet_missing_m3.training_resume'
        self.assertIsNotNone(importlib.util.find_spec(name), 'complete-state recovery is not implemented')
        cls = importlib.import_module(name).TrainingState
        model = torch.nn.Linear(2, 1).double()
        model.register_buffer('private_rng', torch.arange(5, dtype=torch.uint8))
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        state = cls(path, identity=identity or {'source': 'abc', 'protocol': 'fixed'},
                    schedule_identity={'kind': 'epoch_stateless', 'seed': 66})
        return state.bind(model, optimizer, scheduler=scheduler), model, optimizer, scheduler

    def step(self, model, optimizer, scheduler):
        optimizer.zero_grad()
        model(torch.randn(3, 2, dtype=torch.double)).square().mean().backward()
        optimizer.step()
        scheduler.step()

    def best(self, state, model, epoch, rate):
        state.save_best(epoch=epoch, rate=rate, checkpoint={
            'model': model.state_dict(), 'epoch': epoch, 'selection_rate': float(rate),
            'selection_score': .8, 'selection_protocol': 'per-rate-test-oracle', 'config': {}})

    def test_uninterrupted_next_step_and_all_rng_match(self):
        with tempfile.TemporaryDirectory() as root:
            state, model, opt, sched = self.state(root)
            self.assertIsNone(state.restore())
            random.seed(7); np.random.seed(7); torch.manual_seed(7)
            self.step(model, opt, sched)
            self.best(state, model, 1, '0.0')
            state.commit_epoch(next_epoch=1, history=[{'epoch': 1}],
                               selection_state={'selected_epoch_by_rate': {'0.0': 1}},
                               schedule_state={'next_epoch': 1})
            expected_draws = (random.random(), float(np.random.rand()), torch.rand(2))
            self.step(model, opt, sched)
            expected = {k: v.clone() for k, v in model.state_dict().items()}
            restored, other, other_opt, other_sched = self.state(root)
            info = restored.restore()
            self.assertEqual(info['next_epoch'], 1)
            self.assertEqual(info['schedule_state'], {'next_epoch': 1})
            self.assertEqual(random.random(), expected_draws[0])
            self.assertEqual(float(np.random.rand()), expected_draws[1])
            self.assertTrue(torch.equal(torch.rand(2), expected_draws[2]))
            self.step(other, other_opt, other_sched)
            for k, value in expected.items():
                self.assertTrue(torch.equal(value, other.state_dict()[k]), k)
            self.assertEqual(other_sched.state_dict(), sched.state_dict())

    def test_partial_best_transaction_restores_committed_versions_and_history(self):
        with tempfile.TemporaryDirectory() as root:
            state, model, opt, sched = self.state(root)
            self.best(state, model, 1, '0.0')
            self.best(state, model, 1, '0.1')
            state.commit_epoch(next_epoch=1, history=[{'epoch': 1}], selection_state={'old': True})
            old = model.weight.detach().clone()
            with torch.no_grad(): model.weight.add_(100)
            self.best(state, model, 2, '0.0')
            Path(root, 'history.json').write_text('[{"epoch":1},{"epoch":2}]')
            restore, _, _, _ = self.state(root)
            info = restore.restore()
            self.assertEqual(info['selection_state'], {'old': True})
            for rate in ('0p0', '0p1'):
                best = torch.load(Path(root, f'best_miss_{rate}.pt'), weights_only=False)
                self.assertEqual(best['epoch'], 1)
                self.assertTrue(torch.equal(best['model']['weight'], old))
            self.assertEqual(json.loads(Path(root, 'history.json').read_text()), [{'epoch': 1}])
            self.assertEqual(len(list(Path(root, 'training_versions').glob('epoch_*_model.pt'))), 2)

    def test_one_immutable_model_per_improving_epoch(self):
        with tempfile.TemporaryDirectory() as root:
            state, model, _, _ = self.state(root)
            self.best(state, model, 1, '0.0')
            self.best(state, model, 1, '0.1')
            self.assertEqual(len(list(Path(root, 'training_versions').glob('epoch_*_model.pt'))), 1)

    def test_identity_and_corrupt_reference_fail_closed(self):
        with tempfile.TemporaryDirectory() as root:
            state, model, _, _ = self.state(root)
            self.best(state, model, 1, '0.0')
            state.commit_epoch(next_epoch=1, history=[{'epoch': 1}], selection_state={})
            wrong, _, _, _ = self.state(root, {'source': 'changed', 'protocol': 'fixed'})
            with self.assertRaisesRegex(ValueError, 'identity'):
                wrong.restore()
            next(Path(root, 'training_versions').glob('epoch_*_model.pt')).write_bytes(b'corrupt')
            new, _, _, _ = self.state(root)
            with self.assertRaisesRegex(ValueError, 'hash|integrity'):
                new.restore()

    def test_best_only_is_not_a_resume(self):
        with tempfile.TemporaryDirectory() as root:
            state, model, _, _ = self.state(root)
            self.best(state, model, 1, '0.0')
            with self.assertRaisesRegex(ValueError, 'complete|last_training'):
                state.restore()


if __name__ == '__main__':
    unittest.main()
