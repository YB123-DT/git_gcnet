import importlib
from dataclasses import replace
import random
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch


class RunnerTests(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module('gcnet_missing_m3.train_mc_fwc')
        except ModuleNotFoundError:
            self.fail('Standalone MC-FWC runner is not implemented')

    def test_config_is_plain_validation_cfg84(self):
        m = self.module()
        c = m.baseline_config(66, 'cpu', 150)
        self.assertEqual(c.checkpoint_selection, 'validation')
        self.assertEqual(c.training_objective, 'emotion-only')
        self.assertEqual((c.osram_output_dim, c.osram_num_heads, c.osram_write_step), (1600, 8, .6))
        self.assertFalse(c.paired_history_views)
        self.assertEqual(c.osram_meaningful_block, 'none')

    def test_learning_rate_boundary(self):
        m = self.module()
        self.assertEqual([m.learning_rate(e) for e in (0, 99, 100, 149)], [.001, .001, .0001, .0001])

    def test_selection_single_validation_mean(self):
        m = self.module()
        rows = {str(i / 10): {'weighted_f1': i / 10} for i in range(8)}
        self.assertAlmostEqual(m.selection_score(rows), .35)
        with self.assertRaises(ValueError):
            m.selection_score({'0.0': {'weighted_f1': 1.}})

    def test_resume_restores_rng_optimizer_and_best(self):
        m = self.module()
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        model(torch.ones(1, 2)).sum().backward(); optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'last.pt'
            identity = {'mode': 'E4', 'seed': 66}
            best = {'epoch': 1, 'score': .7, 'model': model.state_dict()}
            m.save_training(path, model, optimizer, identity, 1, [{'epoch': 1}], best)
            expected = (random.random(), np.random.rand(), torch.rand(3))
            model.weight.data.zero_()
            state = m.restore_training(path, model, optimizer, identity)
            actual = (random.random(), np.random.rand(), torch.rand(3))
            self.assertEqual(expected[:2], actual[:2]); self.assertTrue(torch.equal(expected[2], actual[2]))
            self.assertEqual(state['best']['epoch'], 1)
            self.assertEqual(state['next_epoch'], 1)
            self.assertTrue(optimizer.state)
            with self.assertRaises(ValueError):
                m.restore_training(path, model, optimizer, {'mode': 'E2'})

    def test_safe_subset_view_masks_nan_and_keeps_complete(self):
        m = self.module()
        complete = torch.tensor([[[1., float('nan'), 3.]]])
        view = {'complete': complete, 'availability': torch.ones(1, 1, 3), 'umask': torch.ones(1, 1)}
        result = m.replace_availability(view, torch.tensor([[[1., 0., 1.]]]), (1, 1, 1))
        self.assertTrue(torch.equal(result['incomplete'], torch.tensor([[[1., 0., 3.]]])))
        self.assertIs(result['complete'], complete)

    def test_deterministic_seed_independent_of_mode(self):
        m = self.module()
        self.assertEqual(m.batch_seed(66, 2, 1, 'pairs'), m.batch_seed(66, 2, 1, 'pairs'))
        self.assertNotEqual(m.batch_seed(66, 2, 1, 'pairs'), m.batch_seed(66, 3, 1, 'pairs'))

    def test_metrics_neutral_filter_and_negative_recall(self):
        m = self.module()
        result = m.prediction_metrics(np.array([-1., 0., 1.]), np.array([-.1, 100., .2]))
        self.assertEqual(result['accuracy'], 1.)
        self.assertEqual(result['negative_recall'], 1.)
        self.assertEqual(result['positive_ratio'], .5)
        self.assertEqual(result['n_nonzero'], 2)

    def test_e0_matches_original_task_gradient_and_step(self):
        m = self.module()
        from gcnet_missing_m3.mc_fwc import credit_backward
        torch.set_num_threads(1)
        config = replace(m.baseline_config(), latent_dim=8, osram_output_dim=12,
                         osram_num_heads=2, osram_key_dim=3, osram_value_dim=4)
        torch.manual_seed(812)
        first = m.build_model(config, (3, 4, 5)).train()
        second = m.build_model(config, (3, 4, 5)).train()
        second.load_state_dict(first.state_dict(), strict=True)
        x = torch.randn(3, 2, 12)
        view = dict(incomplete=x, availability=torch.ones(3, 2, 3),
                    qmask=torch.zeros(2, 3, dtype=torch.long), umask=torch.ones(2, 3),
                    lengths=[3, 3], labels=torch.randn(2, 3))
        rng = torch.get_rng_state()
        expected = first([x], view['availability'], view['qmask'], view['umask'], [3, 3], predict_missing=False)[0]
        loss = m.old._task_loss('CMUMOSI', expected, view['labels'], view['umask'])
        loss.backward()
        torch.set_rng_state(rng)
        actual, new_loss, _ = credit_backward(second, view, [], mode='E0')
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(new_loss, loss, rtol=0, atol=0)
        for p, q in zip(first.parameters(), second.parameters()):
            if p.grad is None:
                self.assertIsNone(q.grad)
            else:
                torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
        for model in (first, second):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-5).step()
        for p, q in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)

    def test_smoke_never_selects_or_evaluates_validation(self):
        m = self.module()
        class Sampler:
            def set_epoch(self, epoch):
                pass
        class Loader(list):
            sampler = Sampler()
        x = torch.randn(3, 2, 12)
        view = dict(complete=x, incomplete=x, availability=torch.ones(3, 2, 3),
                    qmask=torch.zeros(2, 3, dtype=torch.long), umask=torch.ones(2, 3),
                    lengths=[3, 3], labels=torch.randn(2, 3), conversation_ids=['one', 'two'])
        original_build = m.build_model
        def small_build(config, dimensions):
            return original_build(replace(config, latent_dim=8, osram_output_dim=12,
                osram_num_heads=2, osram_key_dim=3, osram_value_dim=4), dimensions)
        with tempfile.TemporaryDirectory() as directory:
            args = m.parser().parse_args(['--output', directory, '--dataset-root', directory,
                                         '--mode', 'E0', '--device', 'cpu', '--smoke-steps', '3'])
            with patch.object(m, 'make_loaders', return_value=(Loader([[], []]), None, None, (3,4,5), {})), \
                 patch.object(m.old, '_prepare_view', return_value=view), \
                 patch.object(m, 'build_model', side_effect=small_build), \
                 patch.object(m, 'evaluate', side_effect=AssertionError('Smoke must not evaluate')):
                result = m.train(args)
            self.assertEqual(len(result['steps']), 3)
            self.assertFalse((Path(directory)/'best.pt').exists())
            self.assertFalse((Path(directory)/'last.pt').exists())

if __name__ == '__main__':
    unittest.main()
