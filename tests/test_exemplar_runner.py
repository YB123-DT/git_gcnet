"""Bounded experiment contract; no dataset/GPU needed for this layer."""
import importlib
import types
import unittest
from pathlib import Path


class ExemplarRunnerTests(unittest.TestCase):
    def worker(self):
        return importlib.import_module('experiments.osram_exemplar_mosi_20261023.worker')

    def test_budget_is_only_two_seed66_mosi_arms(self):
        run = importlib.import_module('experiments.osram_exemplar_mosi_20261023.run')
        jobs = run.approved_jobs()
        self.assertEqual([r['variant'] for r in jobs], ['feature-only', 'label-aware'])
        self.assertTrue(all(r['seed'] == 66 and r['dataset'] == 'CMUMOSI' for r in jobs))

    def test_schedule_keeps_original_two_stage_rates(self):
        worker = self.worker()
        optimizer = types.SimpleNamespace(param_groups=[{'name': 'backbone'}])
        for epoch, expected in [(0, 3e-4), (99, 3e-4), (100, 3e-5), (149, 3e-5)]:
            worker.set_learning_rates(optimizer, epoch)
            self.assertEqual(optimizer.param_groups[0]['lr'], expected)
        with self.assertRaises(ValueError):
            worker.set_learning_rates(optimizer, 150)

    def test_reference_parity_does_not_require_equal_task_losses(self):
        worker = self.worker()
        row = dict(epoch=1, mask_trace_sha256='abc', optimizer_steps=2,
                   learning_rates=[3e-4], task_MSE=12.)
        worker.check_training_parity(row, dict(row, task_MSE=.2))
        with self.assertRaises(ValueError):
            worker.check_training_parity(row, dict(row, mask_trace_sha256='different'))

    def test_config_normalization_only_allows_new_disabled_defaults(self):
        worker = self.worker()
        worker.validate_reference_config({'x': 1, 'new': False}, {'x': 1}, {'new': False})
        with self.assertRaises(ValueError):
            worker.validate_reference_config({'x': 2}, {'x': 1}, {})
        with self.assertRaises(ValueError):
            worker.validate_reference_config({'x': 1, 'new': True}, {'x': 1}, {'new': False})

    def test_training_command_cannot_expand_scope(self):
        run = importlib.import_module('experiments.osram_exemplar_mosi_20261023.run')
        args = types.SimpleNamespace(output=Path('/tmp/example'), source_commit='abc')
        command = run.worker_command(args, run.approved_jobs()[0], 'train', 'GPU-test')
        self.assertIn('experiments.osram_exemplar_mosi_20261023.worker', command)
        self.assertIn('feature-only', command)
        with self.assertRaises(ValueError):
            run.worker_command(args, dict(run.approved_jobs()[0], seed=67), 'train', 'GPU-test')

    def test_completion_requires_artifacts_not_just_status(self):
        run = importlib.import_module('experiments.osram_exemplar_mosi_20261023.run')
        job = dict(run.approved_jobs()[0], output='/tmp/nonexistent_exemplar')
        self.assertFalse(run.completion_matches(job, dict(status='complete', phase='train',
            seed=66, dataset='CMUMOSI', variant='feature-only', epoch=150), 'train'))


if __name__ == '__main__':
    unittest.main()
