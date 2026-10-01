import importlib
import unittest
from pathlib import Path
from unittest.mock import patch


class PairedRunnerTests(unittest.TestCase):
    def test_count_weighted_drop_summary(self):
        run = importlib.import_module('experiments.osram_paired_history_views_20261001.run')
        self.assertTrue(hasattr(run, 'aggregate_diagnostics'))
        stats = run.aggregate_diagnostics([
            dict(anchor_count=3, eligible_anchor_count=2, dropped_observed_count=2,
                 view1_observed_count=10, valid_utterance_count=5, batches_no_negatives=1),
            dict(anchor_count=5, eligible_anchor_count=5, dropped_observed_count=1,
                 view1_observed_count=20, valid_utterance_count=8, batches_no_negatives=0)])
        self.assertEqual(stats['anchor_count'],8)
        self.assertEqual(stats['eligible_anchor_count'],7)
        self.assertAlmostEqual(stats['actual_drop_ratio'],.1)

    def test_only_requested_variants_and_config_delta(self):
        run = importlib.import_module('experiments.osram_paired_history_views_20261001.run')
        original = {'seed': 66, 'epochs': 100, 'batch_size': 32}
        with patch.object(run.common, 'configuration_dict'), patch.object(run, 'read', return_value=original):
            for variant, weight in [('control', 0.), ('contrastive', .1)]:
                actual = run.configuration_dict(variant, Path('/unused'))
                self.assertEqual(actual, dict(original, paired_history_views=True,
                    history_drop_prob=.2, history_contrast_weight=weight,
                    history_contrast_temperature=.1))
            with self.assertRaises(ValueError): run.configuration_dict('search', Path('/unused'))

    def test_gpu_and_smoke_guard(self):
        run = importlib.import_module('experiments.osram_paired_history_views_20261001.run')
        args = run.parser().parse_args(['--preflight'])
        run.validate_args(args)
        args.gpu = '4'
        with self.assertRaises(ValueError): run.validate_args(args)


if __name__ == '__main__': unittest.main()
