import tempfile
import unittest
from pathlib import Path

from experiments.osram_cfg84_memory_shift_residual_20260928 import run


class RunnerTest(unittest.TestCase):
    def reference(self, root):
        config = dict(seed=66, dataset='CMUMOSI', epochs=100,
            training_objective='emotion-only', completion_path='none', classification_completion=False,
            train_rate_mode='cyclic', train_missing_rates=[i / 10 for i in range(8)],
            checkpoint_selection='test-oracle-per-rate', osram_bidirectional=False,
            osram_write_step=.6, osram_readout_fusion='flat', osram_ablation='full',
            osram_emotion_ablation='full', osram_gap_read='residual', osram_output_dim=1600,
            osram_num_heads=8, osram_key_dim=64, osram_value_dim=64)
        run.write(root / 'seed_66/config.json', config)
        return config

    def test_only_readout_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = self.reference(root)
            new = run.configuration_dict(66, root)
            self.assertEqual(new, dict(old, osram_readout_fusion='memory-shift-residual'))

    def test_rejects_other_query(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = self.reference(root)
            run.write(root / 'seed_66/config.json', dict(old, osram_history_query_adapter=True))
            with self.assertRaises(ValueError):
                run.configuration_dict(66, root)

    def test_rejects_protocol_drift(self):
        changes = [
            {'train_rate_mode': 'conversation-mixed'},
            {'completion_path': 'jepa'},
            {'classification_completion': True},
            {'completion_write_to_memory': True},
            {'osram_write_step': .7},
            {'osram_bidirectional': True},
            {'osram_output_dim': 800},
            {'training_objective': 'joint'},
            {'checkpoint_selection': 'validation'},
            {'epochs': 1},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = self.reference(root)
            for change in changes:
                with self.subTest(change=change):
                    run.write(root / 'seed_66/config.json', dict(old, **change))
                    with self.assertRaises(ValueError):
                        run.configuration_dict(66, root)

    def test_gpu_zero_only_and_smoke_separation(self):
        args = run.parser().parse_args(['--preflight'])
        run.validate_args(args)
        args.gpus = ['1']
        with self.assertRaises(ValueError):
            run.validate_args(args)
        args.gpus = ['0']
        args.smoke = True
        with self.assertRaises(ValueError):
            run.validate_args(args)
        args.output_root = args.output_root.parent / 'smoke'
        run.validate_args(args)

    def test_aggregate_rates(self):
        metrics = {'test': {str(i / 10): {'weighted_f1': i / 10} for i in range(8)}}
        self.assertAlmostEqual(run.rate_scores(metrics)['mean_8rate'], 35.)
        self.assertAlmostEqual(run.rate_scores(metrics)['high_missing'], 60.)

    def test_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run.write(root / 'status.json', {})
            with self.assertRaises(FileExistsError):
                run.assert_fresh(root)


if __name__ == '__main__':
    unittest.main()

