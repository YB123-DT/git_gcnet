"""CPU/stdlib checks for the locked causal-readout experiment protocol."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    path = ROOT / 'experiments/osram_causal_readout_20260910' / f'{name}.py'
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(f'causal_{name}', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def config():
    return dict(dataset='CMUMOSI', seed=66, backbone_type='osram', fusion_type='mean',
                osram_bidirectional=False, osram_forward_slot_reuse=False,
                osram_write_step=.6, osram_num_heads=8, osram_key_dim=32,
                osram_value_dim=32, osram_output_dim=700, osram_ablation='full',
                osram_predictor_mode='structured', osram_query_availability=True,
                train_rate_mode='cyclic', epochs=100, checkpoint_selection='test-oracle',
                initial_backbone_checkpoint=None, training_objective='joint',
                evaluate_test=True)


def history():
    return [dict(epoch=e, test_oracle={str(r / 10): {'weighted_f1': .5}
                                     for r in range(8)}) for e in range(1, 101)]


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.runner = module('run')
        self.assertIsNotNone(self.runner, 'runner implementation missing')

    def test_clone_changes_only_emotion_flag(self):
        old = config()
        for variant in ('local-only', 'local-base', 'local-gap'):
            actual = self.runner.training_settings(old, variant, 66)
            self.assertEqual(actual, dict(old, osram_emotion_ablation=variant))
        self.assertNotIn('osram_emotion_ablation', old)

    def test_rejects_wrong_protocol(self):
        for key, bad in dict(seed=67, osram_write_step=1., osram_num_heads=4,
                             osram_output_dim=500, osram_bidirectional=True,
                             epochs=99, osram_ablation='no-gap', completion_path='pre_osram_b2',
                             initial_backbone_checkpoint='best.pt', b2_base_checkpoint='x',
                             osram_emotion_ablation='local-only').items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.runner.training_settings(dict(config(), **{key: bad}), 'local-only', 66)


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.summary = module('summarize')
        self.assertIsNotNone(self.summary, 'summary implementation missing')

    def test_argmax_is_independent_per_rate_and_ties_choose_earliest(self):
        rows = history()
        rows[3]['test_oracle']['0.0']['weighted_f1'] = .9
        rows[8]['test_oracle']['0.0']['weighted_f1'] = .9
        rows[19]['test_oracle']['0.1']['weighted_f1'] = .8
        result = self.summary.extract_best(rows)
        self.assertEqual(result['0.0'], {'epoch': 4, 'weighted_f1': .9})
        self.assertEqual(result['0.1'], {'epoch': 20, 'weighted_f1': .8})
        self.assertEqual(result['0.7']['epoch'], 1)

    def test_incomplete_nonfinite_and_missing_rates_rejected(self):
        for rows in (history()[:-1], history() + history()[:1]):
            with self.assertRaises(ValueError):
                self.summary.extract_best(rows)
        for value in (float('nan'), float('inf')):
            rows = history()
            rows[0]['test_oracle']['0.0']['weighted_f1'] = value
            with self.assertRaises(ValueError):
                self.summary.extract_best(rows)
        rows = history()
        del rows[0]['test_oracle']['0.2']
        with self.assertRaises(ValueError):
            self.summary.extract_best(rows)

    def test_pending_has_no_aggregate_and_mask_mismatch_is_invalid(self):
        with tempfile.TemporaryDirectory() as temp:
            root, full = Path(temp) / 'new', Path(temp) / 'full'
            for variant, base in [('full', full), ('local-only', root / 'mosi/local-only')]:
                directory = base / 'seed_66'
                directory.mkdir(parents=True)
                cfg = config()
                if variant != 'full':
                    cfg['osram_emotion_ablation'] = variant
                    (directory / 'PROVENANCE.json').write_text(json.dumps({'status': 'complete'}))
                for name, value in [('config', cfg), ('history', history()),
                                    ('metrics', {'mask_sha256': {str(r / 10): 'a' * 64 for r in range(8)}})]:
                    (directory / f'{name}.json').write_text(json.dumps(value))
            report = self.summary.collect(root, full)
            self.assertEqual(report['status'], 'pending')
            self.assertEqual(report['aggregates'], [])
            path = root / 'mosi/local-only/seed_66/metrics.json'
            metrics = json.loads(path.read_text())
            metrics['mask_sha256']['0.0'] = 'b' * 64
            path.write_text(json.dumps(metrics))
            report = self.summary.collect(root, full)
            self.assertTrue(any('mask' in x['reason'] for x in report['invalid']))

    def test_complete_report_has_all_seeds_deltas_and_selected_epochs(self):
        with tempfile.TemporaryDirectory() as temp:
            root, full = Path(temp) / 'new', Path(temp) / 'full'
            for seed in range(66, 71):
                for variant in ('full', 'local-only', 'local-base', 'local-gap'):
                    base = full if variant == 'full' else root / 'mosi' / variant
                    directory = base / f'seed_{seed}'
                    directory.mkdir(parents=True)
                    cfg = dict(config(), seed=seed)
                    rows = history()
                    if variant != 'full':
                        cfg.update(osram_emotion_ablation=variant, completion_path='none',
                                   b2_base_checkpoint=None, b2_pretrain_checkpoint=None)
                        rows[41]['test_oracle']['0.0']['weighted_f1'] = .6
                        (directory / 'PROVENANCE.json').write_text(json.dumps({'status': 'complete'}))
                    for name, value in [('config', cfg), ('history', rows),
                                        ('metrics', {'mask_sha256': {str(r / 10): 'a' * 64 for r in range(8)}})]:
                        (directory / f'{name}.json').write_text(json.dumps(value))
            report = self.summary.collect(root, full)
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(len(report['rows']), 160)
            self.assertEqual(len(report['aggregates']), 32)
            row = next(row for row in report['aggregates'] if row['variant'] == 'local-only' and row['rate'] == '0.0')
            self.assertEqual(row['n'], 5)
            self.assertAlmostEqual(row['delta_full'], .1)
            overall = next(row for row in report['overall'] if row['variant'] == 'local-only' and row['rates'] == 'all8')
            self.assertAlmostEqual(overall['delta_full'], .1 / 8)
            self.assertEqual(overall['positive_seed_count'], 5)
            high = next(row for row in report['overall'] if row['variant'] == 'local-only' and row['rates'] == 'high')
            self.assertEqual(high['positive_seed_count'], 0)
            self.summary.write_report(report, root / 'summary')
            text = (root / 'summary/summary.md').read_text()
            self.assertIn('INTERNAL DIAGNOSTIC ONLY / NOT A FORMAL PAPER RESULT', text)
            self.assertNotIn(b'\r\n', (root / 'summary/per_seed_rate.csv').read_bytes())
            self.assertIn('local-only | 66 | 0.0 | 42', text)
            self.assertEqual(len((root / 'summary/per_seed_rate.csv').read_text().splitlines()), 161)


if __name__ == '__main__':
    unittest.main()
