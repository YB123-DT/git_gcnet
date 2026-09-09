import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'gcnet_missing_m3/analyze_osram_memory_retention.py'


def record(**updates):
    result = dict(dataset='MOSI', missing_rate=0.5, sample_id='a', time_index=2,
                  target_modality='audio', missing_pattern='100', status='retention',
                  history_distance=1, head=0, err_pre=1, err_decay=3, err_post=2,
                  decay_damage=2, write_damage=-1, max_key_overlap=0.1,
                  mean_key_overlap=0.05)
    result.update(updates)
    return result


class RetentionAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if SCRIPT.exists():
            spec = importlib.util.spec_from_file_location('retention_analysis', SCRIPT)
            cls.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.module)

    def setUp(self):
        self.assertTrue(SCRIPT.exists(), 'analysis implementation is missing')

    def test_query_counts_are_not_head_counts(self):
        rows = [record(), record(head=1), record(sample_id='b', history_distance=9),
                record(sample_id='c', status='NO_HISTORY', history_distance=None)]
        tables = self.module.summarize(rows)
        coverage = tables['coverage'][0]
        self.assertEqual(coverage['missing_count'], 3)
        self.assertEqual(coverage['no_history_count'], 1)
        self.assertAlmostEqual(coverage['no_history_ratio'], 1 / 3)
        distance = tables['distance'][0]
        self.assertEqual(distance['history_query_count'], 2)
        self.assertEqual(distance['distance_mean'], 5)
        self.assertEqual(distance['distance_median'], 5)
        self.assertAlmostEqual(distance['distance_p90'], 8.2)
        self.assertEqual(tables['damage'][0]['head_count'], 2)
        self.assertEqual(tables['damage'][0]['write_damage_mean'], -1)

    def test_missing_metrics_are_not_zero(self):
        tables = self.module.summarize([record(err_pre=None, write_damage=None),
                                        record(head=1, err_pre=4)])
        damage = tables['damage'][0]
        self.assertEqual(damage['err_pre_mean'], 4)
        self.assertEqual(damage['write_damage_mean'], -1)
        self.assertIsNone(damage['cos_pre_mean'])

    def test_bucket_boundaries(self):
        rows = [record(sample_id=str(i), history_distance=d, max_key_overlap=o)
                for i, (d, o) in enumerate(zip([1, 2, 3, 4, 7, 8], [0, .1, .2, .4, .6, 1]))]
        tables = self.module.summarize(rows)
        self.assertEqual([r['distance_bucket'] for r in tables['damage']], ['1', '2-3', '4-7', '8+'])
        self.assertEqual([r['head_count'] for r in tables['overlap']], [1, 1, 1, 1, 2])

    def test_no_history_only_keeps_distance_missing(self):
        tables = self.module.summarize([record(status='NO_HISTORY', history_distance=None)])
        self.assertIsNone(tables['distance'][0]['distance_mean'])
        self.assertEqual(tables['damage'], [])

    def test_formats_and_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for suffix in ['json', 'jsonl', 'csv']:
                path = root / ('input.' + suffix)
                rows = [record()]
                if suffix == 'csv':
                    with path.open('w', newline='') as handle:
                        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(rows)
                else:
                    path.write_text(json.dumps(rows if suffix == 'json' else rows[0]))
                self.assertEqual(self.module.summarize(self.module.load_records(path))['coverage'][0]['missing_count'], 1)
            self.module.write_reports(self.module.summarize(rows), root)
            for name in ['coverage', 'distance', 'damage', 'overlap']:
                self.assertTrue((root / (name + '.csv')).exists())
                self.assertIn('|', (root / (name + '.md')).read_text())
            self.assertIn('future', (root / 'damage.md').read_text())

    def test_cli_and_empty_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'empty.json'
            source.write_text('[]')
            output = root / 'reports'
            self.module.main([str(source), '--output-dir', str(output)])
            self.assertEqual(len(list(output.iterdir())), 8)
            with (output / 'coverage.csv').open() as handle:
                self.assertEqual(list(csv.DictReader(handle)), [])

    def test_conflicting_query_metadata_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            self.module.summarize([record(), record(head=1, history_distance=2)])


if __name__ == '__main__':
    unittest.main()
