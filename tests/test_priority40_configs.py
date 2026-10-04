"""No training: reference preservation, provenance and constructor counts."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.osram_meaningful20_20261003.run import EXPECTED
from experiments.osram_priority40_20261004 import configs


class PriorityConfigTests(unittest.TestCase):
    def baseline(self, seed=66):
        return dict(EXPECTED, seed=seed, osram_meaningful_block='none',
                    user_specific={'arbitrary': [1, 2, 3]}, dropout=.12345)

    def test_exact_forty_configs_and_historical_accounting(self):
        with tempfile.TemporaryDirectory() as temporary:
            reference = Path(temporary) / 'reference.json'
            baseline = self.baseline()
            reference.write_text(json.dumps(baseline))
            destination = Path(temporary) / 'delivery'
            with patch.object(configs, 'parameter_summary', return_value={'mock_count': 7}):
                summary = configs.generate(reference, destination)
            self.assertEqual(len(list(destination.glob('m*.json'))), 40)
            for method in configs.PRIORITY_METHODS:
                emitted = json.loads((destination / (method + '.json')).read_text())
                self.assertEqual(emitted, dict(baseline, osram_meaningful_block=method))
            self.assertEqual(summary['reference']['sha256'], hashlib.sha256(reference.read_bytes()).hexdigest())
            self.assertTrue(summary['code']['code_commit'])
            self.assertFalse(summary['code']['immutable_snapshot'])
            self.assertFalse(summary['training_started'])
            for number in ('m04_', 'm05_', 'm17_', 'm18_', 'm19_', 'm25_', 'm38_', 'm39_', 'm40_'):
                row = next(row for row in summary['methods'] if row['id'].startswith(number))
                self.assertTrue(row['historical_mechanism'])
                self.assertIs(row['count_as_new_method'], False)
            with self.assertRaises(FileExistsError):
                configs.generate(reference, destination)

    def test_reject_bad_reference_before_output_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            reference = Path(temporary) / 'reference.json'
            destination = Path(temporary) / 'delivery'
            for baseline in (dict(self.baseline(), epochs=99),
                             dict(self.baseline(), learning_rate=.01),
                             dict(self.baseline(), seed=12)):
                reference.write_text(json.dumps(baseline))
                with self.assertRaises(ValueError):
                    configs.generate(reference, destination)
                self.assertFalse(destination.exists())

    def test_existing_empty_directory_is_not_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileExistsError):
                configs.generate(Path(temporary) / 'nonexistent.json', temporary)

    def test_actual_lightweight_counts_and_placements(self):
        baseline = self.baseline()
        residual = configs.parameter_summary('m08_mfb', baseline)
        self.assertEqual(residual['new_parameters_excluding_shared'], (64 + 1) * 256 + (256 + 1) * 256)
        self.assertEqual(residual['shared_parameters'], 153600)
        normalization = configs.parameter_summary('m30_dyt', baseline)
        self.assertEqual(normalization['normalization_width'], 4352)
        self.assertEqual(normalization['new_parameters_excluding_shared'], 1)
        adapter = configs.parameter_summary('m11_film', baseline)
        self.assertEqual(adapter['shared_parameters'], 0)
        self.assertEqual(adapter['new_parameters_excluding_shared'], 2049 * 64 + 65 * 512)
        self.assertEqual(configs.placement('m08_mfb'), 'R')
        self.assertEqual(configs.placement('m11_film'), 'I')
        self.assertEqual(configs.placement('m30_dyt'), 'N')


if __name__ == '__main__':
    unittest.main()
