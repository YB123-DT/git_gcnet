import importlib
import json
from pathlib import Path
import unittest


class FilterRunnerTests(unittest.TestCase):
    def test_only_filter_flag_changes(self):
        from experiments.osram_gap_increment_filter_20261003.run import filter_config
        path = Path(__file__).resolve().parents[1] / 'experiments/osram_current_history_relation_20261003/reference/config.json'
        base = json.loads(path.read_text())
        result = filter_config(base)
        self.assertTrue(result['osram_gap_increment_filter'])
        self.assertEqual({k: v for k, v in result.items() if k != 'osram_gap_increment_filter'}, base)

    def test_rejects_wrong_seed_or_other_module(self):
        module = importlib.import_module('experiments.osram_gap_increment_filter_20261003.run')
        path = Path(__file__).resolve().parents[1] / 'experiments/osram_current_history_relation_20261003/reference/config.json'
        base = json.loads(path.read_text())
        for changes in ({'seed': 67}, {'osram_relation_block': True}, {'paired_history_views': True}):
            with self.assertRaises(ValueError):
                module.filter_config(dict(base, **changes))

    def test_authorized_seeds_preserve_reference(self):
        from experiments.osram_gap_increment_filter_20261003.run import filter_config
        path = Path(__file__).resolve().parents[1] / 'experiments/osram_current_history_relation_20261003/reference/config.json'
        base = json.loads(path.read_text())
        for seed in (66, 67, 68):
            reference = dict(base, seed=seed)
            result = filter_config(reference, seed=seed)
            self.assertEqual({k: v for k, v in result.items() if k != 'osram_gap_increment_filter'}, reference)
        with self.assertRaises(ValueError):
            filter_config(dict(base, seed=69), seed=69)


if __name__ == '__main__':
    unittest.main()
