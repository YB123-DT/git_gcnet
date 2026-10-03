import importlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class DecisionRunnerTests(unittest.TestCase):
    def module(self):
        self.assertTrue((ROOT / 'experiments/osram_decision_correction_20261003/run.py').is_file())
        return importlib.import_module('experiments.osram_decision_correction_20261003.run')

    def baseline(self):
        return json.loads((ROOT / 'experiments/osram_current_history_relation_20261003/reference/config.json').read_text())

    def test_only_new_flag_for_authorized_seeds(self):
        module = self.module()
        for seed in (66, 67, 68):
            base = dict(self.baseline(), seed=seed)
            config = module.decision_config(base, seed)
            self.assertTrue(config.pop('osram_decision_correction'))
            self.assertEqual(config, base)

    def test_rejects_incompatible_reference(self):
        module = self.module()
        for changes in ({'seed': 67}, {'epochs': 50}, {'osram_gap_increment_filter': True},
                        {'osram_decision_correction': True}, {'osram_relation_block': True},
                        {'paired_history_views': True}, {'train_rate_mode': 'conversation-mixed'},
                        {'completion_path': 'frozen-dual-projector'}):
            with self.assertRaises(ValueError):
                module.decision_config(dict(self.baseline(), **changes), 66)
        with self.assertRaises(ValueError):
            module.decision_config(dict(self.baseline(), seed=69), 69)

    def test_gpu_whitelist_and_identity(self):
        module = self.module()
        self.assertNotIn('4', module.UUIDS)
        with patch('subprocess.check_output', return_value=module.UUIDS['0']+', 28000\n'):
            self.assertEqual(module.gpu_check('0'), 28000)
        with patch('subprocess.check_output', return_value='wrong-uuid, 28000\n'):
            with self.assertRaises(RuntimeError):
                module.gpu_check('0')


if __name__ == '__main__':
    unittest.main()
