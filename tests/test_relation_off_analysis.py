import importlib.util
from pathlib import Path
import unittest


class RelationOffAnalysisTests(unittest.TestCase):
    def module(self):
        path = Path(__file__).resolve().parents[1] / 'experiments/osram_current_history_relation_20261003/residual_off_analyze.py'
        self.assertTrue(path.exists(), 'Residual-off analysis implementation is missing')
        spec = importlib.util.spec_from_file_location('off_analysis', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_difference_identity_and_nonadditive_transition_counts(self):
        mod = self.module()
        result = mod.decompose([1, -1, 0], [-1, -1, -1], [1, -1, 1], [-1, -1, 1])
        self.assertEqual(result['n'], 2)
        self.assertAlmostEqual(result['on_minus_f'], result['off_minus_f'] + result['on_minus_off'])
        self.assertEqual(result['off_minus_f_corrections'], 1)
        self.assertEqual(result['on_minus_off_harms'], 1)
        self.assertEqual(result['on_minus_f_corrections'], 0)
        self.assertEqual(result['on_minus_f_harms'], 0)
        self.assertEqual(result['f_wf1'], result['on_wf1'])

    def test_macro_not_pooled_and_empty_stratum(self):
        mod = self.module()
        small = dict(seed=66, rate=0., **mod.decompose([1], [1], [1], [1]))
        large = dict(seed=66, rate=.1, **mod.decompose([1]*9, [1]*9, [-1]*9, [-1]*9))
        empty = dict(seed=66, rate=.2, **mod.decompose([], [], [], []))
        result = mod.aggregate([small, large, empty])
        self.assertEqual(result['off_wf1'], 50)
        self.assertEqual(result['n'], 10)
        self.assertEqual(result['nonempty_seed_rate_cells'], 2)
        self.assertIsNone(empty['on_wf1'])


if __name__ == '__main__':
    unittest.main()
