import importlib.util
from pathlib import Path
import unittest


class SummaryTests(unittest.TestCase):
    def module(self):
        path = Path(__file__).resolve().parents[1] / 'experiments/osram_nested_diagnostics_20261007/summarize.py'
        self.assertTrue(path.is_file(), 'The grouped diagnostic summary is not implemented')
        spec = importlib.util.spec_from_file_location('nested_diag_summary', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_equal_fold_then_seed(self):
        rows = [{'seed': s, 'fold': f, 'value': s/100+f/1000}
                for s in (66, 67, 68) for f in (1, 2, 3, 4, 5)]
        result = self.module().aggregate(rows, (1, 2, 3, 4, 5))
        self.assertAlmostEqual(result['mean'], .673)
        self.assertAlmostEqual(result['sd'], .01)
        self.assertTrue(result['complete_three_seeds'])

    def test_incomplete_seed_never_promoted(self):
        rows = [{'seed': 66, 'fold': 1, 'value': .8}, {'seed': 67, 'fold': 1, 'value': .7}]
        result = self.module().aggregate(rows, (1,))
        self.assertIsNone(result['mean'])
        self.assertFalse(result['complete_three_seeds'])
        self.assertEqual(result['missing'], [[68, 1]])

    def test_missing_fold_prevents_complete_claim(self):
        rows = [{'seed': s, 'fold': f, 'value': .8} for s in (66, 67, 68) for f in (1, 2)
                if (s, f) != (67, 2)]
        result = self.module().aggregate(rows, (1, 2))
        self.assertIsNone(result['mean'])
        self.assertEqual(result['missing'], [[67, 2]])


if __name__ == '__main__':
    unittest.main()
