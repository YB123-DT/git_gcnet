import importlib.util
from pathlib import Path
import unittest


class DispatchScopeTests(unittest.TestCase):
    def module(self):
        path = Path(__file__).resolve().parents[1] / 'experiments/osram_nested_diagnostics_20261007/dispatch.py'
        self.assertTrue(path.is_file(), 'The bounded diagnostics dispatcher is not implemented')
        spec = importlib.util.spec_from_file_location('nested_diag_dispatch', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_exact_scope(self):
        rows = self.module().tasks()
        self.assertEqual(len(rows), 36)
        self.assertEqual(len({(r['dataset'], r['seed'], r['fold']) for r in rows}), 36)
        for dataset, expected in [('CMUMOSI', 3), ('CMUMOSEI', 3), ('IEMOCAPFour', 15), ('IEMOCAPSix', 15)]:
            self.assertEqual(sum(r['dataset'] == dataset for r in rows), expected)

    def test_only_fixed_old_nested(self):
        for row in self.module().tasks():
            self.assertIn(row['seed'], (66, 67, 68))
            if row['dataset'] == 'CMUMOSI':
                self.assertIn('nested_gnn_rooted_evidence', row['source'])
            self.assertNotIn('rootaware', row['source'])
            self.assertNotIn('local8', row['source'])

    def test_bad_gpu_excluded(self):
        module = self.module()
        self.assertNotIn(4, module.GPUS)
        self.assertEqual(module.GPUS, (6,))

    def test_regression_sources_are_not_baselines(self):
        rows = self.module().tasks()
        seed66 = next(r for r in rows if r['dataset'] == 'CMUMOSI' and r['seed'] == 66)
        self.assertIn('attempt2/runs/nested_gnn_rooted_evidence/seed_66', seed66['source'])
        mosei = next(r for r in rows if r['dataset'] == 'CMUMOSEI')
        self.assertIn('osram_nested_cross_dataset_20261006', mosei['source'])


if __name__ == '__main__':
    unittest.main()
