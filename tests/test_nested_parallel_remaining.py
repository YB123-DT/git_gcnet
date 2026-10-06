import importlib
import unittest


class PendingFoldTests(unittest.TestCase):
    def test_only_unclaimed_folds_are_dispatched(self):
        mod = importlib.import_module('experiments.osram_nested_cross_dataset_20261006.parallel_remaining')
        queue = {'tasks': [{'task': {'fold': f}} for f in (1, 2, 3)]}
        self.assertEqual(mod.pending_folds(queue), [4, 5])
        queue['tasks'].append({'task': {'fold': 4}})
        self.assertEqual(mod.pending_folds(queue), [5])
        queue['tasks'].append({'task': {'fold': 5}})
        self.assertEqual(mod.pending_folds(queue), [])

    def test_invalid_fold_record_rejected(self):
        mod = importlib.import_module('experiments.osram_nested_cross_dataset_20261006.parallel_remaining')
        with self.assertRaises(ValueError):
            mod.pending_folds({'tasks': [{'task': {'fold': 6}}]})


if __name__ == '__main__':
    unittest.main()
