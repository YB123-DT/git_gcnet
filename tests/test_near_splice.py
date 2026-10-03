import unittest
import numpy as np


class NearSpliceTests(unittest.TestCase):
    def test_fixed_boundary_and_preserved_far(self):
        from experiments.osram_current_history_relation_20261003.near_splice import splice
        local = np.array([-.25001, -.25, 0, .25, .25001])
        flat = np.arange(5.)
        relation = flat + 10
        result = splice(local, flat, relation)
        np.testing.assert_array_equal(result, [0, 11, 12, 13, 4])
        np.testing.assert_array_equal(flat, np.arange(5.))

    def test_no_label_or_relation_argument(self):
        import inspect
        from experiments.osram_current_history_relation_20261003.near_splice import splice
        self.assertEqual(list(inspect.signature(splice).parameters),
                         ['local', 'flat', 'relation_prediction'])
