import unittest
import numpy as np


class StoredMaskTests(unittest.TestCase):
    def test_reuses_masks_and_rejects_changed_anchor(self):
        from experiments.osram_history_drift_20261002.compare_paired import validate_saved_pair
        a = np.ones((3, 1, 3), dtype=np.float32)
        b = a.copy(); b[0, 0, 1] = 0
        anchor = np.array([[False], [True], [True]])
        validate_saved_pair(a, b, np.ones((1, 3)), anchor)
        with self.assertRaises(AssertionError):
            validate_saved_pair(a, b, np.ones((1, 3)), np.ones((3, 1), dtype=bool))

    def test_rejects_non_text_deletion(self):
        from experiments.osram_history_drift_20261002.compare_paired import validate_saved_pair
        a = np.ones((2, 1, 3), dtype=np.float32)
        b = a.copy(); b[0, 0, 0] = 0
        with self.assertRaises(AssertionError):
            validate_saved_pair(a, b, np.ones((1, 2)), np.array([[False], [True]]))


if __name__ == '__main__':
    unittest.main()
