"""Small synthetic checks for alignment and targeted recovery accounting."""
import unittest

import numpy as np

from .analyze import CASES, align, counts, summarize


class EarlyHarmTests(unittest.TestCase):
    def test_polarity_zero_and_neutral_exclusion(self):
        result = counts(np.array([1, -1, 0, 1]), np.array([1, -1, 1, 0]),
                        np.array([0, 1, -1, 1]), np.ones(4, dtype=bool))
        self.assertEqual(result, dict(N=3, corrections=1, harms=2, both_correct=0, both_wrong=0))

    def test_alignment_rejects_different_availability(self):
        a = dict(labels=np.array([1]), availability=np.array([[1, 0, 0]]),
                 conversation_ids=np.array(['x']), utterance_indices=np.array([1]))
        b = {**a, 'availability': np.array([[0, 1, 0]])}
        with self.assertRaises(AssertionError):
            align(a, b)

    def test_zero_based_early_scope_and_overlap(self):
        data = dict(labels=np.ones(5), availability=np.array([[1, 0, 0]] * 5),
                    conversation_ids=np.array(['a', 'a', 'b', 'c', 'd']),
                    utterance_indices=np.array([0, 1, 2, 3, 1]),
                    flat=np.ones(5), full=-np.ones(5))
        for c in CASES:
            data[c] = -np.ones(5)
        data['all_off'] = np.array([1, 1, -1, 1, -1])
        data['Gap-T_off'] = np.array([1, 1, 1, 1, -1])
        result = summarize(data)
        self.assertEqual(result['cohort']['N'], 3)
        self.assertEqual(result['target_records'], 3)
        self.assertEqual(result['cases']['all_off']['target']['restored_positive'], 1)
        self.assertEqual(result['cases']['Gap-T_off']['target']['restored_positive'], 2)
        self.assertEqual(result['target_restored_by_any_off_case'], 2)
        self.assertEqual(result['target_all_off_restored_and_slot_restored']['Gap-T_off'], 1)


if __name__ == '__main__':
    unittest.main()
