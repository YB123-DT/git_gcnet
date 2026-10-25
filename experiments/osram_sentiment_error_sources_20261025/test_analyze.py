import unittest

import numpy as np

try:
    from .analyze import strict_history, tally, split_rate
except ModuleNotFoundError:
    strict_history = tally = split_rate = None


class ErrorSourceTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(strict_history, 'History/error accounting not implemented')

    def test_strict_past_and_conversation_isolation_with_unsorted_rows(self):
        ids = np.array(['a', 'b', 'a', 'a'])
        indices = np.array([1, 0, 0, 2])
        av = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0]])
        seen = strict_history(ids, indices, av)
        np.testing.assert_array_equal(seen, [[1, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 1]])

    def test_missing_indices_and_duplicates_are_rejected(self):
        for indices in ([0, 2], [0, 0]):
            with self.assertRaises(ValueError):
                strict_history(np.array(['a', 'a']), np.array(indices), np.ones((2, 3)))

    def test_zero_label_exclusion_and_zero_prediction_is_negative(self):
        row = tally(np.array([1, -1, 0, 1]), np.array([0, 0, 1, 2]),
                    np.ones(4, dtype=bool), np.array([0, 1, 2, 3]))
        self.assertEqual((row['N'], row['errors'], row['positive_to_negative'], row['negative_to_positive']),
                         (3, 1, 1, 0))

    def test_neutral_history_still_counts_and_partitions_reconcile(self):
        # Text at a neutral first turn must count as past observed Text.
        data = dict(labels=np.array([0., 1., -1.]), prediction=np.array([0., -1., -1.]),
                    availability=np.array([[0, 1, 0], [1, 0, 0], [1, 0, 0]]),
                    conversation_ids=np.array(['a', 'a', 'b']), utterance_indices=np.array([0, 1, 0]))
        result = split_rate(data)
        self.assertEqual(result['patterns']['A']['N'], 2)
        self.assertEqual(result['patterns']['A']['errors'], 1)
        self.assertEqual(result['missing_modalities']['T']['seen']['errors'], 1)
        self.assertEqual(result['missing_modalities']['T']['never']['errors'], 0)
        self.assertEqual(sum(row['N'] for row in result['joint']), 2)
        self.assertEqual(sum(row['errors'] for row in result['joint']), 1)


if __name__ == '__main__':
    unittest.main()
