import importlib
import unittest


class PairedTextSummaryTest(unittest.TestCase):
    def test_neutral_exclusion_and_flip_directions(self):
        m = importlib.import_module('experiments.osram_history_drift_20261002.compare_summary')
        rows = [dict(label=1, pred1=.1, pred2=-.1),
                dict(label=-1, pred1=.1, pred2=-.1),
                dict(label=0, pred1=.1, pred2=-.1)]
        s = m.score(rows)
        self.assertEqual(s['n'], 2)
        self.assertEqual(s['correct_to_wrong'], 1)
        self.assertEqual(s['wrong_to_correct'], 1)
        self.assertEqual(s['flip_percent'], 100.)
        self.assertAlmostEqual(s['delta_wf1'], 0.)

    def test_unchanged_predictions_have_no_delta(self):
        m = importlib.import_module('experiments.osram_history_drift_20261002.compare_summary')
        s = m.score([dict(label=1, pred1=.3, pred2=.3),
                     dict(label=-1, pred1=-.3, pred2=-.3)])
        self.assertEqual(s['before_wf1'], 100.)
        self.assertEqual(s['after_wf1'], 100.)
        self.assertEqual(s['delta_wf1'], 0.)


if __name__ == '__main__':
    unittest.main()
