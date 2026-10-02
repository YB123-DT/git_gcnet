import importlib
import unittest


class DriftAnalysisTests(unittest.TestCase):
    def module(self):
        return importlib.import_module('experiments.osram_history_drift_20261002.analyze')

    def test_zero_cosine_exclusion_and_neutral_flip_filter(self):
        mod=self.module()
        rows=[dict(label=1,pred1=.2,pred2=-.2,delta_local=0,absmax_local=0,
                   base_cos=.2,gap_cos=float('nan'),hidden_cos=.01,prediction_shift=.4,
                   prior_deleted_bit_count=1),
              dict(label=0,pred1=-.1,pred2=.1,delta_local=0,absmax_local=0,
                   base_cos=.4,gap_cos=.5,hidden_cos=.02,prediction_shift=.2,
                   prior_deleted_bit_count=3)]
        s=mod.summarize_rows(rows)
        self.assertEqual(s['anchors'],2)
        self.assertEqual(s['nonzero_labels'],1)
        self.assertEqual(s['gap_cos_n'],1)
        self.assertAlmostEqual(s['gap_cos'],.5)
        self.assertEqual(s['correct_to_wrong'],1)
        self.assertEqual(s['wrong_to_correct'],0)
        self.assertEqual(s['flip_percent'],100.)

    def test_macro_is_rate_balanced(self):
        mod=self.module()
        actual=mod.rate_macro([dict(anchors=10,prediction_shift=1.),dict(anchors=1,prediction_shift=3.)])
        self.assertEqual(actual['anchors'],11)
        self.assertEqual(actual['prediction_shift'],2.)


if __name__=='__main__': unittest.main()
