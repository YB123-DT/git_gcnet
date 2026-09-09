import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('streak',Path(__file__).resolve().parents[1]/'experiments/osram_retention_20260909/analyze_missing_streaks.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class StreakTest(unittest.TestCase):
    def records(self):
        rows=[]
        for head in (0,1):
            for t,pre,decay,post in [(1,.2,.3,.6),(2,.6,.7,.5)]:
                rows.append(dict(status='retention',sample_id='c',target_modality='audio',head=head,
                    time_index=t,history_distance=t,err_pre=pre,err_decay=decay,err_post=post,
                    decay_damage=decay-pre,write_damage=post-decay,max_key_overlap=.2))
        return rows

    def test_telescoping_signed_and_head_deduplication(self):
        heads,gaps,_=m.reconstruct(self.records(),dict(dataset='toy',seed=66,rate=.7))
        self.assertEqual(len(heads),2);self.assertEqual(len(gaps),1)
        g=gaps[0];self.assertEqual(g['gap_length'],2)
        self.assertAlmostEqual(g['cumulative_write'],.1)
        self.assertAlmostEqual(g['cumulative_decay'],.2)
        self.assertAlmostEqual(g['error_change'],.3)
        self.assertLess(g['closure_error'],1e-10)

    def test_missing_step_rejected(self):
        with self.assertRaises(ValueError):m.reconstruct(self.records()[1:],dict(dataset='toy',seed=66,rate=.7))

    def test_no_history_excluded(self):
        rows=self.records()+[dict(status='NO_HISTORY',sample_id='c',target_modality='visual',time_index=0)]
        self.assertEqual(m.reconstruct(rows,dict(dataset='toy',seed=66,rate=.7))[2],1)

    def test_buckets_and_constant_correlation(self):
        self.assertEqual([m.bucket(x) for x in (1,2,3,4,7,8)],['1','2-3','2-3','4-7','4-7','8+'])
        self.assertIsNone(m.spearman([1,1,1],[2,3,4]))


if __name__=='__main__':unittest.main()
