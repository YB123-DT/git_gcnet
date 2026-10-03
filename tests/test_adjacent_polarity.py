import unittest


class AdjacentPolarityTests(unittest.TestCase):
    def test_pair_scores_and_macro_do_not_pool_samples(self):
        from experiments.osram_context_audit_20261003.adjacent import score,macro
        items=[dict(label=1,local_prediction=-.1,memory_prediction=.1,pattern='AV'),
               dict(label=-1,local_prediction=-.2,memory_prediction=.2,pattern='T')]
        s=score(items)
        self.assertEqual((s['corrections'],s['harms'],s['n']),(1,1,2))
        self.assertAlmostEqual(s['delta'],0.)
        a=dict(seed=66,n=100,corrections=1,harms=0,no_text=0,local_wf1=50.,memory_wf1=60.,delta=10.)
        b=dict(seed=66,n=1,corrections=0,harms=1,no_text=1,local_wf1=50.,memory_wf1=30.,delta=-20.)
        self.assertEqual(macro([a,b])['delta'],-5.)

    def test_no_skipping_neutral_or_crossing_conversation(self):
        from experiments.osram_context_audit_20261003.adjacent import attach
        rows=[dict(seed=66,rate=0.,conversation_id=c,utterance_index=t,label=y,
                   local_prediction=p) for c,t,y,p in
              [('x',0,1.,.2),('x',1,0.,.2),('x',2,-1.,.2),('x',3,1.,-.25),
               ('y',0,1.,.8),('y',1,1.,-.3)]]
        kept,excluded=attach(rows)
        self.assertEqual(len(kept),2)
        self.assertEqual(kept[0]['relation'],'opposite')
        self.assertEqual(kept[0]['boundary'],'near')
        self.assertEqual(kept[1]['relation'],'same')
        self.assertEqual(kept[1]['boundary'],'far')
        self.assertEqual(excluded,dict(first=2,neutral_pair=2))

    def test_missing_immediate_predecessor_fails(self):
        from experiments.osram_context_audit_20261003.adjacent import attach
        with self.assertRaises(AssertionError):
            attach([dict(seed=66,rate=0.,conversation_id='x',utterance_index=2,label=1.,local_prediction=1.)])


if __name__=='__main__':unittest.main()
