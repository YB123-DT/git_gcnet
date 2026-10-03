import unittest
import numpy as np
import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch


class RelationAuditTests(unittest.TestCase):
    def test_explicit_ids_align_shuffled_predictions(self):
        from experiments.osram_current_history_relation_20261003.analyze import align_predictions
        baseline = dict(labels=np.array([1., -1.]), availability=np.array([[1,0,0],[0,1,0]]))
        new = dict(labels=np.array([-1.,1.]), availability=baseline['availability'][::-1],
                   predictions=np.array([-.2,.3]), utterance_ids=np.array(['b','a']))
        np.testing.assert_array_equal(align_predictions(new, baseline, ['a','b']), [.3,-.2])
        del new['utterance_ids']
        with self.assertRaises(ValueError):
            align_predictions(new, baseline, ['a','b'])

    def test_ids_duplicate_or_mask_change_rejected(self):
        from experiments.osram_current_history_relation_20261003.analyze import align_predictions
        base=dict(labels=np.array([1.,1.]),availability=np.ones((2,3)))
        new=dict(base,predictions=np.zeros(2),utterance_ids=np.array(['a','a']))
        with self.assertRaises(ValueError):align_predictions(new,base,['a','b'])
        new.pop('utterance_ids');new['availability']=np.zeros((2,3))
        with self.assertRaises(ValueError):align_predictions(new,base,['a','b'])

    def test_counts_neutral_exclusion_and_rate_macro(self):
        from experiments.osram_current_history_relation_20261003.analyze import score, macro
        result=score([1.,-1.,0.],[-1.,-1.,1.],[1.,1.,-1.],[.1,-.1,0.])
        self.assertEqual((result['n'],result['corrections'],result['harms']),(2,1,1))
        a=dict(result,seed=66,baseline_wf1=60.,relation_wf1=80.,local_wf1=40.,delta=20.,n=100)
        b=dict(result,seed=66,baseline_wf1=60.,relation_wf1=40.,local_wf1=40.,delta=-20.,n=1)
        self.assertEqual(macro([a,b])['delta'],0.)

    def test_full_offline_identity_comparison(self):
        from experiments.osram_current_history_relation_20261003.analyze import ROOT, RATES, main, score
        source=ROOT/'experiments/osram_cfg84_history_scale_20260929/results'
        with tempfile.TemporaryDirectory() as temporary:
            run=Path(temporary)/'run';run.mkdir()
            output=Path(temporary)/'analysis'
            metrics={'test':{}}
            for rate in RATES:
                path=source/f'seed_66_miss_{rate:.1f}_alpha_1.0.npz'
                shutil.copyfile(path,run/f"predictions_miss_{str(rate).replace('.', 'p')}.npz")
                with np.load(path) as data:
                    result=score(data['labels'],data['predictions'],data['predictions'],data['predictions'])
                metrics['test'][str(rate)]={'weighted_f1':result['baseline_wf1']/100}
            (run/'metrics.json').write_text(json.dumps(metrics))
            with patch('sys.argv',['analyze','--run',str(run),'--output',str(output)]), contextlib.redirect_stdout(io.StringIO()):
                main()
            summary=json.loads((output/'summary.json').read_text())
            self.assertAlmostEqual(summary['overall']['baseline_wf1'],81.06809539495711)
            self.assertEqual(summary['overall']['delta'],0.)
            self.assertTrue(all(c['delta']==0 and c['corrections']==0 and c['harms']==0 for c in summary['fixed_four_cells']))
            self.assertEqual(sum(c['n'] for c in summary['fixed_four_cells']),4824)


if __name__=='__main__':unittest.main()
