import unittest
import tempfile
import json
from pathlib import Path
import numpy as np
from experiments.osram_readout_paths_20260930.analyze import paired_metrics, analyze


class PairedMetricsTests(unittest.TestCase):
    def test_mosi_neutral_in_loss_not_sign_counts(self):
        y = np.array([1., -1., 0., 1.])
        base = np.array([[-1.], [-1.], [2.], [1.]])
        new = np.array([[1.], [1.], [0.], [1.]])
        r = paired_metrics(y, new, base, 'CMUMOSI')
        self.assertEqual(r['wrong_to_right'], 1)
        self.assertEqual(r['right_to_wrong'], 1)
        self.assertEqual(r['metric_count'], 3)
        self.assertEqual(r['sample_count'], 4)
        self.assertEqual(r['loss'], 1.)
        self.assertEqual(r['delta_loss'], -1.)
        # Equal correction/harm counts do not imply equal weighted F1.
        self.assertAlmostEqual(r['delta_weighted_f1'], -2./15.)

    def test_ce_stable_and_identity(self):
        y = np.array([0, 1])
        z = np.array([[1000., 0.], [0., 1000.]])
        r = paired_metrics(y, z, z, 'IEMOCAPSix')
        self.assertEqual(r['loss'], 0.)
        self.assertEqual(r['wrong_to_right'], 0)
        self.assertEqual(r['right_to_wrong'], 0)
        self.assertEqual(r['weighted_f1'], 1.)

    def test_nested_artifacts_and_neutral_only_pattern(self):
        settings = [(a,a,m) for a in (.8,1.,1.2) for m in (.8,1.,1.2)] + [(.8,1.,1.),(1.2,1.,1.),(1.,.8,1.),(1.,1.2,1.)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'input'/'seed_66').mkdir(parents=True)
            np.savez(root/'input'/'seed_66'/'validation.npz',
                predictions=np.ones((13,3,1)), labels=np.array([0.,1.,-1.]),
                availability=np.array([[1,0,0],[0,1,0],[0,1,0]]),
                conversation_ids=np.array(['c1','c1','c2']), utterance_indices=np.array([0,1,0]),
                settings=np.asarray(settings),seed=66,rate=.7,split='validation')
            analyze(root/'input',root/'output')
            result=json.loads((root/'output'/'SUMMARY.json').read_text())
            self.assertEqual(result['files'],1)
            self.assertTrue(all(r['delta_loss_mean']==0 for r in result['rows']))
            self.assertTrue(all(r['delta_wf1_pp_mean'] is None for r in result['rows'] if r['pattern']=='A'))


if __name__ == '__main__':
    unittest.main()
