import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


PATH = Path(__file__).resolve().parents[1] / 'experiments/osram_frozen_memory_audit_20261009/analyze.py'


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(PATH.exists(), 'analysis implementation missing')
        spec = importlib.util.spec_from_file_location('analysis', PATH)
        self.a = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.a)

    def test_matched_identity_and_neutral_exclusion(self):
        a = dict(test_conversation=np.array(['x', 'x', 'x']), test_utterance=np.array([0, 1, 2]),
                 test_row=np.array([0, 1, 2]), test_target=np.array([-1., 0., 1.]),
                 test_prediction=np.array([1., 1., 1.]))
        b = {k: v[[2, 0, 1]] for k, v in a.items()}
        b['test_prediction'] = np.array([-1., -1., -1.])
        r = self.a.compare_predictions(a, b)
        self.assertEqual((r['corrections'], r['harms'], r['nonneutral_n']), (1, 1, 2))
        self.assertEqual(r['n'], 3)

    def test_macro_rates_do_not_weight_by_rows(self):
        rows = [dict(rate=r, seed=66, task='current', probe='A', n=n, mse=v,
                     weighted_f1_nonzero=v, acc_nonzero=v)
                for r, n, v in [(0., 100, 0.), (.7, 1, 1.)]]
        result = self.a.aggregate_probes(rows)
        mean = next(x for x in result if x['scope'] == 'mean8' and x['seed'] == 66)
        self.assertEqual(mean['mse'], .5)
        self.assertEqual(mean['rates_n'], 2)
        self.assertFalse(mean['complete'])

    def test_cluster_bootstrap_and_empty(self):
        rows = [dict(conversation_id='c', delta_mse_delete_minus_control=2.)] * 4
        self.assertEqual(self.a.cluster_interval(rows, 30), [2., 2.])
        self.assertIsNone(self.a.cluster_interval([], 30))

    def test_incomplete_root_strict_json(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / 'root'
            root.mkdir()
            (root / 'STATUS.json').write_text(json.dumps(dict(status='running', rates=[0., .1], records=[])))
            output = Path(d) / 'out'
            result = self.a.analyze([root], output, bootstrap=0)
            self.assertTrue(result['partial'])
            self.assertEqual(len(result['incomplete']), 2)
            self.assertIn('TEST-ORACLE', (output / 'RESULT.md').read_text())
            self.assertNotIn('NaN', (output / 'SUMMARY.json').read_text())

    def test_downloaded_complete_rate_uses_local_predictions(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / 'root'
            directory = root / 'rate_0p7'
            directory.mkdir(parents=True)
            result = dict(rate=.7, frozen_unchanged=True, model_frozen=True,
                          baseline_parity_verified=True, baseline_reference_weighted_f1=.5,
                          original_test=dict(weighted_f1_nonzero=.5))
            (root / 'STATUS.json').write_text(json.dumps(dict(status='complete', rates=[.7], records=[result])))
            (directory / 'RESULT.json').write_text(json.dumps(result))
            (directory / 'interventions.json').write_text(json.dumps(dict(rows=[], coverage=[])))
            runs = []
            for task in self.a.TASKS:
                for probe in ('A', 'B', 'C'):
                    path = directory / 'probes' / 'seed66' / task / probe
                    path.mkdir(parents=True)
                    np.savez(path / 'predictions.npz', test_conversation=np.array(['x', 'x']),
                             test_utterance=np.array([0, 1]), test_row=np.array([0, 1]),
                             test_target=np.array([-1., 1.]), test_prediction=np.array([-1., 1.]))
                    runs.append(dict(seed=66, task=task, probe=probe, selection_split='test',
                                     predictions='/unavailable/server/predictions.npz'))
            (directory / 'probes' / 'summary.json').write_text(json.dumps(dict(runs=runs, protocol=dict(seeds=[66]))))
            report = self.a.analyze([root], Path(d) / 'out', bootstrap=0)
            self.assertEqual(report['completed_rates'], [.7])
            self.assertTrue(report['verification_passed'])
            self.assertEqual(len(report['probe_comparisons']), 8)
            self.assertEqual(len(report['comparison_macro']), 16)
            self.assertEqual(report['incomplete'], [])

    def test_intervention_metrics_zero_neutral_and_exact_lag(self):
        rows = []
        for t, y in enumerate([0., -1., 1.]):
            rows.append(dict(conversation_id='c', utterance_index=t, rate=.7, split='test',
                             family='remove_T_vs_A', pattern=7, y=y,
                             pred_real=1., pred_delete=-1., pred_control=1., lag_mismatch=0,
                             lag_delete=1, lag_control=1, current_local_max_error=0.))
        output = self.a.intervention_summary(rows, .7, [], 0)
        all_rows = next(r for r in output if r['pattern'] == 'all' and r['subset'] == 'all')
        self.assertEqual(all_rows['delete_corrections'], 1)
        self.assertEqual(all_rows['delete_harms'], 1)
        self.assertEqual(all_rows['delete_flips'], 2)
        self.assertEqual(len(output), 4)


if __name__ == '__main__':
    unittest.main()
