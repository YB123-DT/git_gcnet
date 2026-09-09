"""Stdlib-only regression checks for five-seed summary statistics."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'experiments/osram_write_intervention_20260909/summarize_five_seed.py'


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SCRIPT.exists(), 'summary implementation is missing')
        spec = importlib.util.spec_from_file_location('summary', SCRIPT)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for seed in range(66, 71):
            results = []
            for rate in (0, .1, .3, .5, .7):
                for mode in ('reference', 'protected', 'global', 'fixed0.9'):
                    value = .7 + (seed - 65) * .001 * (mode == 'global')
                    results.append(dict(mode=mode, rate=rate, mask_sha256=str(rate),
                        task_metrics=dict(weighted_f1=value, accuracy=value, macro_f1=value),
                        retention={}, current_observed_write_fit={'audio': {'counts': seed,
                        'metrics': {'fit_gain': {'count': seed, 'mean': seed - 65}}}}))
            run = self.root / f'iemocap4_seed{seed}'
            run.mkdir()
            (run / 'metadata.json').write_text(json.dumps(dict(seed=seed, config={'seed': seed},
                checkpoint=f'/seed{seed}/best.pt', checkpoint_sha256=str(seed),
                weights_unchanged=True, all_mode_masks_equal=True, results=results)))

    def test_paired_stats_and_seed_equal_write_fit(self):
        self.module.summarize(self.root)
        with (self.root / 'summary.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        paired = next(r for r in rows if r['section'] == 'task' and r['rate'] == 'all'
                      and r['mode'] == 'global-minus-fixed0.9' and r['metric'] == 'weighted_f1')
        self.assertAlmostEqual(float(paired['mean']), .003)
        self.assertAlmostEqual(float(paired['sample_std']), .00158113883008419)
        self.assertEqual(paired['positive'], '5')
        self.assertEqual(float(paired['sign_flip_p_two_sided']), .0625)
        fit = next(r for r in rows if r['section'] == 'write_fit' and r['rate'] == '0'
                   and r['mode'] == 'global' and r['metric'] == 'fit_gain' and r['modality'] == 'audio')
        self.assertEqual(float(fit['mean']), 3)
        self.assertTrue((self.root / 'RESULT_FULL5.md').exists())

    def test_rejects_missing_mode(self):
        path = self.root / 'iemocap4_seed66/metadata.json'
        data = json.loads(path.read_text())
        data['results'].pop()
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            self.module.summarize(self.root)

    def test_rejects_failed_integrity(self):
        path = self.root / 'iemocap4_seed66/metadata.json'
        data = json.loads(path.read_text())
        data['weights_unchanged'] = False
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'weights_unchanged'):
            self.module.summarize(self.root)

    def test_rejects_mask_mismatch_even_with_true_flag(self):
        path = self.root / 'iemocap4_seed66/metadata.json'
        data = json.loads(path.read_text())
        data['results'][0]['mask_sha256'] = 'wrong'
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'mask'):
            self.module.summarize(self.root)


if __name__ == '__main__':
    unittest.main()
