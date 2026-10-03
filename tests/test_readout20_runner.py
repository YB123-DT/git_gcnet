import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'experiments/osram_readout20_20261003'


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((PACKAGE / 'run.py').exists(), 'runner not implemented')
        from experiments.osram_readout20_20261003 import run, queue, summarize
        self.run, self.queue, self.summary = run, queue, summarize

    def base(self):
        return dict(self.run.EXPECTED, unrelated_original_setting='preserve')

    def test_withdrawn_screen_cannot_launch(self):
        self.assertTrue(hasattr(self.run, 'require_active_screen'))
        with self.assertRaisesRegex(RuntimeError, 'withdrawn'):
            self.run.require_active_screen()
        with self.assertRaisesRegex(RuntimeError, 'withdrawn'):
            self.run.train(None)
        with self.assertRaisesRegex(RuntimeError, 'withdrawn'):
            self.queue.coordinate(None)

    def test_only_candidate_config_delta(self):
        original = self.base()
        changed = self.run.candidate_config(original, 'film')
        self.assertEqual(set(changed) - set(original), {'osram_readout_candidate'})
        self.assertEqual({k: changed[k] for k in original}, original)
        for field, bad in [('seed', 67), ('epochs', 99), ('task_regression_loss', 'smooth-l1')]:
            with self.assertRaises(ValueError):
                self.run.candidate_config(dict(original, **{field: bad}), 'film')
        for field in self.run.FORBIDDEN_FLAGS:
            with self.assertRaises(ValueError):
                self.run.candidate_config(dict(original, **{field: True}), 'film')

    def test_effective_config_normalizes_tuple_serialization(self):
        from dataclasses import dataclass
        @dataclass
        class Config:
            train_missing_rates: tuple = (0., .1)
        self.assertTrue(hasattr(self.run, 'effective_config_dict'))
        self.assertEqual(self.run.effective_config_dict(Config()), {'train_missing_rates': [0., .1]})

    def test_twenty_locked_candidates_and_commit(self):
        self.assertEqual(len(self.run.CANDIDATES), 20)
        self.assertEqual(len(set(self.run.CANDIDATES)), 20)
        manifest = dict(code_commit='a' * 40, candidates=list(self.run.CANDIDATES))
        self.run.validate_manifest(manifest, 'a' * 40)
        for altered in [dict(manifest, candidates=list(self.run.CANDIDATES[:-1])),
                        dict(manifest, code_commit='b' * 40)]:
            with self.assertRaises(ValueError):
                self.run.validate_manifest(altered, 'a' * 40)
        with self.assertRaises(ValueError):
            self.run.candidate_config(self.base(), 'none')

    def test_bad_gpu_and_identity_pid_reuse(self):
        for gpu in ('0', '4', '6', '7'):
            with self.assertRaises(ValueError):
                self.run.validate_gpu(gpu)
        self.run.validate_gpu('5')
        self.run.validate_gpu('1')
        self.assertFalse(self.queue.process_matches(dict(pid=99999999, start_ticks='0', boot_id='x')))
        own = self.queue.process_identity(__import__('os').getpid())
        self.assertTrue(self.queue.process_matches(own))
        self.assertFalse(self.queue.process_matches(dict(own, start_ticks='0')))

    def test_summary_math_and_incomplete_never_wins(self):
        metrics = dict(test={f'{i / 10:.1f}': {'weighted_f1': .8 + i / 100} for i in range(8)})
        scores = self.summary.rate_scores(metrics)
        self.assertAlmostEqual(scores['mean_8rate'], 83.5)
        self.assertAlmostEqual(scores['high_missing'], 86.)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertEqual(self.run.completion_status(root)[0], 'missing')
            (root / 'PROVENANCE.json').write_text(json.dumps({'status': 'complete'}))
            self.assertEqual(self.run.completion_status(root)[0], 'incomplete')

    def test_queue_has_no_torch_import(self):
        import subprocess
        code = 'import sys; from experiments.osram_readout20_20261003 import queue; assert "torch" not in sys.modules'
        subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True)

    def test_stale_queue_complete_cannot_bypass_artifact_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = root / 'reference'
            reference.mkdir()
            metrics = dict(test={f'{i / 10:.1f}': {'weighted_f1': .8} for i in range(8)})
            self.run.write(reference / 'metrics.json', metrics)
            self.run.write(root / 'QUEUE.json', {'children': {'film': {'status': 'complete'}}})
            result = self.summary.summarize(root, reference)
            self.assertEqual(result['completed'], 0)
            self.assertNotEqual(result['candidates'][0]['status'], 'complete')

    def test_complete_requires_hundred_epochs_eight_best_and_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            artifacts = {
                'history.json': [{'epoch': i} for i in range(1, 101)],
                'metrics.json': {'selection_protocol': 'per-rate-test-oracle',
                    'test': {f'{i / 10:.1f}': {'weighted_f1': .8} for i in range(8)}},
                'config.json': {'seed': 66, 'epochs': 100}, 'PARAMETERS.json': {'total': 42}}
            for name, value in artifacts.items():
                self.run.write(output / name, value)
            for i in range(8):
                (output / f'best_miss_0p{i}.pt').write_bytes(b'unit-test-artifact')
            record = dict(status='complete', source_sha256={'a': 'b'}, reference_sha256={'a': 'b'},
                          code_commit='a' * 40, manifest_sha256='a' * 64, environment={'python': 'test'},
                          pid=10, gpu='1', gpu_uuid='GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049', outputs_verified=True,
                          config={'seed': 66, 'epochs': 100},
                          artifact_sha256={name: self.run.sha(output / name) for name in artifacts})
            self.run.write(output / 'PROVENANCE.json', record)
            self.assertEqual(self.run.completion_status(output)[0], 'complete')
            self.run.write(output / 'PROVENANCE.json', dict(record, gpu='5'))
            self.assertEqual(self.run.completion_status(output)[0], 'incomplete')
            self.run.write(output / 'PROVENANCE.json', record)
            self.run.write(output / 'history.json', [{'epoch': 1}] * 100)
            self.assertEqual(self.run.completion_status(output)[0], 'incomplete')
            self.run.write(output / 'history.json', artifacts['history.json'])
            (output / 'best_miss_0p7.pt').unlink()
            self.assertEqual(self.run.completion_status(output)[0], 'incomplete')


if __name__ == '__main__':
    unittest.main()
