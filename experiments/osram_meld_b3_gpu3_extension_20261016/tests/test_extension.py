import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.osram_meld_b3_gpu3_extension_20261016 import run


class ExtensionTests(unittest.TestCase):
    def test_frozen_reference_hashes_the_bytes_it_actually_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / 'original'
            (original / 'queue').mkdir(parents=True)
            trials = b'[]'
            queue = json.dumps({'jobs': {str(i): {'config': {
                'osram_mbt_bottlenecks': i % 8 + 1,
                'osram_mbt_rounds': i // 8 + 1}} for i in range(20)}}).encode()
            (original / 'TRIALS.json').write_bytes(trials)
            (original / 'queue' / 'QUEUE.json').write_bytes(queue)
            # A fresh path hash could see a later atomically replaced version.
            with patch.object(run, 'digest', return_value='later-file-version'):
                ref = run.freeze_reference(root / 'new', original, {})
            self.assertEqual(ref['original_trials_sha256'], hashlib.sha256(trials).hexdigest())
            self.assertEqual(ref['original_queue_sha256'], hashlib.sha256(queue).hexdigest())

    def test_config_preserves_original_flat(self):
        config = run.make_config(8, 4)
        for key, value in dict(learning_rate=.001, weight_decay=1e-5,
                               dropout=.5, epochs=50, seed=66, batch_size=32,
                               osram_output_dim=1600,
                               missing_mask_protocol='gcnet-random-mask-v1').items():
            self.assertEqual(config[key], value)
        self.assertFalse(config.get('osram_disable_local_skip', False))

    def test_reference_is_frozen_and_reserves_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / 'original'
            original.mkdir()
            trials = [dict(number=i, state='RUNNING', params=dict(bottlenecks=i % 8 + 1,
                       rounds=i // 8 + 1), user_attrs={}) for i in range(20)]
            # Original export can lag the queue by one pending proposal.
            (original / 'TRIALS.json').write_text(json.dumps(trials[:19]))
            (original / 'queue').mkdir()
            (original / 'queue' / 'QUEUE.json').write_text(json.dumps({'jobs': {
                str(t['number']): {'config': dict(
                    osram_mbt_bottlenecks=t['params']['bottlenecks'],
                    osram_mbt_rounds=t['params']['rounds'])} for t in trials}}))
            flat = {'mask_digest': 'same', 'on': {'mean': {'ACC': 60., 'UA': 35.}}}
            frozen = run.freeze_reference(root / 'new', original, flat)
            self.assertEqual(len(frozen['reserved']), 20)
            (original / 'TRIALS.json').write_text('[]')
            self.assertEqual(run.freeze_reference(root / 'new', original, flat), frozen)

    @unittest.skipUnless(importlib.util.find_spec('optuna'), 'Optuna on approved remote environment')
    def test_four_unique_tpe_proposals_and_restart(self):
        import optuna
        class FakeScheduler:
            calls = 0
            def run_jobs(self, jobs):
                self.calls += 1
                self.jobs = jobs
                return [dict(status='complete', summary=dict(status='complete',
                    testbest={'mean': {'ACC': 61., 'UA': 36.}},
                    paired={'mask_digest': 'same', 'on': {'mean': {'ACC': 61., 'UA': 36.}},
                            'off': {'mean': {'ACC': 60., 'UA': 35.}}})) for _ in jobs]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reserved = [dict(bottlenecks=i % 8 + 1, rounds=i // 8 + 1) for i in range(20)]
            references = [dict(number=i, params=c, value=.1 * i) for i, c in enumerate(reserved[:15])]
            reference = dict(reserved=reserved, completed=references,
                flat_paired={'mask_digest': 'same', 'on': {'mean': {'ACC': 60., 'UA': 35.}}})
            # Simulate interruption between ask and persisting the second parameter.
            study = optuna.create_study(study_name=run.STUDY_NAME,
                storage='sqlite:///' + str(root / 'EXTENSION.sqlite3'), direction='maximize')
            study.enqueue_trial({'bottlenecks': 8, 'rounds': 4})
            partial = study.ask()
            self.assertEqual(partial.suggest_int('bottlenecks', 1, 8), 8)
            scheduler = FakeScheduler()
            report = run.run_extension(root, reference, scheduler)
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(len(report['new_runs']), 4)
            proposed = {(j['config']['osram_mbt_bottlenecks'], j['config']['osram_mbt_rounds'])
                        for j in scheduler.jobs}
            self.assertEqual(len(proposed), 4)
            self.assertIn((8, 4), proposed)
            self.assertTrue(proposed.isdisjoint({tuple(c.values()) for c in reserved}))
            calls = scheduler.calls
            run.run_extension(root, reference, scheduler)
            self.assertEqual(scheduler.calls, calls)
            study = optuna.load_study(study_name=run.STUDY_NAME,
                storage='sqlite:///' + str(root / 'EXTENSION.sqlite3'))
            self.assertEqual(len(study.trials), 19)


if __name__ == '__main__':
    unittest.main()
