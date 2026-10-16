import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from experiments.osram_meld_b3_capacity_hpo_20261016 import run


class CapacityTests(unittest.TestCase):
    def test_original_flat_hyperparameters_and_new_knobs(self):
        config = run.make_config(5, 3, 256, 8, 512, .3)
        self.assertEqual(config['osram_mbt_dim'], 256)
        self.assertEqual(config['osram_mbt_heads'], 8)
        self.assertEqual(config['osram_mbt_ffn_hidden'], 512)
        self.assertEqual(config['osram_mbt_lr_multiplier'], .3)
        for key, expected in dict(learning_rate=.001, weight_decay=1e-5, dropout=.5,
                batch_size=32, epochs=50, seed=66, osram_output_dim=1600,
                missing_mask_protocol='gcnet-random-mask-v1').items():
            self.assertEqual(config[key], expected)
        self.assertFalse(config.get('osram_disable_local_skip', False))

    def test_each_new_candidate_changes_other_knob(self):
        defaults = dict(bottlenecks=5, rounds=3, dim=128, heads=4, ffn_hidden=256,
                        lr_multiplier=1.)
        self.assertFalse(run.is_new_knob_candidate(defaults))
        self.assertTrue(run.is_new_knob_candidate(dict(defaults, heads=8)))
        for key, value in [('dim', 96), ('heads', 3), ('ffn_hidden', 96),
                           ('lr_multiplier', .2), ('rounds', 5)]:
            with self.assertRaises(ValueError):
                run.make_config(**dict(defaults, **{key: value}))

    def test_unique_full_tuple_allows_old_token_pair_with_other_capacity(self):
        old = dict(bottlenecks=5, rounds=3, **run.DEFAULT_KNOBS)
        self.assertNotEqual(run.signature(old), run.signature(dict(old, heads=8)))
        candidates = run.feasible_candidates([old])
        self.assertTrue(candidates)
        self.assertTrue(all(run.is_new_knob_candidate(c) for c in candidates))
        self.assertTrue(any(c['bottlenecks'] == 5 and c['rounds'] == 3 for c in candidates))

    @unittest.skipUnless(importlib.util.find_spec('optuna'), 'approved remote environment')
    def test_finite_four_actual_tpe_trials_and_resume(self):
        class FakeScheduler:
            calls = 0
            def run_jobs(self, jobs):
                self.calls += 1
                self.jobs = jobs
                metrics = {'ACC': 61., 'UA': 36., 'WF1': 57., 'CE': 1.}
                return [dict(status='complete', summary=dict(status='complete',
                    paired={'mask_digest': 'same', 'on': {'rates': {}, 'mean': metrics,
                                                        'high': metrics}},
                    testbest={'rates': {}, 'mean': metrics, 'high': metrics},
                    parameters=1, block_parameters=1, training_seconds=1, peakmemorybytes=1))
                    for _ in jobs]
        reference = dict(reserved=[dict(bottlenecks=i % 8 + 1, rounds=i // 8 + 1,
                                      **run.DEFAULT_KNOBS) for i in range(24)],
            completed=[dict(key=str(i), params=dict(bottlenecks=i + 1, rounds=1,
                **run.DEFAULT_KNOBS), value=.1 * i) for i in range(6)],
            flat_paired={'mask_digest': 'same', 'on': {'mean': {'ACC': 60., 'UA': 35.}}})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            scheduler = FakeScheduler()
            report = run.run_search(output, reference, scheduler)
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(len(report['trials']), 4)
            configs = [r['params'] for r in report['trials']]
            self.assertEqual(len({run.signature(c) for c in configs}), 4)
            self.assertTrue(all(run.is_new_knob_candidate(c) for c in configs))
            for config, required in zip(configs, run.REQUIRED_KNOBS):
                self.assertNotEqual(config[required], run.DEFAULT_KNOBS[required])
            calls = scheduler.calls
            run.run_search(output, reference, scheduler)
            self.assertEqual(scheduler.calls, calls)
            import optuna
            study = optuna.load_study(study_name=run.STUDY_NAME,
                storage='sqlite:///' + str(output / 'CAPACITY.sqlite3'))
            self.assertEqual(len(study.trials), 10)


if __name__ == '__main__':
    unittest.main()
