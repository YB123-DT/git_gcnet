import importlib.util
from pathlib import Path
import tempfile
import unittest

from experiments.osram_meld_b3_capacity_hpo_20261016 import run as capacity


class Fixed43Tests(unittest.TestCase):
    def test_reference_filter_keeps_only_matching_structure(self):
        from experiments.osram_meld_b3_fixed43_hpo_20261016 import run
        import json
        from unittest.mock import patch
        roots = []
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for label, count in [('original', 20), ('extension', 4), ('capacity', 4)]:
                root = base / label
                (root / 'queue').mkdir(parents=True)
                jobs = {}
                for i in range(count):
                    if label == 'capacity':
                        params = dict(bottlenecks=i + 4, rounds=3, dim=64, heads=2,
                                      ffn_hidden=512, lr_multiplier=.1)
                    else:
                        index = i if label == 'original' else i + 20
                        params = dict(bottlenecks=index % 8 + 1, rounds=index // 8 + 1,
                                      **capacity.DEFAULT_KNOBS)
                    config = capacity.make_config(**params)
                    out = root / str(i)
                    out.mkdir()
                    summary = dict(status='complete', provenance={'git_sha':
                        run.WORKER_SHA if label == 'capacity' else run.HISTORICAL_WORKER_SHA})
                    (out / 'SUMMARY.json').write_text(json.dumps(summary))
                    jobs[str(i)] = dict(id=str(i), config=config, output=str(out))
                (root / 'queue' / 'QUEUE.json').write_text(json.dumps({'jobs': jobs}))
                roots.append((label, root, count))
            output = base / 'new'
            output.mkdir()
            with patch.object(run, 'paired_objective', return_value=.2):
                reference = run.freeze_reference(output, roots, {'mask_digest': 'same'})
            self.assertEqual(len(reference['reserved']), 28)
            self.assertEqual(len(reference['completed']), 2)
            self.assertTrue(all((r['params']['bottlenecks'], r['params']['rounds']) == (4, 3)
                                for r in reference['completed']))
            self.assertTrue(str(run.WORKER_ROOT).endswith('_6ce8ca6'))
            self.assertEqual(run.WORKER_SHA, '6ce8ca60d507fe81dd37d9ce74623104ee2e7098')

    def test_candidates_never_change_four_tokens_three_rounds(self):
        reserved = [dict(bottlenecks=4, rounds=3, **capacity.DEFAULT_KNOBS)]
        rows = capacity.feasible_candidates(reserved, fixed_structure=(4, 3))
        self.assertTrue(rows)
        self.assertTrue(all((r['bottlenecks'], r['rounds']) == (4, 3) for r in rows))
        self.assertTrue(all(capacity.signature(r) != capacity.signature(reserved[0]) for r in rows))
        self.assertEqual(capacity.feasible_candidates(reserved, {'rounds': 2}, fixed_structure=(4, 3)), [])

    def test_reuses_matching_largest_smoke_and_rejects_wrong_worker(self):
        from experiments.osram_meld_b3_fixed43_hpo_20261016 import run
        from unittest.mock import patch
        smoke = dict(status='complete', smoke_only=True, peak_memory_mib=3204.0830078125,
            provenance={'git_sha': run.WORKER_SHA}, config=capacity.make_config(8, 4, 256, 8, 512, 1.))
        with patch.object(run, 'digest', return_value='same'):
            self.assertEqual(run.validate_smoke(smoke, Path('prior.json'))['peak_memory_mib'],
                             3204.0830078125)
            smoke['provenance']['git_sha'] = 'other'
            with self.assertRaises(RuntimeError):
                run.validate_smoke(smoke, Path('prior.json'))

    @unittest.skipUnless(importlib.util.find_spec('optuna'), 'approved remote environment')
    def test_fixed_four_actual_trials_and_resume(self):
        class Scheduler:
            calls = 0
            def run_jobs(self, jobs):
                self.calls += 1
                self.jobs = jobs
                metrics = dict(ACC=61., UA=36., WF1=57., CE=1.)
                return [dict(status='complete', summary=dict(status='complete',
                    paired={'mask_digest': 'same', 'on': dict(rates={}, mean=metrics, high=metrics)},
                    testbest=dict(rates={}, mean=metrics, high=metrics),
                    parameters=1, block_parameters=1, training_seconds=1, peakmemorybytes=1))
                    for _ in jobs]
        observations = [dict(bottlenecks=4, rounds=3, **capacity.DEFAULT_KNOBS),
                        dict(bottlenecks=4, rounds=3, dim=64, heads=2, ffn_hidden=512, lr_multiplier=.1)]
        reference = dict(reserved=observations,
            completed=[dict(key=str(i), params=p, value=.1 * i) for i, p in enumerate(observations)],
            flat_paired={'mask_digest': 'same', 'on': {'mean': {'ACC': 60., 'UA': 35.}}})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            scheduler = Scheduler()
            kwargs = dict(fixed_structure=(4, 3), study_name='test_fixed43',
                          previous_budget=28, job_prefix='b3_fixed43', n_startup_trials=2)
            report = capacity.run_search(output, reference, scheduler, **kwargs)
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(len(report['trials']), 4)
            configs = [r['params'] for r in report['trials']]
            self.assertEqual(len({capacity.signature(p) for p in configs}), 4)
            self.assertTrue(all((p['bottlenecks'], p['rounds']) == (4, 3) for p in configs))
            for p, required in zip(configs, capacity.REQUIRED_KNOBS):
                self.assertNotEqual(p[required], capacity.DEFAULT_KNOBS[required])
            calls = scheduler.calls
            capacity.run_search(output, reference, scheduler, **kwargs)
            self.assertEqual(scheduler.calls, calls)


if __name__ == '__main__':
    unittest.main()
