"""Bounded, no-GPU tests for the authorized Nested-only screening queue."""
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest


class NestedCmercQueueTests(unittest.TestCase):
    def runner(self):
        name = 'experiments.osram_nested_cmerc_tuning_20261023.run'
        try:
            spec = importlib.util.find_spec(name)
        except ModuleNotFoundError:
            spec = None
        self.assertIsNotNone(spec, 'The finite Nested-only controller is missing')
        return __import__(name, fromlist=['main'])

    def test_exact_fourteen_jobs_no_iemocap_four_or_new_seeds(self):
        run = self.runner()
        jobs = run.plan_jobs(Path('/fresh'), 'sha', run.GPU)
        self.assertEqual(len(jobs), 14)
        self.assertEqual(len({j['output'] for j in jobs}), 14)
        self.assertEqual({j['dataset'] for j in jobs}, {'CMUMOSEI', 'IEMOCAPSix'})
        self.assertEqual([j['dataset'] for j in jobs[:2]], ['CMUMOSEI', 'IEMOCAPSix'])
        self.assertEqual([j['method'] for j in jobs[:2]], ['none'] * 2)
        methods = {'none', 'nested_gnn_rooted_evidence', 'nested_dim32', 'nested_dim128',
                   'mosei_nested_depth2', 'mosei_nested_mlp128', 'mosei_nested_groups4'}
        for dataset in ('CMUMOSEI', 'IEMOCAPSix'):
            selected = [j for j in jobs if j['dataset'] == dataset]
            self.assertEqual({j['method'] for j in selected}, methods)
            for job in selected:
                self.assertEqual(job['seed'], 66)
                self.assertEqual(job['fold'], 1 if dataset == 'CMUMOSEI' else 5)
                self.assertEqual(job['epochs'], 60 if dataset == 'CMUMOSEI' else 100)
                self.assertEqual(job['learning_rate'], 1e-4 if dataset == 'CMUMOSEI' else 1e-3)
                self.assertIn('cmerc-repro/bin/python', job['command'][0])
                self.assertEqual(job['command'][job['command'].index('--method') + 1], job['method'])
                self.assertEqual(job['mask_protocol'], 'gcnet-random-mask-v1')

    def test_admission_requires_resources_and_hard_four_process_limit(self):
        admit = self.runner().can_admit
        self.assertTrue(admit(free_gpu_gib=26, ram_gib=64, disk_gib=25, active_count=3, per_gpu=4))
        for changed in ({'free_gpu_gib': 25.99}, {'ram_gib': 63.99}, {'disk_gib': 24.99}, {'active_count': 4}):
            args = dict(free_gpu_gib=26, ram_gib=64, disk_gib=25, active_count=3, per_gpu=4)
            args.update(changed)
            self.assertFalse(admit(**args))
        with self.assertRaises(ValueError):
            admit(free_gpu_gib=100, ram_gib=100, disk_gib=100, active_count=0, per_gpu=5)
        # A process still loading CPU data must reserve its unseen CUDA allocation.
        self.assertFalse(admit(free_gpu_gib=19, ram_gib=64, disk_gib=25, active_count=2, per_gpu=4))
        self.assertTrue(admit(free_gpu_gib=20, ram_gib=64, disk_gib=25, active_count=2, per_gpu=4))

    def test_uuid_whitelist_excludes_failed_gpu4_and_indices(self):
        run = self.runner()
        run.validate_gpu_uuid(run.GPU)
        run.validate_gpu_uuid(run.GPU1)
        for invalid in ('4', run.BAD_UUID, 'GPU-unknown', 'GPU-good'):
            with self.assertRaises(ValueError):
                run.validate_gpu_uuid(invalid)

    def test_completion_needs_matching_full_budget_summary(self):
        run = self.runner()
        with tempfile.TemporaryDirectory() as temporary:
            job = run.plan_jobs(Path(temporary), 'sha', run.GPU)[0]
            output = Path(job['output'])
            output.mkdir(parents=True)
            summary = dict(status='complete', phase='train', dataset=job['dataset'],
                           method=job['method'], fold=job['fold'], seed=66, source_commit='sha',
                           epoch=job['epochs'], evaluation={str(i / 10): {} for i in range(8)},
                           aggregate={'router': {}})
            (output / 'SUMMARY.json').write_text(json.dumps(summary))
            run.validate_summary(job)
            for field, value in (('epoch', 1), ('source_commit', 'other'), ('method', 'flat')):
                wrong = dict(summary, **{field: value})
                (output / 'SUMMARY.json').write_text(json.dumps(wrong))
                with self.assertRaises(ValueError):
                    run.validate_summary(job)

    def test_resume_refuses_living_workers_and_failed_jobs(self):
        run = self.runner()
        import os
        job = run.plan_jobs(Path('/fresh'), 'sha', run.GPU)[0]
        job.update(status='running', pid=os.getpid())
        with self.assertRaises(RuntimeError):
            run.recover_jobs({'jobs': [job]})
        job.update(status='failed')
        with self.assertRaises(RuntimeError):
            run.recover_jobs({'jobs': [job]})

    def test_plain_archive_manifest_checks_bytes_commit_and_unlisted_code(self):
        run = self.runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            code = root / 'gcnet_missing_m3' / 'osram.py'
            code.parent.mkdir()
            code.write_text('value = 1\n')
            files = {'gcnet_missing_m3/osram.py': hashlib.sha256(code.read_bytes()).hexdigest()}
            digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            manifest = dict(source_commit='sha', files=files, manifest_sha256=digest)
            (root / 'SOURCE_SNAPSHOT.json').write_text(json.dumps(manifest))
            self.assertEqual(run.verify_source_snapshot(root, 'sha')['manifest_sha256'], digest)
            with self.assertRaises(ValueError):
                run.verify_source_snapshot(root, 'wrong')
            code.write_text('value = 2\n')
            with self.assertRaises(ValueError):
                run.verify_source_snapshot(root, 'sha')
            code.write_text('value = 1\n')
            extra = code.parent / 'extra.py'
            extra.write_text('unexpected = True\n')
            with self.assertRaises(ValueError):
                run.verify_source_snapshot(root, 'sha')
            extra.unlink()
            (root / 'SOURCE_SNAPSHOT.json').write_text(json.dumps(dict(manifest, manifest_sha256='wrong')))
            with self.assertRaises(ValueError):
                run.verify_source_snapshot(root, 'sha')


if __name__ == '__main__':
    unittest.main()
