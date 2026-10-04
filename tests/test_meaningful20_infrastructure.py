import importlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest


PREFIX = 'experiments.osram_meaningful20_20261003.'


class InfrastructureTests(unittest.TestCase):
    def module(self, name):
        self.assertIsNotNone(importlib.util.find_spec(PREFIX + name), f'{name} is not implemented')
        return importlib.import_module(PREFIX + name)

    def cards(self):
        return [{'id': f'candidate_{i:02}', 'design_sha256': f'{i:064x}',
                 'source': {'paper': 'https://example.org/paper', 'code': 'publisher listing'},
                 'accepted': True} for i in range(20)]

    def test_manifest_requires_twenty_distinct_accepted_cards(self):
        m = self.module('manifest')
        data = {'round_id': 'round01', 'status': 'source_accepted', 'cards': self.cards()}
        self.assertEqual(len(m.validate_round(data)), 20)
        for cards in (self.cards()[:19], self.cards()[:-1] + [self.cards()[0]]):
            with self.assertRaises(ValueError):
                m.validate_round(dict(data, cards=cards))
        cards = self.cards(); cards[0]['accepted'] = False
        with self.assertRaises(ValueError): m.validate_round(dict(data, cards=cards))
        with self.assertRaises(ValueError):
            m.validate_round(data, prior_design_hashes={self.cards()[0]['design_sha256']})

    def test_snapshot_detects_mutation_and_rejects_symlinks(self):
        m = self.module('manifest')
        with tempfile.TemporaryDirectory() as root:
            source = Path(root, 'source'); source.mkdir()
            (source / 'model.py').write_text('value = 1\n')
            snapshot = Path(root, 'snapshot')
            m.create_snapshot(source, snapshot, files=['model.py'], code_commit='a' * 40)
            m.verify_snapshot(snapshot)
            (source / 'model.py').write_text('value = 2\n')
            self.assertEqual((snapshot / 'model.py').read_text(), 'value = 1\n')
            (snapshot / 'model.py').write_text('value = 3\n')
            with self.assertRaises(ValueError): m.verify_snapshot(snapshot)
            (source / 'link.py').symlink_to(source / 'model.py')
            with self.assertRaises(ValueError):
                m.create_snapshot(source, Path(root, 'bad'), files=['link.py'], code_commit='a' * 40)

    def test_gpu_four_and_uuid_remapping_are_rejected(self):
        p = self.module('preflight')
        mapping = {'0': 'GPU-good', '4': 'GPU-bad'}
        p.validate_gpu('0', 'GPU-good', mapping)
        for index, uuid in [('4', 'GPU-bad'), ('0', 'GPU-bad'), ('0', 'GPU-other')]:
            with self.assertRaises(ValueError): p.validate_gpu(index, uuid, mapping)
        banned = 'GPU-46fb379f-cc90-dc82-9b5e-5d011f552264'
        with self.assertRaises(ValueError): p.validate_gpu('0', banned, {'0': banned, '4': 'renumbered'})

    def test_admission_uses_memory_compute_and_full_disk_budget(self):
        p = self.module('preflight')
        profile = {'peak_mib': 4000, 'artifact_gib': 20}
        gpu = {'free_mib': 7000, 'utilization': 20, 'temperature': 50}
        self.assertTrue(p.admission(gpu, profile, disk_free_gib=40)[0])
        for bad, disk in [(dict(gpu, free_mib=5000), 40), (dict(gpu, utilization=95), 40), (gpu, 30)]:
            self.assertFalse(p.admission(bad, profile, disk_free_gib=disk)[0])

    def test_readiness_requires_real_cpu_cuda_evidence_and_profile(self):
        p = self.module('preflight')
        with tempfile.TemporaryDirectory() as root:
            log = Path(root, 'test.log'); log.write_text('Ran tests\nOK\n')
            sha = self.module('manifest').sha
            evidence = {'command': ['python', '-m', 'unittest'], 'returncode': 0,
                        'log': str(log), 'sha256': sha(log)}
            record = {'candidate': 'x', 'design_sha256': 'd', 'source_sha256': {'a.py': 'h'},
                      'cpu': evidence, 'cuda': dict(evidence, gpu_index='0', gpu_uuid='GPU-good'),
                      'profile': {'peak_mib': 1000, 'artifact_gib': 30, 'batch_size': 32,
                                  'train_and_eval': True}, 'status': 'ready'}
            p.validate_readiness(record, candidate='x', design_sha256='d', source_sha256={'a.py': 'h'})
            record['cuda'] = dict(record['cuda'], returncode=1)
            with self.assertRaises(ValueError):
                p.validate_readiness(record, candidate='x', design_sha256='d', source_sha256={'a.py': 'h'})

    def test_fixed_primary_ranking_waits_for_all_twenty(self):
        s = self.module('summarize')
        ids = [c['id'] for c in self.cards()]
        rows = {name: {'status': 'complete', 'mean8': .80, 'high': .99} for name in ids}
        rows[ids[3]]['mean8'] = .82; rows[ids[2]]['mean8'] = .82
        rows[ids[1]]['mean8'] = .83; rows[ids[0]]['mean8'] = .81
        self.assertEqual(s.rank_promotions(ids, rows, .805), [ids[1], ids[2], ids[3]])
        rows[ids[-1]]['status'] = 'running'
        with self.assertRaises(ValueError): s.rank_promotions(ids, rows, .805)

    def test_primary_tie_not_promoted_and_three_seed_negative_repeats(self):
        s = self.module('summarize')
        ids = [c['id'] for c in self.cards()]
        rows = {name: {'status': 'complete', 'mean8': .8} for name in ids}
        self.assertEqual(s.rank_promotions(ids, rows, .8), [])
        baseline = {66: .8, 67: .8, 68: .8}
        self.assertEqual(s.replication_decision({'a': {66: .81, 67: .79, 68: .78}}, baseline)['next'], 'new_round')
        self.assertEqual(s.replication_decision({'a': {66: .81, 67: .8, 68: .8}}, baseline)['best'], 'a')

    def test_process_identity_rejects_reused_pid(self):
        q = self.module('queue')
        record = q.process_identity(os.getpid())
        self.assertTrue(q.process_matches(record))
        self.assertFalse(q.process_matches(dict(record, start_ticks='wrong')))

    def test_queue_reconciles_live_intent_instead_of_relaunch(self):
        q = self.module('queue')
        job = {'status': 'launch_intent', 'output': '/missing', **q.process_identity(os.getpid())}
        self.assertEqual(q.reconcile_job(job)['status'], 'running')
        dead = dict(job, pid=-99, status='launch_intent')
        self.assertEqual(q.reconcile_job(dead)['status'], 'inspection_pending')

    def test_config_preserves_all_baseline_fields_except_new_block(self):
        run = self.module('run')
        reference = Path(__file__).resolve().parents[1] / 'experiments/osram_current_history_relation_20261003/reference/config.json'
        baseline = json.loads(reference.read_text())
        candidate = run.candidate_config(baseline, 'node', seed=66)
        self.assertEqual({k: v for k, v in candidate.items() if k != 'osram_meaningful_block'}, baseline)
        with self.assertRaises(ValueError): run.candidate_config(dict(baseline, epochs=1), 'node', seed=66)
        with self.assertRaises(ValueError): run.candidate_config(dict(baseline, osram_readout_candidate='film'), 'node', seed=66)

    def test_partial_metrics_cannot_be_complete(self):
        run = self.module('run')
        with tempfile.TemporaryDirectory() as root:
            Path(root, 'metrics.json').write_text('{"test":{}}')
            self.assertNotEqual(run.completion_status(root)[0], 'complete')

    def test_launch_failure_is_terminal_until_root_cause_changes(self):
        q = self.module('queue')
        with tempfile.TemporaryDirectory() as root:
            output = Path(root, 'seed_66')
            Path(root, 'seed_66.launch-failure.json').write_text(json.dumps(
                {'run_id': 'x', 'error': 'bad source', 'failure_category': 'preflight'}))
            job = q.reconcile_job({'status': 'running', 'run_id': 'x', 'output': str(output), 'pid': -99})
            self.assertEqual(job['status'], 'failed')
            self.assertEqual(job['failure_category'], 'preflight')

    def test_active_runs_reserve_future_version_disk(self):
        q = self.module('queue')
        self.assertTrue(hasattr(q, 'reserved_disk_gib'), 'active retained-version disk must be reserved')
        with tempfile.TemporaryDirectory() as root:
            reserved = q.reserved_disk_gib([{'output': root, 'profile': {'artifact_gib': 40}}])
            self.assertEqual(reserved, 40)

    def test_throughput_regression_prevents_further_concurrency(self):
        p = self.module('preflight')
        gpu = {'free_mib': 10000, 'utilization': 10, 'temperature': 50, 'throughput_ratio': .85}
        ok, reason = p.admission(gpu, {'peak_mib': 1000, 'artifact_gib': 1}, disk_free_gib=100)
        self.assertFalse(ok)
        self.assertIn('throughput', reason)

    def test_snapshot_can_pin_unrelated_dirty_file_to_committed_bytes(self):
        m = self.module('manifest')
        import inspect
        self.assertIn('committed_overrides', inspect.signature(m.create_snapshot).parameters)
        with tempfile.TemporaryDirectory() as root:
            source = Path(root, 'source'); source.mkdir()
            (source / 'model.py').write_text('unrelated user edit\n')
            target = Path(root, 'snapshot')
            m.create_snapshot(source, target, files=['model.py'], code_commit='a' * 40,
                              committed_overrides={'model.py': b'committed baseline\n'})
            self.assertEqual((target / 'model.py').read_text(), 'committed baseline\n')
            self.assertEqual((source / 'model.py').read_text(), 'unrelated user edit\n')
            m.verify_snapshot(target)

    def test_live_throughput_windows_reduce_admission_cap(self):
        from unittest.mock import patch
        q = self.module('queue')
        with tempfile.TemporaryDirectory() as root:
            a, b = Path(root, 'a'), Path(root, 'b')
            a.mkdir(); b.mkdir()
            jobs = {'a': {'status': 'running', 'gpu_uuid': 'gpu', 'output': str(a)}}
            state = {'jobs': jobs}
            with patch.object(q.time, 'time', return_value=0): q.update_throughput(state, list(jobs.values()))
            (a / 'history.json').write_text('[{}]')
            with patch.object(q.time, 'time', return_value=60): q.update_throughput(state, list(jobs.values()))
            jobs['b'] = {'status': 'running', 'gpu_uuid': 'gpu', 'output': str(b)}
            with patch.object(q.time, 'time', return_value=61): q.update_throughput(state, list(jobs.values()))
            (a / 'history.json').write_text('[{},{}]')
            with patch.object(q.time, 'time', return_value=181): q.update_throughput(state, list(jobs.values()))
            self.assertEqual(state['throughput']['gpu']['concurrency_cap'], 1)

    def test_data_binding_requires_hashed_canonical_labels_and_ignores_inherited_root(self):
        from unittest.mock import patch
        run = self.module('run')
        self.assertTrue(hasattr(run, 'bind_data_environment'))
        m = self.module('manifest')
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            roots = [root / 'CMUMOSI' / 'features' / name for name in ('a', 't', 'v')]
            files = {}
            for directory in roots:
                directory.mkdir(parents=True)
                feature = directory / 'x.npy'; feature.write_bytes(b'feature')
                files[str(feature)] = m.sha(feature)
            label = root / 'CMUMOSI' / 'CMUMOSI_features_raw_2way.pkl'
            label.write_bytes(b'labels and official splits')
            files[str(label)] = m.sha(label)
            manifest = root / 'DATA.json'
            record = {'dataset_root': str(root), 'feature_roots': [str(p) for p in roots],
                      'split_files': [str(label)], 'files': files}
            m.write(manifest, record)
            with patch.dict(os.environ, {'GCNET_DATASET_ROOT': '/wrong'}), patch.dict('sys.modules', {'config': None, 'gcnet.config': None}):
                self.assertEqual(run.bind_data_environment(manifest), tuple(str(p) for p in roots))
                self.assertEqual(os.environ['GCNET_DATASET_ROOT'], str(root))
                self.assertEqual(os.environ.get('GCNET_CACHE_ROOT'), str(root / ('cache_' + m.sha(manifest)[:16])))
                m.write(manifest, dict(record, split_files=[]))
                with self.assertRaises(ValueError): run.bind_data_environment(manifest)
                m.write(manifest, record)
                label.write_bytes(b'changed labels')
                with self.assertRaises(ValueError): run.bind_data_environment(manifest)

    def test_child_rejects_changed_pinned_audit_or_data_manifest(self):
        from types import SimpleNamespace
        run, m = self.module('run'), self.module('manifest')
        self.assertTrue(hasattr(run, 'verify_launch_hashes'))
        with tempfile.TemporaryDirectory() as root:
            fields = {}
            for name in ('manifest', 'baseline_audit', 'data_manifest', 'readiness'):
                path = Path(root, name + '.json'); path.write_text('{}')
                fields[name] = path; fields[name + '_sha256'] = m.sha(path)
            args = SimpleNamespace(**fields)
            run.verify_launch_hashes(args)
            fields['data_manifest'].write_text('{"changed":true}')
            with self.assertRaises(ValueError): run.verify_launch_hashes(args)

    def test_observed_nonzero_exit_cannot_be_promoted(self):
        from unittest.mock import patch
        q = self.module('queue')
        job = {'status': 'running', 'pid': -99, 'output': '/missing', 'process_exit_code': 1}
        with patch.object(q, 'completion_status', return_value=('complete', None)):
            self.assertEqual(q.reconcile_job(job)['status'], 'failed')
            self.assertEqual(q.reconcile_job(dict(job, process_exit_code=0))['status'], 'complete')
            job.pop('process_exit_code')
            self.assertNotEqual(q.reconcile_job(job)['status'], 'complete')

    def test_recoverable_interrupt_uses_complete_state_but_execution_error_does_not(self):
        q = self.module('queue')
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp, 'seed_66'); output.mkdir()
            (output / 'last_training.pt').write_bytes(b'complete-state-fixture')
            marker = Path(tmp, 'seed_66.launch-failure.json')
            job = {'output': str(output), 'run_id': 'x', 'pid': -99, 'status': 'running', 'process_exit_code': 1}
            marker.write_text(json.dumps({'run_id': 'x', 'failure_category': 'interrupted', 'error': 'KeyboardInterrupt'}))
            self.assertEqual(q.reconcile_job(job)['status'], 'interrupted')
            marker.write_text(json.dumps({'run_id': 'x', 'failure_category': 'execution_error', 'error': 'bug'}))
            self.assertEqual(q.reconcile_job(job)['status'], 'failed')

    def test_launch_intent_already_contains_artifact_reservation(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        q, m = self.module('queue'), self.module('manifest')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cards = {'round_id': 'round01', 'status': 'source_accepted', 'cards': self.cards()}
            manifest = root / 'ROUND.json'; m.write(manifest, cards)
            audit = root / 'AUDIT.json'; m.write(audit, {'runs': [{'seed': s, 'mean8': .8} for s in (66, 67, 68)]})
            data = root / 'DATA.json'; m.write(data, {})
            source = root / 'source'; source.mkdir(); (source / 'model.py').write_text('x=1')
            snapshot = root / 'snapshot'; snap = m.create_snapshot(source, snapshot, files=['model.py'], code_commit='a' * 40)
            log = root / 'checks.log'; log.write_text('passed')
            evidence = {'command': ['tests'], 'returncode': 0, 'log': str(log), 'sha256': m.sha(log)}
            profile = {'peak_mib': 1000, 'artifact_gib': 1, 'batch_size': 32, 'train_and_eval': True}
            ready_root = root / 'ready'
            m.write(ready_root / 'candidate_00/READY.json', {'status': 'ready', 'candidate': 'candidate_00',
                'design_sha256': cards['cards'][0]['design_sha256'], 'source_sha256': snap['source_sha256'],
                'snapshot_root': str(snapshot), 'cpu': evidence,
                'cuda': dict(evidence, gpu_index='0', gpu_uuid='GPU-good'), 'profile': profile})
            output = root / 'queue'
            args = SimpleNamespace(root=output, manifest=manifest, baseline_audit=audit, data_manifest=data,
                reference_root=root, readiness_root=ready_root, python='python', max_concurrent=1,
                once=True, poll_seconds=1)
            def launch(*unused, **kwargs):
                intent = m.read(output / 'QUEUE.json')['jobs']['candidate_00:66']
                self.assertEqual(intent.get('profile'), profile)
                return SimpleNamespace(pid=os.getpid(), poll=lambda: None)
            gpu = {'0': {'uuid': 'GPU-good', 'free_mib': 20000, 'utilization': 0, 'temperature': 50},
                   '4': {'uuid': 'GPU-bad', 'free_mib': 20000, 'utilization': 0, 'temperature': 50}}
            with patch.object(q, 'query_gpus', return_value=gpu), patch.object(q.os, 'getloadavg', return_value=(0, 0, 0)), patch.object(q.shutil, 'disk_usage', return_value=SimpleNamespace(free=100 * 1024 ** 3)), patch.object(q.subprocess, 'Popen', side_effect=launch) as popen:
                q.coordinate(args)
                self.assertEqual(popen.call_count, 1)

    def test_data_manifest_builder_covers_features_and_canonical_labels(self):
        m = self.module('manifest')
        self.assertTrue(hasattr(m, 'build_data_manifest'))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('wav2vec-large-c-UTT', 'deberta-large-4-UTT', 'manet_UTT'):
                directory = root / 'CMUMOSI/features' / name; directory.mkdir(parents=True)
                (directory / 'f.npy').write_bytes(b'feature')
            (root / 'CMUMOSI/CMUMOSI_features_raw_2way.pkl').write_bytes(b'labels')
            record = m.build_data_manifest(root)
            self.assertEqual(len(record['files']), 4)
            self.assertEqual(len(record['split_files']), 1)

    def test_dispatch_filter_skips_busy_bundle_without_blocking_normal_candidates(self):
        from inspect import signature
        from types import SimpleNamespace
        from unittest.mock import patch
        q, m = self.module('queue'), self.module('manifest')
        self.assertIn('gpu_filter', signature(q.coordinate).parameters)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, audit, data = (root / name for name in ('ROUND.json', 'AUDIT.json', 'DATA.json'))
            m.write(manifest, dict(round_id='round01', status='source_accepted', cards=self.cards()))
            m.write(audit, {'runs': [{'seed': seed, 'mean8': .8} for seed in (66, 67, 68)]})
            m.write(data, {})
            for candidate in ('candidate_00', 'candidate_01'):
                m.write(root / 'ready' / candidate / 'READY.json', {'snapshot_root': str(root)})
            profile = dict(peak_mib=1000, artifact_gib=1)
            gpus = {index: dict(uuid='GPU-' + index, free_mib=20000, utilization=0, temperature=50)
                    for index in ('0', '3', '4')}
            for window_open, expected, expected_gpu in ((False, 'candidate_01', '0'), (True, 'candidate_00', '3')):
                args = SimpleNamespace(root=root / str(window_open), manifest=manifest, baseline_audit=audit,
                    data_manifest=data, reference_root=root, readiness_root=root / 'ready', python='python',
                    max_concurrent=3, once=True, poll_seconds=1)
                def allowed(candidate, index, actual_args):
                    self.assertIs(actual_args, args)
                    return window_open and index == '3' if candidate == 'candidate_00' else index == '0'
                with patch.object(q, 'query_gpus', return_value=gpus), \
                     patch.object(q, 'verify_snapshot', return_value={'source_sha256': {'model.py': 'hash'}}), \
                     patch.object(q, 'validate_readiness', return_value=profile), \
                     patch.object(q.os, 'getloadavg', return_value=(0, 0, 0)), \
                     patch.object(q.shutil, 'disk_usage', return_value=SimpleNamespace(free=100 * 1024 ** 3)), \
                     patch.object(q.subprocess, 'Popen', return_value=SimpleNamespace(pid=os.getpid())) as popen:
                    q.coordinate(args, gpu_filter=allowed)
                    command = popen.call_args.args[0]
                    self.assertEqual(command[command.index('--candidate') + 1], expected)
                    self.assertEqual(command[command.index('--gpu') + 1], expected_gpu)
                    self.assertEqual(popen.call_count, 1)

    def test_snapshot_includes_only_explicit_required_json_fixtures(self):
        m = self.module('manifest')
        self.assertTrue(hasattr(m, 'REQUIRED_SNAPSHOT_JSON'))
        required = set(m.REQUIRED_SNAPSHOT_JSON)
        self.assertIn('experiments/osram_current_history_relation_20261003/reference/config.json', required)
        self.assertIn('experiments/osram_meaningful20_20261003/ROUND1.json', required)
        self.assertIn('experiments/osram_meaningful20_20261003/BASELINE_AUDIT.json', required)
        for family in m.FAMILIES:
            self.assertIn(f'experiments/osram_meaningful20_20261003/{family}.json', required)
        self.assertFalse(any(name.startswith('dataset/') for name in required))


if __name__ == '__main__':
    unittest.main()
