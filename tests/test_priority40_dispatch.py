import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class PriorityDispatcherTests(unittest.TestCase):
    def test_census_unions_external_cuda_child_and_initializing_parent_without_double_count(self):
        from experiments.osram_priority40_20261004 import dispatch
        gpu = dispatch.GPUS['2']
        with tempfile.TemporaryDirectory() as directory:
            known = dict(pid=1, candidate='m01_test', gpu_uuid=gpu, output=directory,
                         profile=dict(peak_mib=1700, artifact_gib=10))
            reservations = [dict(pid=10, candidate='m01_test', gpu_uuid=gpu, output=directory),
                            dict(pid=11, candidate='m02_test', gpu_uuid=gpu, output=directory+'/second')]
            high_water = {}
            with patch.object(dispatch.subprocess, 'check_output', return_value=f'{gpu}, 1, 3000\n{gpu}, 9, 1000\n'), \
                 patch.object(dispatch, 'queue_jobs', return_value=[known]), \
                 patch.object(dispatch, 'identity', return_value='1'), \
                 patch.object(dispatch.Path, 'iterdir', return_value=iter([])):
                actors, reserved = dispatch.census([], reservations, high_water, 2000)
            self.assertEqual({actor['pid'] for actor in actors}, {1,9,11})
            self.assertEqual(reserved, 30)  # known child, pending child, unknown CUDA process
            self.assertEqual(next(actor['estimate_mib'] for actor in actors if actor['pid']==1), 3000)

    def test_crash_before_pid_write_stops_new_submissions(self):
        from experiments.osram_priority40_20261004 import dispatch
        with tempfile.TemporaryDirectory() as directory:
            state = {'jobs': {'m01_test': dict(status='launch_intent', lane_root=directory)}}
            dispatch.reconcile(state, {})
            self.assertEqual(state['jobs']['m01_test']['status'], 'inspection_pending')

    def test_global_jobs_and_initializing_reservations_count_toward_eleven(self):
        from experiments.osram_priority40_20261004.dispatch import admission
        gpu = dict(uuid='gpu', free_mib=30000, temperature=55)
        jobs = [dict(gpu_uuid='gpu', pid=i, used_mib=1000, estimate_mib=2000)
                for i in range(11)]
        self.assertFalse(admission(gpu, jobs, 2000, 100, 0)[0])
        self.assertTrue(admission(gpu, jobs[:10], 2000, 100, 0)[0])
        jobs[-1]['used_mib'] = 0  # child not yet in nvidia-smi
        self.assertFalse(admission(gpu, jobs, 2000, 100, 0)[0])

    def test_live_growth_and_disk_margin_not_forty_future_reservations(self):
        from experiments.osram_priority40_20261004.dispatch import admission
        gpu = dict(uuid='gpu', free_mib=4000, temperature=55)
        jobs = [dict(gpu_uuid='gpu', pid=1, used_mib=8110, estimate_mib=1700)]
        self.assertFalse(admission(gpu, jobs, 2000, 100, 0)[0])
        gpu['free_mib'] = 20000
        self.assertTrue(admission(gpu, jobs, 2000, 40, 5)[0])
        self.assertFalse(admission(gpu, jobs, 2000, 30, 5)[0])

    def test_only_requested_forty_in_order(self):
        from experiments.osram_priority40_20261004.dispatch import load_candidates
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for group in range(2):
                path = Path(directory) / f'{group}.json'
                path.write_text(json.dumps({'cards': [{'id': f'm{i:02d}_test'}
                    for i in range(group * 20 + 1, group * 20 + 21)]}))
                paths.append(path)
            self.assertEqual(len(load_candidates(paths)), 40)
            bad = json.loads(paths[1].read_text())
            bad['cards'][-1]['id'] = 'v3_extra'
            paths[1].write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                load_candidates(paths)

    def test_only_precreation_admission_failure_may_wait_again(self):
        from experiments.osram_priority40_20261004.dispatch import retryable_admission
        self.assertTrue(retryable_admission('Occupied-lane admission rejected before job creation: memory', False))
        self.assertFalse(retryable_admission('CUDA out of memory', False))
        self.assertFalse(retryable_admission('Occupied-lane admission rejected before job creation: memory', True))


if __name__ == '__main__':
    unittest.main()
