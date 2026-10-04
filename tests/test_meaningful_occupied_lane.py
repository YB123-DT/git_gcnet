import importlib.util
import unittest
import hashlib
from pathlib import Path
import tempfile


class OccupiedLaneTests(unittest.TestCase):
    def test_deferred_smoke_is_not_reported_as_passed(self):
        from experiments.osram_meaningful20_round2_20261004 import occupied_lane as lane
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'cpu.log'
            log.write_text('tests\nOK\n')
            record = dict(status='cuda_smoke_deferred_by_user', candidate='candidate',
                design_sha256='design', source_sha256={'file': 'hash'},
                cpu=dict(log=str(log), returncode=0, sha256=hashlib.sha256(log.read_bytes()).hexdigest()),
                profile=dict(peak_mib=1700., measurement_status='estimated_not_measured'))
            args = dict(candidate='candidate', design_sha256='design', source_sha256={'file': 'hash'})
            self.assertEqual(lane.validate_deferred(record, **args)['peak_mib'], 1700.)
            with self.assertRaises(ValueError):
                lane.validate_deferred(dict(record, status='ready'), **args)
            log.write_text('failed')
            with self.assertRaises(ValueError):
                lane.validate_deferred(record, **args)

    def test_explicit_twelve_cap_keeps_memory_and_count_limits(self):
        from experiments.osram_meaningful20_round2_20261004 import occupied_lane as lane
        gpu = dict(uuid='GPU-two', free_mib=12000, utilization=90, temperature=55)
        profile = dict(peak_mib=1700, artifact_gib=10)
        jobs = [dict(pid=i, gpu_uuid='GPU-two', profile=profile) for i in range(11)]
        allocations = {('GPU-two', i): 2000 for i in range(11)}
        self.assertTrue(lane.group_admission(gpu, profile, jobs, allocations,
                        max_per_gpu=12, disk_free_gib=200)[0])
        self.assertFalse(lane.group_admission(gpu, profile, jobs + [dict(jobs[0], pid=12)],
                         allocations, max_per_gpu=12, disk_free_gib=200)[0])
        self.assertFalse(lane.group_admission(dict(gpu, free_mib=2000), profile, jobs,
                         allocations, max_per_gpu=12, disk_free_gib=200)[0])
        self.assertTrue(lane.group_admission(dict(gpu, free_mib=10500), profile, jobs,
                        allocations, max_per_gpu=12, disk_free_gib=200, global_reserve_mib=1792)[0])
        self.assertFalse(lane.group_admission(gpu, profile, [], {}, max_per_gpu=4,
                         disk_free_gib=200, global_reserve_mib=1792)[0])
        self.assertTrue(lane.group_admission(gpu, profile, jobs[:10], allocations,
                        max_per_gpu=11, disk_free_gib=200)[0])
        self.assertFalse(lane.group_admission(gpu, profile, jobs, allocations,
                         max_per_gpu=11, disk_free_gib=200)[0])

    def test_two_per_gpu_admits_second_but_rejects_third(self):
        from experiments.osram_meaningful20_round2_20261004 import occupied_lane as lane
        gpu = dict(uuid='GPU-zero', free_mib=7900, utilization=100, temperature=60)
        profile = dict(peak_mib=2000, artifact_gib=10)
        jobs = [dict(pid=10, gpu_uuid='GPU-zero', profile=profile)]
        allocations = {('GPU-zero', 10): 2000}
        self.assertTrue(lane.group_admission(gpu, profile, jobs, allocations,
                        max_per_gpu=2, disk_free_gib=100)[0])
        self.assertFalse(lane.group_admission(gpu, profile, jobs + [dict(jobs[0], pid=11)],
                         allocations, max_per_gpu=2, disk_free_gib=100)[0])
        self.assertFalse(lane.group_admission(dict(gpu, free_mib=3000), profile, jobs,
                         allocations, max_per_gpu=2, disk_free_gib=100)[0])

    def test_busy_override_preserves_all_other_resource_checks(self):
        name = 'experiments.osram_meaningful20_round2_20261004.occupied_lane'
        self.assertIsNotNone(importlib.util.find_spec(name))
        from experiments.osram_meaningful20_round2_20261004.occupied_lane import busy_admission
        gpu = dict(free_mib=6000, utilization=100, temperature=55)
        profile = dict(peak_mib=2000, artifact_gib=20)
        self.assertTrue(busy_admission(gpu, profile, disk_free_gib=100)[0])
        self.assertEqual(gpu['utilization'],100)
        for change in (dict(free_mib=3000),dict(temperature=90),dict(utilization=float('nan'))):
            self.assertFalse(busy_admission(dict(gpu,**change),profile,disk_free_gib=100)[0])
        self.assertFalse(busy_admission(gpu,profile,disk_free_gib=30)[0])

    def test_four_per_gpu_reserves_real_pending_allocations(self):
        from experiments.osram_meaningful20_round2_20261004 import occupied_lane as lane
        self.assertTrue(hasattr(lane,'group_admission'))
        gpu = dict(uuid='GPU-five',free_mib=13000,utilization=100,temperature=55)
        profile = dict(peak_mib=2000,artifact_gib=10)
        jobs = [dict(pid=i,gpu_uuid='GPU-five',profile=profile) for i in (10,11,12)]
        allowed,reason = lane.group_admission(gpu,profile,jobs,{},max_per_gpu=4,disk_free_gib=100)
        self.assertFalse(allowed); self.assertIn('pending',reason.lower())
        allocations = {('GPU-five',pid):2000 for pid in (10,11,12)}
        self.assertTrue(lane.group_admission(gpu,profile,jobs,allocations,max_per_gpu=4,disk_free_gib=100)[0])
        self.assertFalse(lane.group_admission(gpu,profile,jobs+[dict(jobs[0],pid=13)],allocations,max_per_gpu=4,disk_free_gib=100)[0])
        self.assertFalse(lane.group_admission(gpu,profile,jobs,allocations,max_per_gpu=1,disk_free_gib=100)[0])
        self.assertFalse(lane.group_admission(gpu,profile,jobs,allocations,max_per_gpu=4,disk_free_gib=20)[0])
        self.assertEqual(lane.parser().get_default('max_per_gpu'),1)


if __name__ == '__main__': unittest.main()
