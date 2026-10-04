import importlib.util
import unittest


class OccupiedLaneTests(unittest.TestCase):
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
