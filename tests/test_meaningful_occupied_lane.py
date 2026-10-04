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


if __name__ == '__main__': unittest.main()
