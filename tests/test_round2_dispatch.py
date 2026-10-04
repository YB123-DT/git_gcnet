import importlib.util
import unittest
from unittest.mock import patch
from pathlib import Path


class DispatchTests(unittest.TestCase):
    def test_bundle_is_not_occupancy_or_foreign_process_preemption(self):
        self.assertIsNotNone(importlib.util.find_spec('experiments.osram_meaningful20_round2_20261004.dispatch'))
        from experiments.osram_meaningful20_round2_20261004 import dispatch as d
        policy={'bundle_ids':['bundle'],'other_ids':['other'],'bundle_gpu_index':'3',
                'bundle_gpu_uuid':'GPU-three','other_gpu_indices':['0','1','2','5','6','7']}
        with patch.object(d,'compute_processes',return_value=[('GPU-three',42)]), \
                patch.object(d,'own_experiment',return_value=False):
            self.assertFalse(d.allowed_gpu(policy,'bundle','3',Path('/runs/ours')))
            self.assertTrue(d.allowed_gpu(policy,'other','0',Path('/runs/ours')))
        with patch.object(d,'compute_processes',return_value=[]):
            self.assertTrue(d.allowed_gpu(policy,'bundle','3',Path('/runs/ours')))
        with patch.object(d,'compute_processes',return_value=[('GPU-three',42)]), \
                patch.object(d,'own_experiment',return_value=True):
            self.assertTrue(d.allowed_gpu(policy,'bundle','3',Path('/runs/ours')))
        for name,index in [('bundle','0'),('other','3'),('other','4'),('unknown','3')]:
            self.assertFalse(d.allowed_gpu(policy,name,index,Path('/runs/ours')))


if __name__=='__main__': unittest.main()
