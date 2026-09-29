import importlib
import tempfile
from pathlib import Path
import unittest
from tests.test_memory_shift_runner import RunnerTest


class PostGRNRunnerTests(unittest.TestCase):
    def test_only_flag_changes(self):
        self.assertTrue(Path('experiments/osram_cfg84_post_grn_20260929/run.py').is_file())
        run=importlib.import_module('experiments.osram_cfg84_post_grn_20260929.run')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            original=RunnerTest().reference(root)
            self.assertEqual(run.configuration_dict(66,root),dict(original,osram_post_grn=True))
        args=run.parser().parse_args(['--preflight'])
        self.assertEqual(args.gpus,['0'])
        self.assertEqual(str(args.output_root),'/data1/yb/remote_experiments/osram_cfg84_post_grn_20260929/runs')


if __name__=='__main__': unittest.main()
