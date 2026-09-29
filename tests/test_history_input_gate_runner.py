import importlib
import tempfile
from pathlib import Path
import unittest
from tests.test_memory_shift_runner import RunnerTest


class HistoryInputGateRunnerTests(unittest.TestCase):
    def test_only_flag_changes_and_gpu0(self):
        self.assertTrue(Path('experiments/osram_cfg84_history_input_gate_20260929/run.py').exists())
        run=importlib.import_module('experiments.osram_cfg84_history_input_gate_20260929.run')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); original=RunnerTest().reference(root)
            self.assertEqual(run.configuration_dict(66,root),dict(original,osram_history_input_gate=True))
            for bad in ('osram_history_input_gate','osram_post_grn'):
                run.write(root/'seed_66/config.json',dict(original,**{bad:True}))
                with self.assertRaises(ValueError):run.configuration_dict(66,root)
        args=run.parser().parse_args(['--preflight'])
        self.assertEqual(args.gpus,['0']);run.validate_args(args)
        args.gpus=['4']
        with self.assertRaises(ValueError):run.validate_args(args)


if __name__=='__main__':unittest.main()
