import importlib
import tempfile
from pathlib import Path
import unittest
from tests.test_memory_shift_runner import RunnerTest


class EvidenceGateRunnerTests(unittest.TestCase):
    def test_only_gate_and_penalty_change(self):
        path = Path('experiments/osram_local_evidence_gate_20260930/run.py')
        self.assertTrue(path.exists())
        run = importlib.import_module('experiments.osram_local_evidence_gate_20260930.run')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = RunnerTest().reference(root)
            self.assertEqual(run.configuration_dict(66, root), dict(original,
                osram_local_evidence_gate=True, osram_local_evidence_gate_reg_weight=.001))
            for flag in ('osram_local_evidence_gate', 'osram_history_input_gate', 'osram_post_grn'):
                run.write(root / 'seed_66/config.json', dict(original, **{flag: True}))
                with self.assertRaises(ValueError):
                    run.configuration_dict(66, root)
        args = run.parser().parse_args(['--preflight'])
        self.assertEqual(args.gpus, ['0'])
        run.validate_args(args)
        args.gpus = ['4']
        with self.assertRaises(ValueError):
            run.validate_args(args)


if __name__ == '__main__':
    unittest.main()
