import random
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from train_gcnet import seed_everything


class TrainingProtocolTests(unittest.TestCase):
    def test_seed_supports_torch_without_warn_only_keyword(self):
        calls = []

        def old_torch_api(enabled, *args, **kwargs):
            calls.append((enabled, args, kwargs))
            if kwargs:
                raise TypeError("unexpected keyword argument 'warn_only'")

        with mock.patch.object(
            torch, "use_deterministic_algorithms", side_effect=old_torch_api
        ):
            seed_everything(66)

        self.assertEqual(
            calls,
            [
                (True, (), {"warn_only": False}),
                (True, (), {}),
            ],
        )

    def test_cli_accepts_cp_lecc_graph_convolution_variant(self):
        project_root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [sys.executable, "train_gcnet.py", "--help"],
            cwd=project_root / "gcnet",
            capture_output=True,
            text=True,
            check=True,
        )

        self.assertIn("cp_lecc", completed.stdout)

    def test_seed_controls_python_numpy_and_torch(self):
        enabled = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        cudnn_deterministic = torch.backends.cudnn.deterministic
        cudnn_benchmark = torch.backends.cudnn.benchmark
        try:
            seed_everything(2025)
            first = (random.random(), np.random.rand(), torch.rand(3))
            seed_everything(2025)
            second = (random.random(), np.random.rand(), torch.rand(3))

            self.assertEqual(first[0], second[0])
            self.assertEqual(first[1], second[1])
            torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)
        finally:
            torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
            torch.backends.cudnn.deterministic = cudnn_deterministic
            torch.backends.cudnn.benchmark = cudnn_benchmark

    def test_seed_enables_strict_torch_determinism(self):
        enabled = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        cudnn_deterministic = torch.backends.cudnn.deterministic
        cudnn_benchmark = torch.backends.cudnn.benchmark
        try:
            seed_everything(66)

            self.assertTrue(torch.are_deterministic_algorithms_enabled())
            self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())
            self.assertTrue(torch.backends.cudnn.deterministic)
            self.assertFalse(torch.backends.cudnn.benchmark)
        finally:
            torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
            torch.backends.cudnn.deterministic = cudnn_deterministic
            torch.backends.cudnn.benchmark = cudnn_benchmark


if __name__ == "__main__":
    unittest.main()
