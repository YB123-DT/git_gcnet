from __future__ import annotations

import numpy as np
import sys
import random
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gcnet"))

import torch

from train_gcnet import random_mask, set_random_seed


class RandomMaskTest(unittest.TestCase):
    def test_seed_controls_python_numpy_and_torch(self) -> None:
        set_random_seed(66)
        first = (random.random(), np.random.rand(), torch.rand(3))
        set_random_seed(66)
        second = (random.random(), np.random.rand(), torch.rand(3))

        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        torch.testing.assert_close(first[2], second[2])

    def test_supports_modern_numpy_and_keeps_one_view(self) -> None:
        for missing_rate in (0.0, 0.1, 0.7):
            with self.subTest(missing_rate=missing_rate):
                np.random.seed(66)
                mask = random_mask(3, 128, missing_rate)

                self.assertEqual(mask.shape, (128, 3))
                self.assertTrue(np.all(np.logical_or(mask == 0, mask == 1)))
                self.assertTrue(np.all(mask.sum(axis=1) >= 1))
                if missing_rate == 0.0:
                    self.assertTrue(np.all(mask == 1))


if __name__ == "__main__":
    unittest.main()
