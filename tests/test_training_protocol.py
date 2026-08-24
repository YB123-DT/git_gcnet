import random
import unittest

import numpy as np
import torch

from train_gcnet import seed_everything


class TrainingProtocolTests(unittest.TestCase):
    def test_seed_controls_python_numpy_and_torch(self):
        seed_everything(2025)
        first = (random.random(), np.random.rand(), torch.rand(3))
        seed_everything(2025)
        second = (random.random(), np.random.rand(), torch.rand(3))

        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
