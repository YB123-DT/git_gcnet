from __future__ import annotations

import hashlib
import random
import unittest

import numpy as np
import torch

from gcnet_modality_jepa.protocol import EpochSeededSubsetSampler, SeedBundle


class SeedBundleTest(unittest.TestCase):
    def test_component_seeds_are_stable_and_independent(self) -> None:
        first = SeedBundle(master_seed=66)
        second = SeedBundle(master_seed=66)

        self.assertEqual(first.derive("model_init"), second.derive("model_init"))
        self.assertNotEqual(first.derive("model_init"), first.derive("data_order"))

    def test_component_seed_is_sha256_derived_and_31_bit(self) -> None:
        payload = b"66:model_init"
        expected = int.from_bytes(
            hashlib.sha256(payload).digest()[:4], byteorder="big"
        ) & 0x7FFFFFFF

        actual = SeedBundle(master_seed=66).derive("model_init")

        self.assertEqual(actual, expected)
        self.assertGreaterEqual(actual, 0)
        self.assertLess(actual, 2**31)


class EpochSeededSubsetSamplerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.python_rng_state = random.getstate()
        self.numpy_rng_state = np.random.get_state()
        self.torch_rng_state = torch.get_rng_state()

    def tearDown(self) -> None:
        random.setstate(self.python_rng_state)
        np.random.set_state(self.numpy_rng_state)
        torch.set_rng_state(self.torch_rng_state)

    def test_same_seed_and_epoch_ignore_unrelated_global_rng_draws(self) -> None:
        sampler = EpochSeededSubsetSampler(indices=range(20), seed=66)
        sampler.set_epoch(3)
        expected = list(sampler)

        random.seed(100)
        np.random.seed(100)
        torch.manual_seed(100)
        for _ in range(100):
            random.random()
            np.random.rand()
            torch.rand(8)

        repeated = EpochSeededSubsetSampler(indices=range(20), seed=66)
        repeated.set_epoch(3)

        self.assertEqual(list(repeated), expected)

    def test_different_epochs_have_different_orders(self) -> None:
        sampler = EpochSeededSubsetSampler(indices=range(20), seed=66)
        sampler.set_epoch(3)
        epoch_three = list(sampler)

        sampler.set_epoch(4)
        epoch_four = list(sampler)

        self.assertNotEqual(epoch_three, epoch_four)
        self.assertEqual(sorted(epoch_three), list(range(20)))
        self.assertEqual(sorted(epoch_four), list(range(20)))

    def test_iteration_does_not_advance_global_torch_rng(self) -> None:
        torch.manual_seed(1234)
        state_before = torch.get_rng_state()

        sampler = EpochSeededSubsetSampler(indices=range(20), seed=66)
        sampler.set_epoch(3)
        list(sampler)

        self.assertTrue(torch.equal(torch.get_rng_state(), state_before))


if __name__ == "__main__":
    unittest.main()
