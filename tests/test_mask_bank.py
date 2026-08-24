import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from mask_bank import (
    batch_mask_from_bank,
    build_mask_bank,
    load_or_create_mask_bank,
    mask_bank_sha256,
)


class FixedMaskBankTests(unittest.TestCase):
    def setUp(self):
        self.video_ids = {
            "Ses01F_a": ["u0", "u1", "u2"],
            "Ses05M_b": ["u3", "u4"],
        }

    def test_same_seed_and_rate_are_entrywise_identical(self):
        first = build_mask_bank(self.video_ids, 0.3, 66)
        second = build_mask_bank(self.video_ids, 0.3, 66)

        self.assertEqual(first.keys(), second.keys())
        for vid in first:
            np.testing.assert_array_equal(first[vid], second[vid])
        self.assertEqual(mask_bank_sha256(first), mask_bank_sha256(second))

    def test_rate_point_seven_contains_only_one_hot_patterns(self):
        bank = build_mask_bank(self.video_ids, 0.7, 66)
        rows = np.concatenate(list(bank.values()), axis=0)

        np.testing.assert_array_equal(rows.sum(axis=1), np.ones(len(rows)))
        self.assertTrue(np.isin(rows, (0, 1)).all())

    def test_every_valid_utterance_keeps_a_modality(self):
        for rate in (0.0, 0.1, 0.3, 0.5, 0.7):
            with self.subTest(rate=rate):
                bank = build_mask_bank(self.video_ids, rate, 68)
                rows = np.concatenate(list(bank.values()), axis=0)
                self.assertTrue((rows.sum(axis=1) >= 1).all())

    def test_batch_reconstruction_preserves_conversation_order(self):
        bank = {
            "Ses01F_a": np.array(
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8
            ),
            "Ses05M_b": np.array([[1, 1, 0], [1, 0, 1]], dtype=np.uint8),
        }

        actual = batch_mask_from_bank(
            bank, ["Ses05M_b", "Ses01F_a"], max_length=3
        )

        self.assertEqual(tuple(actual.shape), (3, 2, 3))
        torch.testing.assert_close(
            actual[:2, 0],
            torch.tensor([[1, 1, 0], [1, 0, 1]], dtype=torch.uint8),
        )
        torch.testing.assert_close(actual[:, 1], torch.tensor(bank["Ses01F_a"]))
        torch.testing.assert_close(actual[2, 0], torch.ones(3, dtype=torch.uint8))

    def test_saved_bank_is_reused_across_model_arms(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, first_manifest = load_or_create_mask_bank(
                root, self.video_ids, 0.5, 2025
            )
            second, second_manifest = load_or_create_mask_bank(
                root, self.video_ids, 0.5, 2025
            )

            self.assertEqual(first_manifest["sha256"], second_manifest["sha256"])
            self.assertEqual(first_manifest["requested_missing_rate"], 0.5)
            self.assertIn("realized_missing_rate", first_manifest)
            for vid in first:
                np.testing.assert_array_equal(first[vid], second[vid])


if __name__ == "__main__":
    unittest.main()
