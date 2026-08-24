import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from mask_bank import (
    batch_mask_from_bank,
    build_mask_bank,
    build_stage_mask_bundle,
    load_or_create_mask_bank,
    load_or_create_stage_mask_bundle,
    mask_bank_sha256,
    select_stage_mask,
    stage_mask_bundle_sha256,
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


class StageAwareMaskBundleTests(unittest.TestCase):
    def setUp(self):
        self.video_ids = {
            "Ses01F_a": [f"u{index}" for index in range(40)],
            "Ses05M_b": [f"v{index}" for index in range(35)],
        }

    def test_bundle_is_reproducible_but_training_epochs_are_distinct(self):
        first = build_stage_mask_bundle(self.video_ids, 0.5, 66, epochs=3)
        second = build_stage_mask_bundle(self.video_ids, 0.5, 66, epochs=3)

        self.assertEqual(
            stage_mask_bundle_sha256(first), stage_mask_bundle_sha256(second)
        )
        for stage in ("validation", "test"):
            for vid in self.video_ids:
                np.testing.assert_array_equal(
                    first[stage][vid], second[stage][vid]
                )
        for epoch in range(3):
            for vid in self.video_ids:
                np.testing.assert_array_equal(
                    first["train"][epoch][vid], second["train"][epoch][vid]
                )

        epoch_zero = np.concatenate(list(first["train"][0].values()), axis=0)
        epoch_one = np.concatenate(list(first["train"][1].values()), axis=0)
        validation = np.concatenate(list(first["validation"].values()), axis=0)
        test = np.concatenate(list(first["test"].values()), axis=0)
        self.assertFalse(np.array_equal(epoch_zero, epoch_one))
        self.assertFalse(np.array_equal(validation, test))

    def test_stage_selection_is_explicit_and_validated(self):
        bundle = build_stage_mask_bundle(self.video_ids, 0.3, 67, epochs=2)

        self.assertIs(select_stage_mask(bundle, "train", epoch=0), bundle["train"][0])
        self.assertIs(select_stage_mask(bundle, "validation"), bundle["validation"])
        self.assertIs(select_stage_mask(bundle, "test"), bundle["test"])
        with self.assertRaisesRegex(ValueError, "epoch"):
            select_stage_mask(bundle, "train")
        with self.assertRaisesRegex(ValueError, "does not accept"):
            select_stage_mask(bundle, "test", epoch=0)

    def test_every_rate_point_seven_constituent_is_one_hot(self):
        bundle = build_stage_mask_bundle(self.video_ids, 0.7, 68, epochs=3)
        banks = list(bundle["train"]) + [bundle["validation"], bundle["test"]]
        for bank in banks:
            rows = np.concatenate(list(bank.values()), axis=0)
            np.testing.assert_array_equal(rows.sum(axis=1), np.ones(len(rows)))
            self.assertAlmostEqual(1.0 - float(rows.mean()), 2.0 / 3.0)

    def test_persisted_bundle_is_reused_with_constituent_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, first_manifest = load_or_create_stage_mask_bundle(
                root, self.video_ids, 0.5, 69, epochs=3
            )
            second, second_manifest = load_or_create_stage_mask_bundle(
                root, self.video_ids, 0.5, 69, epochs=3
            )

            self.assertEqual(first_manifest, second_manifest)
            self.assertEqual(first_manifest["epochs"], 3)
            self.assertEqual(first_manifest["sha256"], stage_mask_bundle_sha256(first))
            self.assertEqual(len(first_manifest["train_sha256"]), 3)
            self.assertEqual(first_manifest["test_sha256"], mask_bank_sha256(first["test"]))
            self.assertEqual(
                stage_mask_bundle_sha256(first), stage_mask_bundle_sha256(second)
            )


if __name__ == "__main__":
    unittest.main()
