from __future__ import annotations

import random
import unittest

import numpy as np
import torch

from gcnet_modality_jepa.mask_schedule import ConversationMaskSchedule


def assert_numpy_rng_states_equal(
    testcase: unittest.TestCase, before: tuple, after: tuple
) -> None:
    testcase.assertEqual(before[0], after[0])
    testcase.assertTrue(np.array_equal(before[1], after[1]))
    testcase.assertEqual(before[2:], after[2:])


class ConversationMaskScheduleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.python_rng_state = random.getstate()
        self.numpy_rng_state = np.random.get_state()
        self.torch_rng_state = torch.get_rng_state()

    def tearDown(self) -> None:
        random.setstate(self.python_rng_state)
        np.random.set_state(self.numpy_rng_state)
        torch.set_rng_state(self.torch_rng_state)

    def make_schedule(
        self,
        split: str = "train",
        requested_missing_rate: float = 0.4,
        mask_seed: int = 66,
        freeze_evaluation: bool = True,
    ) -> ConversationMaskSchedule:
        return ConversationMaskSchedule(
            dataset="CMUMOSI",
            split=split,
            fold=2,
            requested_missing_rate=requested_missing_rate,
            mask_seed=mask_seed,
            freeze_evaluation=freeze_evaluation,
        )

    def test_same_conversation_is_repeatable_and_order_independent(self) -> None:
        schedule = self.make_schedule()

        expected_a = schedule.generate("conversation-a", 128, "host", epoch=3)
        expected_b = schedule.generate("conversation-b", 128, "host", epoch=3)

        repeated_b = schedule.generate("conversation-b", 128, "host", epoch=3)
        repeated_a = schedule.generate("conversation-a", 128, "host", epoch=3)

        self.assertTrue(
            np.array_equal(expected_a.availability, repeated_a.availability)
        )
        self.assertTrue(
            np.array_equal(expected_b.availability, repeated_b.availability)
        )
        self.assertEqual(expected_a.schedule_hash, repeated_a.schedule_hash)
        self.assertEqual(expected_b.schedule_hash, repeated_b.schedule_hash)

    def test_training_epoch_changes_masks(self) -> None:
        schedule = self.make_schedule()

        epoch_one = schedule.generate("conversation-a", 256, "host", epoch=1)
        epoch_two = schedule.generate("conversation-a", 256, "host", epoch=2)

        self.assertFalse(
            np.array_equal(epoch_one.availability, epoch_two.availability)
        )
        self.assertNotEqual(epoch_one.schedule_hash, epoch_two.schedule_hash)

    def test_validation_and_test_force_epoch_zero(self) -> None:
        for split in ("validation", "test"):
            with self.subTest(split=split):
                schedule = self.make_schedule(split=split)

                epoch_zero = schedule.generate(
                    "conversation-a", 128, "guest", epoch=0
                )
                later_epoch = schedule.generate(
                    "conversation-a", 128, "guest", epoch=99
                )

                self.assertTrue(
                    np.array_equal(
                        epoch_zero.availability, later_epoch.availability
                    )
                )
                self.assertEqual(epoch_zero.schedule_hash, later_epoch.schedule_hash)
                self.assertEqual(later_epoch.epoch, 0)

    def test_official_evaluation_schedule_changes_deterministically_by_epoch(self) -> None:
        schedule = self.make_schedule(
            split="validation", freeze_evaluation=False
        )

        epoch_one = schedule.generate(
            "conversation-a", 256, "guest", epoch=1
        )
        epoch_two = schedule.generate(
            "conversation-a", 256, "guest", epoch=2
        )
        repeated = schedule.generate(
            "conversation-a", 256, "guest", epoch=2
        )

        self.assertFalse(
            np.array_equal(epoch_one.availability, epoch_two.availability)
        )
        self.assertTrue(
            np.array_equal(epoch_two.availability, repeated.availability)
        )
        self.assertEqual(epoch_two.epoch, 2)
        self.assertNotEqual(epoch_one.schedule_hash, epoch_two.schedule_hash)

    def test_split_aliases_and_case_share_canonical_keys_and_hashes(self) -> None:
        schedules = [
            self.make_schedule(split=split)
            for split in ("val", "VAL", "validation", "Validation")
        ]
        results = [
            schedule.generate("conversation-a", 128, "guest", epoch=7)
            for schedule in schedules
        ]

        self.assertTrue(all(schedule.split == "validation" for schedule in schedules))
        self.assertEqual(len({schedule.config_hash for schedule in schedules}), 1)
        self.assertEqual(len({result.schedule_hash for result in results}), 1)
        for result in results[1:]:
            self.assertTrue(
                np.array_equal(results[0].availability, result.availability)
            )

    def test_split_names_are_canonical_and_unknown_splits_are_rejected(self) -> None:
        for supplied, canonical in (
            ("TRAIN", "train"),
            ("val", "validation"),
            ("VALIDATION", "validation"),
            ("Test", "test"),
        ):
            with self.subTest(supplied=supplied):
                self.assertEqual(self.make_schedule(split=supplied).split, canonical)

        for split in ("", "dev", "testing", "train-validation"):
            with self.subTest(split=split):
                with self.assertRaisesRegex(ValueError, "split"):
                    self.make_schedule(split=split)

    def test_supported_rates_produce_binary_length_by_three_masks(self) -> None:
        for rate in (0.0, 0.1, 0.4, 0.7):
            with self.subTest(rate=rate):
                result = self.make_schedule(
                    requested_missing_rate=rate
                ).generate("conversation-a", 257, "host", epoch=4)

                self.assertEqual(result.availability.shape, (257, 3))
                self.assertEqual(result.availability.dtype, np.uint8)
                self.assertTrue(
                    np.all(
                        np.logical_or(
                            result.availability == 0,
                            result.availability == 1,
                        )
                    )
                )
                self.assertTrue(np.all(result.availability.sum(axis=1) >= 1))
                if rate == 0.0:
                    self.assertTrue(np.all(result.availability == 1))

    def test_padding_is_marked_separately_and_excluded_from_rates(self) -> None:
        result = self.make_schedule().generate(
            "conversation-a", length=8, side="host", epoch=1, valid_length=5
        )

        self.assertTrue(np.array_equal(result.valid_utterance_mask[:5], np.ones(5)))
        self.assertTrue(np.array_equal(result.valid_utterance_mask[5:], np.zeros(3)))
        self.assertTrue(np.all(result.availability[:5].sum(axis=1) >= 1))
        self.assertTrue(np.all(result.availability[5:] == 0))
        expected_rate = 1.0 - float(result.availability[:5].mean())
        self.assertAlmostEqual(result.realized_missing_rate, expected_rate)

    def test_host_and_guest_use_independent_conversation_keys(self) -> None:
        schedule = self.make_schedule()

        host = schedule.generate("conversation-a", 256, "host", epoch=3)
        guest = schedule.generate("conversation-a", 256, "guest", epoch=3)

        self.assertFalse(np.array_equal(host.availability, guest.availability))
        self.assertNotEqual(host.schedule_hash, guest.schedule_hash)

    def test_generation_does_not_mutate_global_rng_states(self) -> None:
        random.seed(101)
        np.random.seed(202)
        torch.manual_seed(303)
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        torch_before = torch.get_rng_state().clone()

        self.make_schedule().generate(
            "conversation-a", 128, "host", epoch=5
        )

        self.assertEqual(random.getstate(), python_before)
        assert_numpy_rng_states_equal(self, np.random.get_state(), numpy_before)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_before))

    def test_requested_and_realized_missing_rates_are_reported(self) -> None:
        result = self.make_schedule(requested_missing_rate=0.7).generate(
            "conversation-a", 257, "host", epoch=4
        )

        expected_realized = 1.0 - float(result.availability.mean())
        self.assertEqual(result.requested_missing_rate, 0.7)
        self.assertAlmostEqual(result.realized_missing_rate, expected_realized)
        self.assertGreaterEqual(result.realized_missing_rate, 0.0)
        self.assertLessEqual(result.realized_missing_rate, 2.0 / 3.0)

    def test_config_and_schedule_hashes_are_stable_and_condition_specific(self) -> None:
        first = self.make_schedule()
        repeated = self.make_schedule()
        changed_rate = self.make_schedule(requested_missing_rate=0.5)
        changed_seed = self.make_schedule(mask_seed=67)

        first_result = first.generate("conversation-a", 128, "host", epoch=3)
        repeated_result = repeated.generate(
            "conversation-a", 128, "host", epoch=3
        )

        self.assertEqual(first.config_hash, repeated.config_hash)
        self.assertEqual(first_result.schedule_hash, repeated_result.schedule_hash)
        self.assertEqual(len(first.config_hash), 64)
        self.assertEqual(len(first_result.schedule_hash), 64)
        self.assertNotEqual(first.config_hash, changed_rate.config_hash)
        self.assertNotEqual(first.config_hash, changed_seed.config_hash)

    def test_hashes_match_golden_fixture(self) -> None:
        schedule = self.make_schedule(split="validation")
        result = schedule.generate(
            "golden-conversation", 8, "host", epoch=99
        )

        self.assertEqual(
            schedule.config_hash,
            "7eee6b89ec3ea2d960d016a969a229feb4bf34bdff3190830ba9e187a98c6e2e",
        )
        self.assertEqual(
            result.schedule_hash,
            "927f8e38e0f3507966e890bfc558af24ebf73d5858d46348580e7709bd377882",
        )

    def test_rejects_rates_outside_supported_range(self) -> None:
        for rate in (-0.01, 0.700001, 1.0, float("nan")):
            with self.subTest(rate=rate):
                with self.assertRaises(ValueError):
                    self.make_schedule(requested_missing_rate=rate)

    def test_rejects_zero_and_negative_lengths(self) -> None:
        schedule = self.make_schedule()

        for length in (0, -1):
            with self.subTest(length=length):
                with self.assertRaisesRegex(ValueError, "length must be positive"):
                    schedule.generate("conversation-a", length, "host", epoch=0)

    def test_rejects_negative_epoch(self) -> None:
        with self.assertRaisesRegex(ValueError, "epoch must be non-negative"):
            self.make_schedule().generate(
                "conversation-a", 4, "host", epoch=-1
            )

    def test_rejects_invalid_valid_length(self) -> None:
        schedule = self.make_schedule()

        for valid_length in (0, -1, 5, 1.5):
            with self.subTest(valid_length=valid_length):
                with self.assertRaisesRegex(
                    ValueError, "valid_length must be between one and length"
                ):
                    schedule.generate(
                        "conversation-a",
                        length=4,
                        side="host",
                        valid_length=valid_length,
                    )


if __name__ == "__main__":
    unittest.main()
