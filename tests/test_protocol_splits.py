from __future__ import annotations

import dataclasses
import unittest

from gcnet_modality_jepa.splits import (
    SplitIndices,
    build_iemocap_loso_split,
    build_official_split,
)


def synthetic_iemocap_data():
    vids = []
    labels_by_vid = {}
    label_patterns = ([0, 0, 0], [1, 1, 1], [0, 1], [0, 1])
    for session in range(1, 6):
        for conversation, labels in enumerate(label_patterns, start=1):
            vid = "Ses0{}F_impro{:02d}".format(session, conversation)
            vids.append(vid)
            labels_by_vid[vid] = labels
    return vids, labels_by_vid


class IemocapLosoSplitTest(unittest.TestCase):
    def test_held_out_session_is_test_only_and_indices_form_a_partition(self):
        vids, labels_by_vid = synthetic_iemocap_data()

        split = build_iemocap_loso_split(
            vids,
            labels_by_vid,
            test_session=3,
            validation_fraction=0.25,
            seed=66,
        )

        expected_test = tuple(
            index for index, vid in enumerate(vids) if vid.startswith("Ses03")
        )
        self.assertEqual(split.test, expected_test)
        self.assertTrue(split.train)
        self.assertTrue(split.validation)
        self.assertEqual(
            set(split.train) | set(split.validation) | set(split.test),
            set(range(len(vids))),
        )
        self.assertFalse(set(split.train) & set(split.validation))
        self.assertFalse(set(split.train) & set(split.test))
        self.assertFalse(set(split.validation) & set(split.test))
        self.assertTrue(
            all(not vids[index].startswith("Ses03") for index in split.validation)
        )

    def test_split_is_immutable_and_hash_is_stable_sha256(self):
        vids, labels_by_vid = synthetic_iemocap_data()

        first = build_iemocap_loso_split(vids, labels_by_vid, 5, 0.25, 66)
        second = build_iemocap_loso_split(vids, labels_by_vid, 5, 0.25, 66)

        self.assertIsInstance(first, SplitIndices)
        self.assertIsInstance(first.train, tuple)
        self.assertIsInstance(first.validation, tuple)
        self.assertIsInstance(first.test, tuple)
        self.assertEqual(first, second)
        self.assertEqual(first.split_hash, second.split_hash)
        self.assertEqual(len(first.split_hash), 64)
        self.assertTrue(all(character in "0123456789abcdef" for character in first.split_hash))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            first.train = ()

    def test_split_hash_matches_golden_fixture(self):
        vids, labels_by_vid = synthetic_iemocap_data()

        split = build_iemocap_loso_split(vids, labels_by_vid, 5, 0.25, 66)

        self.assertEqual(
            split.split_hash,
            "d34091f93de9b62ad1ea33c2fdb082a59926dea1d0e023669f30bef062602264",
        )

    def test_split_hash_treats_index_order_as_significant(self):
        ordered = SplitIndices(train=(0, 1), validation=(2,), test=(3,))
        reordered = SplitIndices(train=(1, 0), validation=(2,), test=(3,))

        self.assertNotEqual(ordered.split_hash, reordered.split_hash)

    def test_greedy_validation_approximately_matches_remaining_label_distribution(self):
        vids, labels_by_vid = synthetic_iemocap_data()

        split = build_iemocap_loso_split(vids, labels_by_vid, 5, 0.25, 66)

        validation_labels = [
            label
            for index in split.validation
            for label in labels_by_vid[vids[index]]
        ]
        self.assertEqual(validation_labels.count(0), validation_labels.count(1))

    def test_greedy_validation_matches_fixed_seed_fixture(self):
        vids, labels_by_vid = synthetic_iemocap_data()

        split = build_iemocap_loso_split(vids, labels_by_vid, 5, 0.25, 66)

        self.assertEqual(split.validation, (10, 11, 14, 15))
        self.assertEqual(
            tuple(vids[index] for index in split.validation),
            (
                "Ses03F_impro03",
                "Ses03F_impro04",
                "Ses04F_impro03",
                "Ses04F_impro04",
            ),
        )

    def test_different_seeds_can_choose_different_tied_conversations(self):
        vids, labels_by_vid = synthetic_iemocap_data()
        labels_by_vid = {vid: [0, 1] for vid in vids}

        hashes = {
            build_iemocap_loso_split(vids, labels_by_vid, 5, 0.25, seed).split_hash
            for seed in range(10)
        }

        self.assertGreater(len(hashes), 1)

    def test_rejects_malformed_conversation_id(self):
        vids, labels_by_vid = synthetic_iemocap_data()
        labels_by_vid["Session1_bad"] = [0]
        vids[0] = "Session1_bad"

        with self.assertRaisesRegex(ValueError, "malformed IEMOCAP conversation ID"):
            build_iemocap_loso_split(vids, labels_by_vid, 1, 0.25, 66)

    def test_rejects_valid_session_prefix_with_malformed_remainder(self):
        for malformed_vid in ("Ses01", "Ses01garbage", "Ses010F_impro01"):
            with self.subTest(malformed_vid=malformed_vid):
                vids, labels_by_vid = synthetic_iemocap_data()
                original_vid = vids[0]
                vids[0] = malformed_vid
                labels_by_vid[malformed_vid] = labels_by_vid.pop(original_vid)

                with self.assertRaisesRegex(
                    ValueError, "malformed IEMOCAP conversation ID"
                ):
                    build_iemocap_loso_split(vids, labels_by_vid, 1, 0.25, 66)

    def test_accepts_real_impro_and_script_conversation_id_forms(self):
        vids = []
        labels_by_vid = {}
        for session in range(1, 6):
            impro_vid = "Ses0{}F_impro01".format(session)
            script_vid = "Ses0{}M_script01_1".format(session)
            vids.extend((impro_vid, script_vid))
            labels_by_vid[impro_vid] = [0]
            labels_by_vid[script_vid] = [1]

        split = build_iemocap_loso_split(vids, labels_by_vid, 5, 0.25, 66)

        self.assertEqual(set(split.train + split.validation + split.test), set(range(10)))

    def test_rejects_unknown_test_session(self):
        vids, labels_by_vid = synthetic_iemocap_data()

        for test_session in (0, 6):
            with self.subTest(test_session=test_session):
                with self.assertRaisesRegex(ValueError, "test_session"):
                    build_iemocap_loso_split(
                        vids, labels_by_vid, test_session, 0.25, 66
                    )

    def test_rejects_dataset_without_all_five_sessions(self):
        vids, labels_by_vid = synthetic_iemocap_data()
        incomplete_vids = [vid for vid in vids if not vid.startswith("Ses04")]

        with self.assertRaisesRegex(ValueError, "exactly sessions 1 through 5"):
            build_iemocap_loso_split(
                incomplete_vids, labels_by_vid, 5, 0.25, 66
            )

    def test_rejects_missing_or_empty_labels(self):
        vids, labels_by_vid = synthetic_iemocap_data()
        missing_labels = dict(labels_by_vid)
        del missing_labels[vids[0]]
        empty_labels = dict(labels_by_vid)
        empty_labels[vids[0]] = []

        with self.assertRaisesRegex(ValueError, "missing labels"):
            build_iemocap_loso_split(vids, missing_labels, 5, 0.25, 66)
        with self.assertRaisesRegex(ValueError, "missing labels"):
            build_iemocap_loso_split(vids, empty_labels, 5, 0.25, 66)

    def test_rejects_impossible_small_or_invalid_fraction_splits(self):
        vids = ["Ses0{}F_impro01".format(session) for session in range(1, 6)]
        labels_by_vid = {vid: [0] for vid in vids}

        with self.assertRaisesRegex(ValueError, "nonempty train and validation"):
            build_iemocap_loso_split(vids, labels_by_vid, 1, 0.1, 66)
        for validation_fraction in (0.0, 1.0, -0.1, 1.1):
            with self.subTest(validation_fraction=validation_fraction):
                with self.assertRaisesRegex(ValueError, "validation_fraction"):
                    build_iemocap_loso_split(
                        vids, labels_by_vid, 1, validation_fraction, 66
                    )


class OfficialSplitTest(unittest.TestCase):
    def test_preserves_official_video_sets_exactly(self):
        vids = ["train-b", "test-a", "validation-a", "train-a"]

        split = build_official_split(
            vids,
            train_vids={"train-a", "train-b"},
            validation_vids={"validation-a"},
            test_vids={"test-a"},
        )

        self.assertEqual(split.train, (0, 3))
        self.assertEqual(split.validation, (2,))
        self.assertEqual(split.test, (1,))

    def test_rejects_official_overlap(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            build_official_split(
                ["a", "b", "c"],
                train_vids={"a", "b"},
                validation_vids={"b"},
                test_vids={"c"},
            )

    def test_rejects_missing_or_unknown_official_video_ids(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            build_official_split(
                ["a", "b", "c", "unassigned"],
                train_vids={"a"},
                validation_vids={"b"},
                test_vids={"c"},
            )
        with self.assertRaisesRegex(ValueError, "unknown"):
            build_official_split(
                ["a", "b", "c"],
                train_vids={"a", "unknown"},
                validation_vids={"b"},
                test_vids={"c"},
            )


if __name__ == "__main__":
    unittest.main()
