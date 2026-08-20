from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch.utils.data import Dataset

from gcnet_modality_jepa import train_gcnet


def _result(fscore: float, marker: str = ""):
    diagnostics = {
        "primary_mask": {"realized_missing_rate": 0.1},
        "marker": marker,
    }
    return (
        0.5,
        fscore,
        [marker],
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [[marker], [marker], [], [], []],
        diagnostics,
    )


class _RecordingSampler:
    def __init__(self):
        self.epochs = []

    def set_epoch(self, epoch: int) -> None:
        self.epochs.append(epoch)


class _Loader:
    def __init__(self):
        self.sampler = _RecordingSampler()


class _SyntheticIemocap(Dataset):
    def __init__(self):
        self.vids = [
            "Ses0{}F_impro{:02d}".format(session, conversation)
            for session in range(1, 6)
            for conversation in range(1, 4)
        ]
        self.videoLabelsNew = {vid: [0] for vid in self.vids}

    def __len__(self):
        return len(self.vids)

    def __getitem__(self, index):
        return index

    @staticmethod
    def collate_fn(items):
        return items

    @staticmethod
    def get_featDim():
        return 2, 3, 4


class OfficialEvaluationProtocolTest(unittest.TestCase):
    def test_cli_defaults_to_official_protocol(self):
        args = train_gcnet.build_argument_parser().parse_args([])

        self.assertEqual(args.evaluation_protocol, "official")

    def test_iemocap_programmatic_default_uses_held_out_session_for_validation_and_test(self):
        dataset = _SyntheticIemocap()
        with mock.patch.object(
            train_gcnet, "load_iemocap_dataset", return_value=dataset
        ):
            train_loaders, val_loaders, test_loaders, *_ = train_gcnet.get_loaders(
                "audio",
                "text",
                "video",
                num_folder=5,
                dataset="IEMOCAPSix",
                batch_size=4,
                num_workers=0,
                seed=66,
            )

        self.assertEqual(len(train_loaders), 5)
        for fold, (train_loader, val_loader, test_loader) in enumerate(
            zip(train_loaders, val_loaders, test_loaders), start=1
        ):
            train_indices = set(train_loader.sampler.indices)
            validation_indices = set(val_loader.sampler.indices)
            test_indices = set(test_loader.sampler.indices)
            self.assertEqual(validation_indices, test_indices)
            self.assertFalse(train_indices & test_indices)
            self.assertEqual(
                train_indices | test_indices, set(range(len(dataset)))
            )
            self.assertTrue(
                all(
                    dataset.vids[index].startswith("Ses0{}".format(fold))
                    for index in test_indices
                )
            )
            self.assertIsNot(val_loader, test_loader)
            self.assertEqual(
                train_loader.protocol_metadata["evaluation_protocol"],
                "official",
            )

    def test_programmatic_lifecycle_default_tests_every_epoch_and_selects_by_validation(self):
        args = SimpleNamespace(
            epochs=3,
            seed=66,
            dataset="CMUMOSI",
        )
        train_loader = _Loader()
        val_loader = _Loader()
        test_loader = _Loader()
        validation_scores = [0.2, 0.9, 0.4]
        test_scores = [0.99, 0.3, 0.8]
        validation_index = 0
        test_index = 0
        calls = []

        def evaluate(*unused_args, **kwargs):
            nonlocal validation_index, test_index
            split = kwargs["split"]
            epoch = kwargs["epoch"]
            calls.append((split, epoch, kwargs["collect_artifacts"]))
            if split == "train":
                return _result(0.1, "train-{}".format(epoch))
            if split == "validation":
                score = validation_scores[validation_index]
                validation_index += 1
                return _result(score, "validation-{}".format(epoch))
            score = test_scores[test_index]
            test_index += 1
            return _result(score, "test-{}".format(epoch))

        lifecycle = train_gcnet.run_training_fold(
            args=args,
            model=torch.nn.Linear(1, 1),
            reg_loss=object(),
            cls_loss=object(),
            rec_loss=object(),
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            modality_means=object(),
            mask_rate=0.1,
            optimizer=object(),
            fold=1,
            evaluation_fn=evaluate,
        )

        self.assertEqual(
            [(split, epoch) for split, epoch, _ in calls],
            [
                ("train", 0),
                ("validation", 0),
                ("test", 0),
                ("train", 1),
                ("validation", 1),
                ("test", 1),
                ("train", 2),
                ("validation", 2),
                ("test", 2),
            ],
        )
        self.assertEqual(train_loader.sampler.epochs, [0, 1, 2])
        self.assertEqual(val_loader.sampler.epochs, [0, 1, 2])
        self.assertEqual(test_loader.sampler.epochs, [0, 1, 2])
        self.assertEqual(lifecycle["best_epoch"], 2)
        self.assertEqual(lifecycle["best_validation_f1"], 0.9)
        self.assertEqual(lifecycle["test_result"][1], 0.3)
        self.assertEqual(lifecycle["test_result"][2], ["test-1"])
        self.assertEqual(lifecycle["test_call_count"], 3)
        self.assertTrue(all("test" in record for record in lifecycle["epoch_records"]))

    def test_programmatic_mask_default_varies_evaluation_masks_by_epoch(self):
        args = SimpleNamespace(dataset="CMUMOSI", seed=66)

        schedule = train_gcnet.build_mask_schedule(
            args=args,
            split="validation",
            fold=1,
            mask_rate=0.5,
        )
        first = schedule.generate(
            "conversation-a", length=256, side="host", epoch=0
        )
        second = schedule.generate(
            "conversation-a", length=256, side="host", epoch=1
        )

        self.assertFalse(schedule.freeze_evaluation)
        self.assertEqual((first.epoch, second.epoch), (0, 1))
        self.assertNotEqual(first.schedule_hash, second.schedule_hash)

    def test_lifecycle_manifest_evidence_defaults_to_official(self):
        def compact_result(marker):
            return {
                "diagnostics": {
                    "primary_mask": {"realized_missing_rate": marker}
                }
            }

        lifecycle = {
            "best_epoch": 2,
            "best_validation_f1": 0.9,
            "test_call_count": 2,
            "test_result": _result(0.8),
            "mask_schedule_hashes": {
                "train": "1" * 64,
                "validation": "2" * 64,
                "test": "3" * 64,
            },
            "epoch_records": [
                {
                    "train": compact_result(0.10),
                    "validation": compact_result(0.20),
                    "test": compact_result(0.30),
                },
                {
                    "train": compact_result(0.11),
                    "validation": compact_result(0.20),
                    "test": compact_result(0.31),
                },
            ],
        }

        evidence = train_gcnet.lifecycle_manifest_evidence(lifecycle)

        self.assertEqual(evidence["evaluation_protocol"], "official")
        self.assertEqual(evidence["epochs_completed"], 2)
        self.assertEqual(
            evidence["realized_missing_rates"]["validation"], [0.20, 0.20]
        )
        self.assertEqual(
            evidence["realized_missing_rates"]["test"], [0.30, 0.31]
        )


if __name__ == "__main__":
    unittest.main()
