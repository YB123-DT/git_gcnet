from __future__ import annotations

import gc
import tempfile
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from gcnet_modality_jepa import train_gcnet


def _result(
    fscore,
    total_loss=1.0,
    diagnostics=None,
    vidnames=None,
    artifacts=None,
):
    return (
        0.5,
        fscore,
        ["conversation"] if vidnames is None else vidnames,
        [total_loss, total_loss, 0.0, 0.0, 0.0, 0.0],
        (
            [["predictions"], ["labels"], [], [], ["masks"]]
            if artifacts is None
            else artifacts
        ),
        diagnostics or {},
    )


class _LargeArtifact:
    def __init__(self):
        self.payload = bytearray(1024 * 1024)


class _RecordingSampler:
    def __init__(self):
        self.epochs = []

    def set_epoch(self, epoch):
        self.epochs.append(epoch)


class _Loader:
    def __init__(self):
        self.sampler = _RecordingSampler()


class EvaluationLifecycleTest(unittest.TestCase):
    def make_args(self, epochs=3):
        return SimpleNamespace(
            epochs=epochs,
            seed=70,
            dataset="CMUMOSI",
            epoch_collapse_diagnostics=True,
            evaluation_protocol="strict",
        )

    def test_selects_on_validation_restores_cpu_snapshot_and_tests_once(self):
        model = torch.nn.Linear(1, 1, bias=False)
        train_loader = _Loader()
        val_loader = _Loader()
        test_loader = _Loader()
        calls = []
        validation_scores = [0.2, 0.9, 0.4]
        validation_call_count = 0

        def evaluate(*args, **kwargs):
            nonlocal validation_call_count
            split = kwargs["split"]
            epoch = kwargs["epoch"]
            calls.append((split, epoch, kwargs["compute_diagnostics"]))
            if split == "train":
                with torch.no_grad():
                    model.weight.fill_(epoch + 1)
                return _result(0.1, diagnostics={"train_only": epoch})
            if split == "validation":
                self.assertFalse(any(call[0] == "test" for call in calls))
                score = validation_scores[validation_call_count]
                validation_call_count += 1
                return _result(score)
            self.assertEqual(model.weight.item(), 2.0)
            return _result(0.8, diagnostics={"final": True})

        result = train_gcnet.run_training_fold(
            args=self.make_args(),
            model=model,
            reg_loss=object(),
            cls_loss=object(),
            rec_loss=object(),
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            modality_means=object(),
            mask_rate=0.5,
            optimizer=object(),
            fold=2,
            evaluation_fn=evaluate,
        )

        self.assertEqual(train_loader.sampler.epochs, [0, 1, 2])
        self.assertEqual([call[0] for call in calls].count("train"), 3)
        self.assertEqual([call[0] for call in calls].count("validation"), 3)
        self.assertEqual([call[0] for call in calls].count("test"), 1)
        self.assertEqual(calls[-1], ("test", 0, True))
        self.assertEqual(result["test_call_count"], 1)
        self.assertEqual(result["best_epoch"], 2)
        self.assertEqual(result["best_validation_f1"], 0.9)
        self.assertEqual(result["test_result"][1], 0.8)
        self.assertEqual(len(result["epoch_records"]), 3)
        self.assertTrue(
            all(set(record) >= {"epoch", "train", "validation"}
                for record in result["epoch_records"])
        )
        self.assertTrue(
            all("test" not in record for record in result["epoch_records"])
        )
        for record in result["epoch_records"]:
            for split in ("train", "validation"):
                self.assertEqual(
                    set(record[split]),
                    {"accuracy", "weighted_f1", "loss", "diagnostics"},
                )
                self.assertIsInstance(record[split]["accuracy"], float)
                self.assertIsInstance(record[split]["weighted_f1"], float)
                self.assertEqual(len(record[split]["loss"]), 6)

    def test_passes_epoch_specific_train_and_fixed_evaluation_schedules(self):
        calls = []

        def evaluate(*args, **kwargs):
            calls.append(kwargs)
            return _result(0.5)

        train_gcnet.run_training_fold(
            args=self.make_args(epochs=2),
            model=torch.nn.Linear(1, 1),
            reg_loss=object(),
            cls_loss=object(),
            rec_loss=object(),
            train_loader=_Loader(),
            val_loader=_Loader(),
            test_loader=_Loader(),
            modality_means=object(),
            mask_rate=0.4,
            optimizer=object(),
            fold=4,
            evaluation_fn=evaluate,
        )

        self.assertEqual(
            [(call["split"], call["epoch"]) for call in calls],
            [
                ("train", 0),
                ("validation", 0),
                ("train", 1),
                ("validation", 0),
                ("test", 0),
            ],
        )
        self.assertEqual(calls[0]["mask_schedule"].split, "train")
        self.assertEqual(calls[2]["mask_schedule"].split, "train")
        self.assertIs(calls[0]["mask_schedule"], calls[2]["mask_schedule"])
        self.assertTrue(
            all(call["mask_schedule"].fold == 4 for call in calls)
        )
        self.assertEqual(
            [call["collect_artifacts"] for call in calls],
            [False, False, False, False, True],
        )

    def test_epoch_records_do_not_retain_large_artifact_payloads(self):
        artifact_references = []

        def evaluate(*args, **kwargs):
            if kwargs["split"] == "test":
                return _result(0.8)
            payload = _LargeArtifact()
            artifact_references.append(weakref.ref(payload))
            return _result(
                0.5,
                diagnostics={"compact_metric": 1.0},
                vidnames=[payload],
                artifacts=[[payload], [payload], [payload], [payload], [payload]],
            )

        result = train_gcnet.run_training_fold(
            args=self.make_args(epochs=2),
            model=torch.nn.Linear(1, 1),
            reg_loss=object(),
            cls_loss=object(),
            rec_loss=object(),
            train_loader=_Loader(),
            val_loader=_Loader(),
            test_loader=_Loader(),
            modality_means=object(),
            mask_rate=0.4,
            optimizer=object(),
            fold=1,
            evaluation_fn=evaluate,
        )
        gc.collect()

        self.assertTrue(all(reference() is None for reference in artifact_references))
        for record in result["epoch_records"]:
            for split in ("train", "validation"):
                self.assertEqual(
                    record[split]["diagnostics"], {"compact_metric": 1.0}
                )
                self.assertNotIn("vidnames", record[split])
                self.assertNotIn("artifacts", record[split])

    def test_fold_archive_entry_round_trips_through_existing_consumer(self):
        final_test_payload = {
            "test_labels": [np.array([0, 1])],
            "test_preds": [np.array([[3.0, 0.0], [0.0, 3.0]])],
            "test_hiddens": [],
            "test_names": ["conversation"],
            "test_fmask": [],
        }
        fold_entry = train_gcnet.build_fold_archive_entry(
            best_epoch=4,
            final_test_payload=final_test_payload,
        )

        self.assertEqual(fold_entry[0], 3)
        self.assertIs(fold_entry[-1], final_test_payload)
        self.assertEqual(len(fold_entry), 2)
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "result.npz"
            np.savez_compressed(
                str(archive_path),
                folder_savewhole=np.array([fold_entry], dtype=object),
            )
            with np.load(str(archive_path), allow_pickle=True) as archive:
                consumed_entry = archive["folder_savewhole"][0]
                consumed_best_epoch = int(consumed_entry[0]) + 1
                consumed_payload = consumed_entry[-1]

        self.assertEqual(consumed_best_epoch, 4)
        self.assertIsInstance(consumed_payload, dict)
        self.assertEqual(consumed_payload["test_names"], ["conversation"])

    def test_nonfinite_selection_metric_fails_before_test(self):
        calls = []

        def evaluate(*args, **kwargs):
            calls.append(kwargs["split"])
            if kwargs["split"] == "validation":
                return _result(float("nan"))
            return _result(0.5)

        with self.assertRaisesRegex(ValueError, "validation weighted F1.*finite"):
            train_gcnet.run_training_fold(
                args=self.make_args(epochs=1),
                model=torch.nn.Linear(1, 1),
                reg_loss=object(),
                cls_loss=object(),
                rec_loss=object(),
                train_loader=_Loader(),
                val_loader=_Loader(),
                test_loader=_Loader(),
                modality_means=object(),
                mask_rate=0.4,
                optimizer=object(),
                fold=1,
                evaluation_fn=evaluate,
            )

        self.assertNotIn("test", calls)

    def test_nonfinite_loss_fails_immediately(self):
        calls = []

        def evaluate(*args, **kwargs):
            calls.append(kwargs["split"])
            return _result(0.5, total_loss=float("inf"))

        with self.assertRaisesRegex(ValueError, "train loss.*finite"):
            train_gcnet.run_training_fold(
                args=self.make_args(epochs=1),
                model=torch.nn.Linear(1, 1),
                reg_loss=object(),
                cls_loss=object(),
                rec_loss=object(),
                train_loader=_Loader(),
                val_loader=_Loader(),
                test_loader=_Loader(),
                modality_means=object(),
                mask_rate=0.4,
                optimizer=object(),
                fold=1,
                evaluation_fn=evaluate,
            )

        self.assertEqual(calls, ["train"])


if __name__ == "__main__":
    unittest.main()
