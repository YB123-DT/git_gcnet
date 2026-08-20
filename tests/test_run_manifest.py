from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from gcnet_modality_jepa.run_manifest import (
    MANIFEST_NAME,
    MANIFEST_VERSION,
    ManifestValidationError,
    audit_paired_manifests,
    feature_metadata_hash,
    load_manifest,
    sampler_signature,
    validate_manifest,
    write_manifest_atomic,
)


def complete_manifest() -> dict:
    return {
        "schema": {"name": MANIFEST_NAME, "version": MANIFEST_VERSION},
        "run": {
            "dataset": "CMUMOSI",
            "fold": 1,
            "master_seed": 66,
        },
        "environment": {
            "python": "3.8.20",
            "torch": "1.8.0",
            "cuda": "10.2",
            "cudnn": 7605,
            "pyg": "2.0.1",
            "numpy": "1.21.6",
            "sklearn": "1.0.2",
            "gpu": {
                "index": 0,
                "model": "Tesla V100-SXM2-32GB",
                "driver": "575.51.03",
            },
        },
        "provenance": {
            "command": ["python", "-m", "gcnet_modality_jepa.train_gcnet"],
            "cwd": "/experiment",
            "git_revision": "abc123",
            "git_status": "clean",
        },
        "features": {
            "audio": {"path": "/features/audio", "metadata_sha256": "a" * 64},
            "text": {"path": "/features/text", "metadata_sha256": "b" * 64},
            "visual": {"path": "/features/visual", "metadata_sha256": "c" * 64},
        },
        "split": {
            "indices": {
                "train": [0, 1],
                "validation": [2],
                "test": [3],
            },
            "hash": "d" * 64,
        },
        "samplers": {
            "train": {"seed": 101, "signature": "e" * 64},
            "validation": {"seed": 102, "signature": "f" * 64},
            "test": {"seed": 103, "signature": "0" * 64},
        },
        "masks": {
            "requested_missing_rate": 0.3,
            "config_hashes": {
                "train": "1" * 64,
                "validation": "2" * 64,
                "test": "3" * 64,
            },
            "realized_missing_rates": {
                "train": [0.29, 0.31],
                "validation": 0.30,
                "test": 0.28,
            },
        },
        "seeds": {
            "model_init": 201,
            "training_stochasticity": 202,
            "split": 203,
            "data_order": {
                "train": 101,
                "validation": 102,
                "test": 103,
            },
            "missing_mask": 204,
            "stability_mask": 205,
        },
        "initialization": {"shared_hash": "4" * 64},
        "stability": {"enabled": True, "mask_rate": 0.1, "weight": 0.01},
        "method": {
            "model_variant": "addon",
            "jepa_weight": 0.0,
            "loss_reconstruction": True,
        },
        "lifecycle": {
            "evaluation_protocol": "strict",
            "checkpoint_metric": "validation_weighted_f1",
            "best_epoch": 7,
            "best_validation_f1": 0.71,
            "test_call_count": 1,
            "epochs_completed": 10,
        },
        "metrics": {"weighted_f1": 0.70, "accuracy": 0.69},
        "outputs": {
            "result_archive": "/outputs/result.npz",
            "archive_fold_index": 0,
        },
    }


class FeatureMetadataHashTest(unittest.TestCase):
    def test_hash_is_stable_and_sorted_but_changes_with_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "z").mkdir()
            (root / "z" / "second.npy").write_bytes(b"22")
            (root / "first.npy").write_bytes(b"1")

            first = feature_metadata_hash(root)
            second = feature_metadata_hash(root)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 64)

            path = root / "first.npy"
            path.write_bytes(b"longer")
            self.assertNotEqual(first, feature_metadata_hash(root))

    def test_missing_feature_path_is_rejected(self) -> None:
        with self.assertRaises(FileNotFoundError):
            feature_metadata_hash(Path("/definitely/missing/features"))

    def test_symlink_target_is_part_of_metadata_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "target_a").write_bytes(b"x")
            (root / "target_b").write_bytes(b"x")
            link = root / "feature_link"
            link.symlink_to("target_a")
            timestamp = 1_700_000_000_000_000_000
            os.utime(link, ns=(timestamp, timestamp), follow_symlinks=False)
            first = feature_metadata_hash(link)

            link.unlink()
            link.symlink_to("target_b")
            os.utime(link, ns=(timestamp, timestamp), follow_symlinks=False)
            self.assertNotEqual(first, feature_metadata_hash(link))

    def test_root_directory_symlink_is_hashed_as_a_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "directory_a").mkdir()
            (root / "directory_b").mkdir()
            link = root / "feature_link"
            link.symlink_to("directory_a", target_is_directory=True)
            timestamp = 1_700_000_000_000_000_000
            os.utime(link, ns=(timestamp, timestamp), follow_symlinks=False)
            first = feature_metadata_hash(link)

            link.unlink()
            link.symlink_to("directory_b", target_is_directory=True)
            os.utime(link, ns=(timestamp, timestamp), follow_symlinks=False)
            self.assertNotEqual(first, feature_metadata_hash(link))


class ManifestRoundTripTest(unittest.TestCase):
    def test_atomic_write_normalizes_paths_and_numpy_values(self) -> None:
        manifest = complete_manifest()
        manifest["metrics"]["weighted_f1"] = np.float32(0.7)
        manifest["outputs"]["result_archive"] = Path("/outputs/result.npz")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "manifest.json"
            with mock.patch(
                "gcnet_modality_jepa.run_manifest.os.open", wraps=os.open
            ) as opened:
                write_manifest_atomic(path, manifest)
            loaded = load_manifest(path)

            self.assertEqual(loaded["metrics"]["weighted_f1"], float(np.float32(0.7)))
            self.assertEqual(loaded["outputs"]["result_archive"], "/outputs/result.npz")
            self.assertEqual(list(path.parent.glob(".manifest.json.*.tmp")), [])
            json.dumps(loaded)
            self.assertTrue(
                any(
                    Path(call.args[0]) == path.parent
                    and call.args[1] & getattr(os, "O_DIRECTORY", 0)
                    for call in opened.call_args_list
                ),
                opened.call_args_list,
            )

    def test_sampler_signature_is_order_sensitive_and_repeatable(self) -> None:
        first = sampler_signature([2, 0, 1], seed=99)
        self.assertEqual(first, sampler_signature([2, 0, 1], seed=99))
        self.assertNotEqual(first, sampler_signature([0, 1, 2], seed=99))
        self.assertNotEqual(first, sampler_signature([2, 0, 1], seed=100))

    def test_validation_rejects_schema_and_missing_required_field(self) -> None:
        wrong = complete_manifest()
        wrong["schema"]["version"] = MANIFEST_VERSION + 1
        with self.assertRaisesRegex(ManifestValidationError, "schema.version"):
            validate_manifest(wrong)

        for invalid_version in (True, 1.0):
            with self.subTest(version=invalid_version):
                wrong_type = complete_manifest()
                wrong_type["schema"]["version"] = invalid_version
                with self.assertRaisesRegex(ManifestValidationError, "schema.version"):
                    validate_manifest(wrong_type)

        missing = complete_manifest()
        del missing["split"]["hash"]
        with self.assertRaisesRegex(ManifestValidationError, "split.hash"):
            validate_manifest(missing)

    def test_validation_rejects_malformed_hash_rate_seed_and_split(self) -> None:
        cases = []
        malformed_hash = complete_manifest()
        malformed_hash["initialization"]["shared_hash"] = "not-a-hash"
        cases.append((malformed_hash, "initialization.shared_hash"))

        invalid_rate = complete_manifest()
        invalid_rate["masks"]["requested_missing_rate"] = 0.9
        cases.append((invalid_rate, "requested_missing_rate"))

        boolean_seed = complete_manifest()
        boolean_seed["run"]["master_seed"] = True
        cases.append((boolean_seed, "run.master_seed"))

        overlapping_split = complete_manifest()
        overlapping_split["split"]["indices"]["validation"] = [1]
        cases.append((overlapping_split, "split.indices"))

        invalid_test_count = complete_manifest()
        invalid_test_count["lifecycle"]["test_call_count"] = "1"
        cases.append((invalid_test_count, "test_call_count"))

        for manifest, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ManifestValidationError, message):
                    validate_manifest(manifest)


class PairedAuditTest(unittest.TestCase):
    def test_method_metrics_and_gpu_index_may_differ(self) -> None:
        baseline = complete_manifest()
        jepa = copy.deepcopy(baseline)
        jepa["environment"]["gpu"]["index"] = 7
        jepa["method"] = {
            "model_variant": "replacement",
            "jepa_weight": 0.1,
            "loss_reconstruction": False,
        }
        jepa["metrics"] = {"weighted_f1": 0.73, "accuracy": 0.72}
        jepa["outputs"] = {
            "result_archive": "/outputs/jepa.npz",
            "archive_fold_index": 0,
        }

        self.assertEqual(audit_paired_manifests(baseline, jepa), [])

    def test_each_invariant_family_is_rejected_with_field_path(self) -> None:
        cases = {
            "run.dataset": ("run", "dataset", "CMUMOSEI"),
            "environment.torch": ("environment", "torch", "2.2.2"),
            "environment.gpu.model": ("environment.gpu", "model", "A100"),
            "features.audio.metadata_sha256": (
                "features.audio", "metadata_sha256", "9" * 64
            ),
            "split.hash": ("split", "hash", "8" * 64),
            "samplers.train.signature": (
                "samplers.train", "signature", "7" * 64
            ),
            "masks.config_hashes.test": (
                "masks.config_hashes", "test", "6" * 64
            ),
            "masks.realized_missing_rates.test": (
                "masks.realized_missing_rates", "test", 0.4
            ),
            "initialization.shared_hash": (
                "initialization", "shared_hash", "5" * 64
            ),
            "stability.weight": ("stability", "weight", 0.02),
            "seeds.training_stochasticity": (
                "seeds", "training_stochasticity", 999
            ),
            "provenance.git_revision": (
                "provenance", "git_revision", "different-revision"
            ),
            "provenance.git_status": (
                "provenance", "git_status", " M train_gcnet.py"
            ),
        }
        for expected_path, (parent_path, key, value) in cases.items():
            with self.subTest(path=expected_path):
                baseline = complete_manifest()
                jepa = copy.deepcopy(baseline)
                parent = jepa
                for component in parent_path.split("."):
                    parent = parent[component]
                parent[key] = value
                mismatches = audit_paired_manifests(baseline, jepa)
                self.assertTrue(
                    any(expected_path in mismatch for mismatch in mismatches),
                    mismatches,
                )

    def test_test_call_count_must_be_one_even_when_pair_matches(self) -> None:
        baseline = complete_manifest()
        jepa = copy.deepcopy(baseline)
        baseline["lifecycle"]["test_call_count"] = 2
        jepa["lifecycle"]["test_call_count"] = 2

        with self.assertRaisesRegex(ManifestValidationError, "test_call_count"):
            audit_paired_manifests(baseline, jepa)

    def test_official_iemocap_allows_validation_test_overlap_and_epoch_tests(self) -> None:
        baseline = complete_manifest()
        baseline["run"]["dataset"] = "IEMOCAPSix"
        baseline["run"]["fold"] = 5
        baseline["split"]["indices"] = {
            "train": [0, 1],
            "validation": [2, 3],
            "test": [2, 3],
        }
        baseline["lifecycle"].update({
            "evaluation_protocol": "official",
            "test_call_count": 10,
            "epochs_completed": 10,
        })
        baseline["masks"]["realized_missing_rates"].update({
            "validation": [0.28, 0.30],
            "test": [0.27, 0.29],
        })
        jepa = copy.deepcopy(baseline)
        jepa["method"]["jepa_weight"] = 0.1

        validate_manifest(baseline)
        self.assertEqual(audit_paired_manifests(baseline, jepa), [])

        invalid = copy.deepcopy(baseline)
        invalid["lifecycle"]["test_call_count"] = 9
        with self.assertRaisesRegex(ManifestValidationError, "test_call_count"):
            validate_manifest(invalid)

    def test_cli_returns_zero_for_pair_and_nonzero_for_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline_path = root / "baseline.json"
            jepa_path = root / "jepa.json"
            write_manifest_atomic(baseline_path, complete_manifest())
            write_manifest_atomic(jepa_path, complete_manifest())
            script = Path(__file__).resolve().parents[1] / "scripts" / "audit_paired_runs.py"

            passed = subprocess.run(
                [sys.executable, str(script), str(baseline_path), str(jepa_path)],
                cwd=str(Path(__file__).resolve().parents[1]),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.assertEqual(passed.returncode, 0, passed.stdout)

            mismatched = complete_manifest()
            mismatched["split"]["hash"] = "0" * 64
            write_manifest_atomic(jepa_path, mismatched)
            failed = subprocess.run(
                [sys.executable, str(script), str(baseline_path), str(jepa_path)],
                cwd=str(Path(__file__).resolve().parents[1]),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("split.hash", failed.stdout)


if __name__ == "__main__":
    unittest.main()
