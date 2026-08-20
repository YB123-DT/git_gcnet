from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import Dataset

from gcnet_modality_jepa import train_gcnet
from gcnet_modality_jepa.protocol import SeedBundle
from gcnet_modality_jepa.run_manifest import load_manifest, sampler_signature, validate_manifest


class _Dataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return index

    @staticmethod
    def collate_fn(items):
        return items


class ManifestTrainerIntegrationTest(unittest.TestCase):
    def test_loader_carries_exact_split_and_order_metadata(self):
        loader = train_gcnet._build_protocol_loader(
            _Dataset(),
            indices=(2, 0),
            dataset_name="CMUMOSI",
            fold=1,
            split="train",
            batch_size=2,
            num_workers=0,
            seed_bundle=SeedBundle(66),
            split_hash="a" * 64,
        )

        metadata = loader.protocol_metadata
        expected_seed = SeedBundle(66).derive("data_order:CMUMOSI:fold:1:train")
        self.assertEqual(metadata["indices"], [2, 0])
        self.assertEqual(metadata["split_hash"], "a" * 64)
        self.assertEqual(metadata["order_seed"], expected_seed)
        self.assertEqual(
            metadata["order_signature"], sampler_signature([2, 0], expected_seed)
        )

    def test_mask_audit_excludes_padding(self):
        host = torch.tensor(
            [[[1, 0, 1]], [[0, 1, 1]], [[0, 0, 0]]], dtype=torch.uint8
        )
        guest = torch.tensor(
            [[[1, 1, 1]], [[1, 0, 1]], [[0, 0, 0]]], dtype=torch.uint8
        )
        umask = torch.tensor([[1.0, 1.0, 0.0]])

        audit = train_gcnet.primary_mask_audit(host, guest, umask)

        self.assertEqual(audit["missing_elements"], 3)
        self.assertEqual(audit["total_elements"], 12)
        self.assertEqual(audit["realized_missing_rate"], 0.25)

    def test_complete_fold_manifest_validates_and_round_trips(self):
        args = SimpleNamespace(
            dataset="CMUMOSI",
            seed=66,
            model_variant="replacement",
            jepa_weight=0.1,
            loss_recon=False,
            all_modal_recon_weight=0.0,
            stability_recon_weight=0.01,
            stability_aux_mask_rate=0.1,
        )
        split_hash = "d" * 64

        def metadata(split, indices):
            seed = SeedBundle(66).derive(
                "data_order:CMUMOSI:fold:1:{}".format(split)
            )
            return {
                "split": split,
                "fold": 1,
                "indices": list(indices),
                "split_hash": split_hash,
                "order_seed": seed,
                "order_signature": sampler_signature(indices, seed),
            }

        lifecycle_evidence = {
            "evaluation_protocol": "strict",
            "epochs_completed": 2,
            "best_epoch": 2,
            "best_validation_f1": 0.7,
            "test_call_count": 1,
            "mask_schedule_hashes": {
                "train": "1" * 64,
                "validation": "2" * 64,
                "test": "3" * 64,
            },
            "realized_missing_rates": {
                "train": [0.2, 0.3],
                "validation": 0.25,
                "test": 0.24,
            },
        }
        manifest = train_gcnet.build_fold_run_manifest(
            args=args,
            fold=1,
            loader_metadata={
                "train": metadata("train", (0, 1)),
                "validation": metadata("validation", (2,)),
                "test": metadata("test", (3,)),
            },
            lifecycle_evidence=lifecycle_evidence,
            fold_record={"weighted_f1": 0.69, "accuracy": 0.68},
            feature_evidence={
                "audio": {"path": "/a", "metadata_sha256": "a" * 64},
                "text": {"path": "/t", "metadata_sha256": "b" * 64},
                "visual": {"path": "/v", "metadata_sha256": "c" * 64},
            },
            environment={
                "python": "3.8.20", "torch": "1.8.0", "cuda": "10.2",
                "cudnn": 7605, "pyg": "2.0.1", "numpy": "1.21.6",
                "sklearn": "1.0.2",
                "gpu": {"index": 0, "model": "V100", "driver": "575"},
            },
            provenance={
                "command": ["python", "train"], "cwd": "/repo",
                "git_revision": "abc", "git_status": "clean",
            },
            shared_init_hash="4" * 64,
            training_seed=SeedBundle(66).derive("training_stochasticity:fold:1"),
            mask_rate=0.3,
            output_paths={
                "result_archive": "/out/result.npz",
                "fold_metrics": "/out/run_records/id/fold_metrics.json",
                "archive_fold_index": 0,
            },
        )
        validate_manifest(manifest)
        self.assertEqual(manifest["lifecycle"]["test_call_count"], 1)
        self.assertEqual(manifest["method"]["model_variant"], "replacement")
        self.assertEqual(manifest["outputs"]["archive_fold_index"], 0)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            train_gcnet.write_manifest_atomic(path, manifest)
            self.assertEqual(load_manifest(path), manifest)


if __name__ == "__main__":
    unittest.main()
