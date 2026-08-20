from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from gcnet_modality_jepa.run_manifest import MANIFEST_NAME, MANIFEST_VERSION
from scripts import run_official_missing_sweep as sweep


def _manifest_for(job: sweep.OfficialJob) -> dict:
    if job.dataset.startswith("IEMOCAP"):
        split_indices = {
            "train": [0, 1],
            "validation": [2, 3],
            "test": [2, 3],
        }
        fold = 5
    else:
        split_indices = {
            "train": [0],
            "validation": [1],
            "test": [2],
        }
        fold = 1
    method = (
        {
            "model_variant": "addon",
            "jepa_weight": 0.0,
            "loss_reconstruction": True,
        }
        if job.method == "baseline"
        else {
            "model_variant": "replacement",
            "jepa_weight": 0.1,
            "loss_reconstruction": False,
        }
    )
    return {
        "schema": {"name": MANIFEST_NAME, "version": MANIFEST_VERSION},
        "run": {
            "dataset": job.dataset,
            "fold": fold,
            "master_seed": job.seed,
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
                "index": job.gpu,
                "model": "Tesla V100-SXM2-32GB",
                "driver": "575.51.03",
            },
        },
        "provenance": {
            "command": list(job.command),
            "cwd": "/experiment",
            "git_revision": "abc123",
            "git_status": "clean",
        },
        "features": {
            "audio": {"path": "/features/audio", "metadata_sha256": "a" * 64},
            "text": {"path": "/features/text", "metadata_sha256": "b" * 64},
            "visual": {"path": "/features/visual", "metadata_sha256": "c" * 64},
        },
        "split": {"indices": split_indices, "hash": "d" * 64},
        "samplers": {
            "train": {"seed": 101, "signature": "e" * 64},
            "validation": {"seed": 102, "signature": "f" * 64},
            "test": {"seed": 103, "signature": "0" * 64},
        },
        "masks": {
            "requested_missing_rate": job.missing_rate,
            "config_hashes": {
                "train": "1" * 64,
                "validation": "2" * 64,
                "test": "3" * 64,
            },
            "realized_missing_rates": {
                "train": [job.missing_rate],
                "validation": [job.missing_rate],
                "test": [job.missing_rate],
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
        "method": method,
        "lifecycle": {
            "evaluation_protocol": "official",
            "checkpoint_metric": "validation_weighted_f1",
            "best_epoch": 1,
            "best_validation_f1": 0.71,
            "test_call_count": 2,
            "epochs_completed": 2,
        },
        "metrics": {"weighted_f1": 0.70, "accuracy": 0.69},
        "outputs": {
            "result_archive": "/outputs/result.npz",
            "archive_fold_index": 0,
        },
    }


def _write_evidence(
    job: sweep.OfficialJob,
    manifest: dict | None = None,
    identity: str | None = None,
) -> Path:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    (job.output_dir / "status.json").write_text(
        json.dumps(
            {
                "identity": job.identity if identity is None else identity,
                "returncode": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = (
        job.output_dir
        / "run_records"
        / "123"
        / "run_manifest_fold_{}.json".format(
            5 if job.dataset.startswith("IEMOCAP") else 1
        )
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(_manifest_for(job) if manifest is None else manifest) + "\n",
        encoding="utf-8",
    )
    return manifest_path


class OfficialSweepTest(unittest.TestCase):
    def test_matrix_contains_exactly_640_unique_official_jobs_on_seven_gpus(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs = sweep.build_jobs(
                output_root=Path(directory),
                python="/official/python",
                jobs_per_gpu=3,
            )

        self.assertEqual(len(jobs), 640)
        identities = {
            (job.dataset, job.method, job.missing_rate, job.seed)
            for job in jobs
        }
        self.assertEqual(len(identities), 640)
        self.assertEqual({job.seed for job in jobs}, set(range(66, 76)))
        self.assertEqual(
            {job.missing_rate for job in jobs},
            {round(index / 10, 1) for index in range(8)},
        )
        self.assertEqual({job.gpu for job in jobs}, {0, 1, 2, 3, 5, 6, 7})
        self.assertTrue(all(job.slot in (0, 1, 2) for job in jobs))
        self.assertTrue(
            all("--evaluation-protocol" in job.command for job in jobs)
        )
        self.assertTrue(
            all(
                job.command[job.command.index("--evaluation-protocol") + 1]
                == "official"
                for job in jobs
            )
        )
        for job in jobs:
            if job.dataset.startswith("IEMOCAP"):
                self.assertIn("--fold", job.command)
                self.assertEqual(job.command[job.command.index("--fold") + 1], "5")
            else:
                self.assertNotIn("--fold", job.command)

    def test_broken_gpu_four_and_invalid_density_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "GPU 4"):
                sweep.build_jobs(root, "/official/python", (0, 4), 2)
            with self.assertRaisesRegex(ValueError, "jobs_per_gpu"):
                sweep.build_jobs(root, "/official/python", (0,), 0)
            with self.assertRaisesRegex(ValueError, "jobs_per_gpu"):
                sweep.build_jobs(root, "/official/python", (0,), 4)

    def test_legacy_status_and_exact_valid_manifest_are_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs = sweep.build_jobs(Path(directory), "/official/python")
            baseline = jobs[0]
            jepa = jobs[1]
            _write_evidence(baseline)
            _write_evidence(jepa)

            self.assertTrue(sweep.is_complete(baseline))
            self.assertTrue(sweep.is_complete(jepa))

    def test_completion_rejects_empty_manifest_and_mismatched_status_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs = sweep.build_jobs(Path(directory), "/official/python")
            empty_manifest_job = jobs[0]
            wrong_identity_job = jobs[1]
            _write_evidence(empty_manifest_job, manifest={})
            _write_evidence(wrong_identity_job, identity="stale:identity:0.0:66")

            self.assertFalse(sweep.is_complete(empty_manifest_job))
            self.assertFalse(sweep.is_complete(wrong_identity_job))

    def test_completion_rejects_every_mismatched_job_manifest_field(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template_job = sweep.build_jobs(root, "/official/python")[0]
            cases = {
                "dataset": ("run", "dataset", "IEMOCAPSix"),
                "fold": ("run", "fold", 4),
                "seed": ("run", "master_seed", 75),
                "rate": ("masks", "requested_missing_rate", 0.7),
                "protocol": ("lifecycle", "evaluation_protocol", "strict"),
                "variant": ("method", "model_variant", "replacement"),
                "jepa_weight": ("method", "jepa_weight", 0.1),
                "loss_reconstruction": (
                    "method",
                    "loss_reconstruction",
                    False,
                ),
            }
            for index, (name, (section, key, value)) in enumerate(cases.items()):
                with self.subTest(field=name):
                    job = sweep.OfficialJob(
                        dataset=template_job.dataset,
                        method=template_job.method,
                        missing_rate=template_job.missing_rate,
                        seed=template_job.seed,
                        gpu=template_job.gpu,
                        slot=template_job.slot,
                        output_dir=root / "case_{}".format(index),
                        command=template_job.command,
                    )
                    manifest = copy.deepcopy(_manifest_for(job))
                    manifest[section][key] = value
                    _write_evidence(job, manifest=manifest)
                    self.assertFalse(sweep.is_complete(job))

    def test_successful_process_without_manifest_fails_and_releases_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = sweep.build_jobs(root / "outputs", "/official/python")[0]
            stop_event = threading.Event()
            completed = subprocess.CompletedProcess(job.command, 0)

            with mock.patch.object(sweep.subprocess, "run", return_value=completed):
                self.assertFalse(sweep.run_job(job, root, stop_event))

            self.assertTrue(stop_event.is_set())
            self.assertFalse((job.output_dir / sweep.CLAIM_FILE).exists())
            status = json.loads(
                (job.output_dir / "status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(status["identity"], job.identity)
            self.assertEqual(status["returncode"], 0)
            self.assertIn("manifest", status["error"])

    def test_subprocess_exception_writes_failure_status_and_releases_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = sweep.build_jobs(root / "outputs", "/official/python")[0]
            stop_event = threading.Event()

            with mock.patch.object(
                sweep.subprocess, "run", side_effect=OSError("cannot start")
            ):
                self.assertFalse(sweep.run_job(job, root, stop_event))

            self.assertTrue(stop_event.is_set())
            self.assertFalse((job.output_dir / sweep.CLAIM_FILE).exists())
            status = json.loads(
                (job.output_dir / "status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(status["identity"], job.identity)
            self.assertIsNone(status["returncode"])
            self.assertIn("cannot start", status["error"])

    def test_output_claim_is_atomic_and_reusable_after_release(self):
        with tempfile.TemporaryDirectory() as directory:
            job = sweep.build_jobs(Path(directory), "/official/python")[0]

            first = sweep._acquire_claim(job)
            self.assertIsNotNone(first)
            self.assertIsNone(sweep._acquire_claim(job))
            sweep._release_claim(first)
            second = sweep._acquire_claim(job)
            self.assertIsNotNone(second)
            sweep._release_claim(second)

    def test_worker_result_collection_consumes_every_future(self):
        class StubFuture:
            def __init__(self, result=None, error=None):
                self.value = result
                self.error = error
                self.calls = 0

            def result(self):
                self.calls += 1
                if self.error is not None:
                    raise self.error
                return self.value

        futures = [
            StubFuture(result=False),
            StubFuture(error=RuntimeError("lane failed")),
            StubFuture(result=True),
        ]
        stop_event = threading.Event()

        success, errors = sweep._collect_worker_results(futures, stop_event)

        self.assertFalse(success)
        self.assertEqual([future.calls for future in futures], [1, 1, 1])
        self.assertEqual(len(errors), 1)
        self.assertTrue(stop_event.is_set())

    def test_final_success_requires_all_640_jobs_and_all_320_pair_audits(self):
        self.assertTrue(sweep._scheduler_succeeded(True, 640, 640, 320, 0))
        rejected = [
            (False, 640, 640, 320, 0),
            (True, 639, 640, 320, 0),
            (True, 640, 639, 320, 0),
            (True, 640, 640, 319, 0),
            (True, 640, 640, 320, 1),
        ]
        for arguments in rejected:
            with self.subTest(arguments=arguments):
                self.assertFalse(sweep._scheduler_succeeded(*arguments))

    def test_cli_dry_run_imports_repository_package_and_writes_640_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "dry-run"
            repository_root = Path(__file__).resolve().parents[1]
            environment = dict(os.environ)
            environment["PYTHONPATH"] = ""

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_official_missing_sweep.py",
                    "--output-root",
                    str(output_root),
                    "--dry-run",
                ],
                cwd=str(repository_root),
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stdout)
            tasks = json.loads(
                (output_root / "task_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(tasks), 640)


if __name__ == "__main__":
    unittest.main()
