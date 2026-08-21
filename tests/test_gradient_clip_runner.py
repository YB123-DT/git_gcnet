from __future__ import annotations

import json
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from scripts import run_mosi_gradient_clip_diagnostics as runner


def _write_completion_evidence(
    job: runner.GradientClipJob,
    *,
    epoch_records: list[dict] | None = None,
) -> tuple[Path, dict]:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    (job.output_dir / "status.json").write_text(
        json.dumps({"identity": job.identity, "returncode": 0}) + "\n",
        encoding="utf-8",
    )
    manifest_path = (
        job.output_dir / "run_records" / "1" / "run_manifest_fold_1.json"
    )
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("{}\n", encoding="utf-8")
    metrics_path = job.output_dir / "run_records" / "1" / "fold_metrics.json"
    metrics_path.write_text(
        json.dumps([{"fold": 1, "gradient_clip_norm": 1.0}]) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "run": {"dataset": "CMUMOSI", "fold": 1, "master_seed": job.seed},
        "masks": {"requested_missing_rate": job.missing_rate},
        "lifecycle": {"evaluation_protocol": "official"},
        "method": {
            "model_variant": "addon" if job.method == "baseline" else "replacement",
            "jepa_weight": 0.0 if job.method == "baseline" else 0.1,
            "loss_reconstruction": job.method == "baseline",
        },
        "outputs": {"fold_metrics": str(metrics_path)},
    }
    if epoch_records is not None:
        (job.output_dir / "epoch_collapse_diagnostics.json").write_text(
            json.dumps(epoch_records) + "\n", encoding="utf-8"
        )
    return manifest_path, manifest


def _valid_epoch_records(epochs: int) -> list[dict]:
    return [
        {
            "fold": 1,
            "epoch": epoch,
            "gradient_clip": {
                "configured_norm": 1.0,
                "optimizer_steps": 3,
                "clipped_steps": 1,
                "clipped_fraction": 1.0 / 3.0,
                "pre_clip_norm_mean": 1.25,
                "pre_clip_norm_max": 2.0,
            },
        }
        for epoch in range(1, epochs + 1)
    ]


class GradientClipRunnerTest(unittest.TestCase):
    def test_exact_six_job_matrix_matches_formal_mosi_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            jobs = runner.build_jobs(
                Path(directory), "/official/python", gpu=3
            )

        self.assertEqual(len(jobs), 6)
        self.assertEqual(
            {(job.missing_rate, job.seed, job.method) for job in jobs},
            {
                (rate, seed, method)
                for rate, seed in ((0.3, 73), (0.4, 72), (0.4, 73))
                for method in ("baseline", "jepa")
            },
        )
        self.assertEqual({job.gpu for job in jobs}, {3})
        self.assertLessEqual(len({job.slot for job in jobs}), 3)
        self.assertEqual(len({job.output_dir for job in jobs}), 6)

        required_pairs = {
            "--audio-feature": "wav2vec-large-c-UTT",
            "--text-feature": "deberta-large-4-UTT",
            "--video-feature": "manet_UTT",
            "--dataset": "CMUMOSI",
            "--base-model": "LSTM",
            "--windowp": "2",
            "--windowf": "2",
            "--hidden": "200",
            "--lr": "0.001",
            "--dropout": "0.5",
            "--batch-size": "32",
            "--num-threads": "4",
            "--epochs": "100",
            "--evaluation-protocol": "official",
            "--stability-aux-mask-rate": "0.1",
            "--stability-recon-weight": "0.01",
            "--gradient-clip-norm": "1.0",
        }
        for job in jobs:
            command = list(job.command)
            self.assertIn("--epoch-collapse-diagnostics", command)
            self.assertNotIn("--allow-short-run", command)
            for flag, expected in required_pairs.items():
                self.assertEqual(command[command.index(flag) + 1], expected)
            self.assertEqual(
                command[command.index("--mask-type") + 1],
                "constant-{:.1f}".format(job.missing_rate),
            )
            self.assertEqual(
                command[command.index("--seed") + 1], str(job.seed)
            )
            if job.method == "baseline":
                self.assertIn("--loss-recon", command)
                self.assertEqual(
                    command[command.index("--jepa-weight") + 1], "0"
                )
                self.assertEqual(
                    command[command.index("--model-variant") + 1], "addon"
                )
            else:
                self.assertNotIn("--loss-recon", command)
                self.assertEqual(
                    command[command.index("--jepa-weight") + 1], "0.1"
                )
                self.assertEqual(
                    command[command.index("--model-variant") + 1],
                    "replacement",
                )

    def test_gpu_four_and_more_than_three_concurrent_jobs_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "GPU 4"):
                runner.build_jobs(root, "/official/python", gpu=4)
            with self.assertRaisesRegex(ValueError, "max_concurrent"):
                runner.build_jobs(
                    root, "/official/python", gpu=3, max_concurrent=4
                )

    def test_gpu_idle_gate_accepts_idle_and_rejects_non_idle_memory(self) -> None:
        with mock.patch.object(runner, "_gpu_memory_mb", return_value=768):
            runner.assert_gpu_available(3)
        with mock.patch.object(runner, "_gpu_memory_mb", return_value=769):
            with self.assertRaisesRegex(RuntimeError, "occupied GPU 3"):
                runner.assert_gpu_available(3)

    def test_completion_requires_valid_epoch_diagnostics_for_job_epochs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            job = runner.build_jobs(
                Path(directory), "/official/python", gpu=3, epochs=2
            )[0]
            manifest_path, manifest = _write_completion_evidence(job)

            with mock.patch.object(
                runner, "_latest_manifest", return_value=manifest_path
            ), mock.patch.object(runner, "load_manifest", return_value=manifest):
                self.assertFalse(runner.is_complete(job))

            records = _valid_epoch_records(2)
            (job.output_dir / "epoch_collapse_diagnostics.json").write_text(
                json.dumps(records) + "\n", encoding="utf-8"
            )
            with mock.patch.object(
                runner, "_latest_manifest", return_value=manifest_path
            ), mock.patch.object(runner, "load_manifest", return_value=manifest):
                self.assertTrue(runner.is_complete(job))

        command = list(job.command)
        self.assertEqual(command[command.index("--epochs") + 1], "2")
        self.assertIn("--allow-short-run", command)

    def test_completion_rejects_each_invalid_epoch_gradient_clip_field(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = runner.build_jobs(root, "/official/python", gpu=3, epochs=2)[0]
            valid_records = _valid_epoch_records(2)
            manifest_path, manifest = _write_completion_evidence(
                job, epoch_records=valid_records
            )
            cases = {
                "record_count": _valid_epoch_records(1),
                "owned_fold": [
                    {**record, "fold": 2} for record in valid_records
                ],
                "configured_norm": [
                    {
                        **record,
                        "gradient_clip": {
                            **record["gradient_clip"],
                            "configured_norm": 0.5,
                        },
                    }
                    for record in valid_records
                ],
                "optimizer_steps": [
                    {
                        **record,
                        "gradient_clip": {
                            **record["gradient_clip"],
                            "optimizer_steps": 0,
                        },
                    }
                    for record in valid_records
                ],
                "nonfinite_mean": [
                    {
                        **record,
                        "gradient_clip": {
                            **record["gradient_clip"],
                            "pre_clip_norm_mean": float("nan"),
                        },
                    }
                    for record in valid_records
                ],
                "nonfinite_max": [
                    {
                        **record,
                        "gradient_clip": {
                            **record["gradient_clip"],
                            "pre_clip_norm_max": float("inf"),
                        },
                    }
                    for record in valid_records
                ],
                "clipped_steps": [
                    {
                        **record,
                        "gradient_clip": {
                            **record["gradient_clip"],
                            "clipped_steps": 4,
                        },
                    }
                    for record in valid_records
                ],
                "clipped_fraction": [
                    {
                        **record,
                        "gradient_clip": {
                            **record["gradient_clip"],
                            "clipped_fraction": 1.1,
                        },
                    }
                    for record in valid_records
                ],
            }
            for name, records in cases.items():
                with self.subTest(case=name):
                    (job.output_dir / "epoch_collapse_diagnostics.json").write_text(
                        json.dumps(records) + "\n", encoding="utf-8"
                    )
                    with mock.patch.object(
                        runner, "_latest_manifest", return_value=manifest_path
                    ), mock.patch.object(
                        runner, "load_manifest", return_value=manifest
                    ):
                        self.assertFalse(runner.is_complete(job))

    def test_completed_job_is_resumed_without_launching_or_rewriting_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = runner.build_jobs(root / "outputs", "/official/python", 3)[0]
            job.output_dir.mkdir(parents=True)
            status_path = job.output_dir / "status.json"
            status_path.write_text('{"sentinel": true}\n', encoding="utf-8")

            with mock.patch.object(runner, "is_complete", return_value=True), mock.patch.object(
                runner.subprocess, "run"
            ) as launch:
                completed = runner.run_job(job, root, threading.Event())

            self.assertTrue(completed)
            launch.assert_not_called()
            self.assertEqual(
                json.loads(status_path.read_text(encoding="utf-8")),
                {"sentinel": True},
            )

    def test_incomplete_job_writes_only_its_isolated_logs_and_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = runner.build_jobs(root / "outputs", "/official/python", 3)[0]
            completed_process = subprocess.CompletedProcess(job.command, 0)

            with mock.patch.object(
                runner, "is_complete", side_effect=[False, False, True]
            ), mock.patch.object(
                runner.subprocess, "run", return_value=completed_process
            ) as launch:
                completed = runner.run_job(job, root, threading.Event())

            self.assertTrue(completed)
            launch.assert_called_once()
            self.assertTrue((job.output_dir / "command.json").is_file())
            self.assertTrue((job.output_dir / "train.log").is_file())
            status = json.loads(
                (job.output_dir / "status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(status["identity"], job.identity)
            self.assertEqual(status["returncode"], 0)

    def test_three_completed_pairs_are_audited(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = runner.build_jobs(root / "outputs", "/official/python", 3)
            manifests = {
                job.identity: job.output_dir / "manifest.json" for job in jobs
            }
            for manifest in manifests.values():
                manifest.parent.mkdir(parents=True, exist_ok=True)
                manifest.write_text("{}\n", encoding="utf-8")

            with mock.patch.object(runner, "is_complete", return_value=True), mock.patch.object(
                runner, "_latest_manifest", side_effect=lambda path: next(
                    manifests[job.identity]
                    for job in jobs
                    if job.output_dir == path
                )
            ), mock.patch.object(
                runner.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0, stdout="ok\n"),
            ):
                audited, failures = runner.audit_completed_pairs(
                    jobs, root, "/official/python"
                )

            self.assertEqual((audited, failures), (3, 0))
            audit_logs = list((root / "outputs").glob("**/paired_audit.log"))
            self.assertEqual(len(audit_logs), 3)


if __name__ == "__main__":
    unittest.main()
