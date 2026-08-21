from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from scripts import run_mosi_gradient_clip_diagnostics as runner


TEST_REVISION = "a" * 40
TEST_GIT_STATUS = " M unrelated.txt"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_command(job: runner.GradientClipJob) -> list[str]:
    return [
        str(runner.REPOSITORY_ROOT / "gcnet_modality_jepa" / "train_gcnet.py"),
        *job.command[4:],
    ]


def _contract_sha256(job: runner.GradientClipJob) -> str:
    payload = {
        "command": list(job.command),
        "identity": job.identity,
        "repository_root": str(runner.REPOSITORY_ROOT),
        "source_contract_sha256": job.source_contract_sha256,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_completion_evidence(
    job: runner.GradientClipJob,
    *,
    epoch_records: list[dict] | None = None,
    revision: str = TEST_REVISION,
) -> tuple[Path, dict, dict]:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = (
        job.output_dir / "run_records" / "1" / "run_manifest_fold_1.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{}\n", encoding="utf-8")
    metrics_path = job.output_dir / "run_records" / "1" / "fold_metrics.json"
    metrics_path.write_text(
        json.dumps([{"fold": 1, "gradient_clip_norm": 1.0}]) + "\n",
        encoding="utf-8",
    )
    result_path = job.output_dir / "result.npz"
    result_path.write_bytes(b"result")
    manifest = {
        "run": {"dataset": "CMUMOSI", "fold": 1, "master_seed": job.seed},
        "masks": {"requested_missing_rate": job.missing_rate},
        "lifecycle": {
            "evaluation_protocol": "official",
            "epochs_completed": job.epochs,
        },
        "provenance": {
            "cwd": str(runner.REPOSITORY_ROOT),
            "git_revision": revision,
            "git_status": TEST_GIT_STATUS,
            "command": _manifest_command(job),
        },
        "method": {
            "model_variant": "addon" if job.method == "baseline" else "replacement",
            "jepa_weight": 0.0 if job.method == "baseline" else 0.1,
            "loss_reconstruction": job.method == "baseline",
        },
        "outputs": {
            "fold_metrics": str(metrics_path),
            "result_archive": str(result_path),
        },
    }
    epoch_path = job.output_dir / "epoch_collapse_diagnostics.json"
    if epoch_records is not None:
        epoch_path.write_text(
            json.dumps(epoch_records) + "\n", encoding="utf-8"
        )
    status = {
        "identity": job.identity,
        "returncode": 0,
        "attempt_started_at_unix": 123.0,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "epoch_diagnostics_path": str(epoch_path.resolve()),
        "epoch_diagnostics_sha256": (
            _sha256(epoch_path) if epoch_path.exists() else "0" * 64
        ),
        "command_contract_sha256": _contract_sha256(job),
        "source_contract_sha256": job.source_contract_sha256,
        "code_revision": revision,
        "git_status": TEST_GIT_STATUS,
        "python_executable": str(Path(job.command[0]).resolve()),
        "fold_metrics_path": str(metrics_path.resolve()),
        "fold_metrics_sha256": _sha256(metrics_path),
        "result_archive_path": str(result_path.resolve()),
        "result_archive_sha256": _sha256(result_path),
    }
    (job.output_dir / "status.json").write_text(
        json.dumps(status) + "\n", encoding="utf-8"
    )
    return manifest_path, manifest, status


def _valid_epoch_records(epochs: int) -> list[dict]:
    return [
        {
            "fold": 1,
            "epoch": epoch,
            "train_weighted_f1": 0.7,
            "val_weighted_f1": 0.6,
            "temporal_zero_ratio": 0.2,
            "regression_head_weight_norm": 1.5,
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


def _bound_is_complete(
    job: runner.GradientClipJob,
    manifest: dict,
    revision: str = TEST_REVISION,
) -> bool:
    with mock.patch.object(
        runner, "load_manifest", return_value=manifest
    ), mock.patch.object(
        runner, "_code_revision", return_value=revision
    ), mock.patch.object(
        runner, "_git_status", return_value=TEST_GIT_STATUS
    ):
        return runner.is_complete(job)


class GradientClipRunnerTest(unittest.TestCase):
    def test_recursive_source_digest_changes_for_previously_omitted_module(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.py").write_text("CONFIG = 1\n", encoding="utf-8")
            modality = root / "gcnet_modality_jepa"
            replacement = root / "gcnet_jepa_replacement"
            scripts = root / "scripts"
            for path in (modality, replacement, scripts):
                path.mkdir(parents=True)
            (modality / "train_gcnet.py").write_text("TRAIN = 1\n", encoding="utf-8")
            omitted = modality / "previously_omitted.py"
            omitted.write_text("VALUE = 1\n", encoding="utf-8")
            (replacement / "model.py").write_text("MODEL = 1\n", encoding="utf-8")
            (scripts / "run_mosi_gradient_clip_diagnostics.py").write_text(
                "RUNNER = 1\n", encoding="utf-8"
            )

            before = runner._source_contract_sha256(root)
            omitted.write_text("VALUE = 2\n", encoding="utf-8")
            after = runner._source_contract_sha256(root)

        self.assertNotEqual(before, after)

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
            records = _valid_epoch_records(2)
            _, manifest, _ = _write_completion_evidence(
                job, epoch_records=records
            )
            (job.output_dir / "epoch_collapse_diagnostics.json").unlink()

            self.assertFalse(_bound_is_complete(job, manifest))

            _, manifest, _ = _write_completion_evidence(
                job, epoch_records=records
            )
            self.assertTrue(_bound_is_complete(job, manifest))

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
            _, manifest, _ = _write_completion_evidence(
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
                "nonfinite_collapse": [
                    {**record, "prediction_std": float("nan")}
                    for record in valid_records
                ],
                "nonfinite_nested": [
                    {
                        **record,
                        "primary_mask": {
                            "realized_missing_rate": float("inf")
                        },
                    }
                    for record in valid_records
                ],
                "negative_collapse_norm": [
                    {**record, "regression_head_weight_norm": -0.1}
                    for record in valid_records
                ],
                "negative_norm": [
                    {
                        **record,
                        "gradient_clip": {
                            **record["gradient_clip"],
                            "pre_clip_norm_mean": -0.1,
                        },
                    }
                    for record in valid_records
                ],
                "max_below_mean": [
                    {
                        **record,
                        "gradient_clip": {
                            **record["gradient_clip"],
                            "pre_clip_norm_mean": 2.0,
                            "pre_clip_norm_max": 1.0,
                        },
                    }
                    for record in valid_records
                ],
                "fraction_mismatch": [
                    {
                        **record,
                        "gradient_clip": {
                            **record["gradient_clip"],
                            "clipped_fraction": 0.5,
                        },
                    }
                    for record in valid_records
                ],
                "zero_clipped_above_threshold": [
                    {
                        **record,
                        "gradient_clip": {
                            **record["gradient_clip"],
                            "clipped_steps": 0,
                            "clipped_fraction": 0.0,
                            "pre_clip_norm_mean": 0.5,
                            "pre_clip_norm_max": 1.1,
                        },
                    }
                    for record in valid_records
                ],
                "clipped_without_exceeding_threshold": [
                    {
                        **record,
                        "gradient_clip": {
                            **record["gradient_clip"],
                            "clipped_steps": 1,
                            "pre_clip_norm_mean": 0.5,
                            "pre_clip_norm_max": 1.0,
                        },
                    }
                    for record in valid_records
                ],
                "boolean_numeric": [
                    {
                        **record,
                        "gradient_clip": {
                            **record["gradient_clip"],
                            "pre_clip_norm_max": True,
                        },
                    }
                    for record in valid_records
                ],
                "duplicate_epoch": [
                    {**record, "epoch": 1} for record in valid_records
                ],
            }
            for name, records in cases.items():
                with self.subTest(case=name):
                    _, manifest, _ = _write_completion_evidence(
                        job, epoch_records=records
                    )
                    self.assertFalse(_bound_is_complete(job, manifest))

    def test_completed_job_is_resumed_without_launching_or_rewriting_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            job = runner.build_jobs(
                Path(directory) / "outputs", "/official/python", 3
            )[0]
            job.output_dir.mkdir(parents=True)
            status_path = job.output_dir / "status.json"
            status_path.write_text('{"sentinel": true}\n', encoding="utf-8")

            with mock.patch.object(runner, "is_complete", return_value=True), mock.patch.object(
                runner.subprocess, "run"
            ) as launch:
                completed = runner.run_job(
                    job, runner.REPOSITORY_ROOT, threading.Event()
                )

            self.assertTrue(completed)
            launch.assert_not_called()
            self.assertEqual(
                json.loads(status_path.read_text(encoding="utf-8")),
                {"sentinel": True},
            )

    def test_successful_attempt_binds_new_artifacts_in_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            job = runner.build_jobs(
                Path(directory) / "outputs", "/official/python", 3, epochs=2
            )[0]
            completed_process = subprocess.CompletedProcess(job.command, 0)
            manifest_box: dict[str, dict] = {}

            def launch(*args, **kwargs):
                _, manifest, _ = _write_completion_evidence(
                    job, epoch_records=_valid_epoch_records(job.epochs)
                )
                manifest_box["value"] = manifest
                return completed_process

            with mock.patch.object(
                runner.subprocess, "run", side_effect=launch
            ) as process, mock.patch.object(
                runner,
                "load_manifest",
                side_effect=lambda path: manifest_box["value"],
            ), mock.patch.object(
                runner, "_code_revision", return_value=TEST_REVISION
            ), mock.patch.object(
                runner, "_git_status", return_value=TEST_GIT_STATUS
            ):
                completed = runner.run_job(
                    job, runner.REPOSITORY_ROOT, threading.Event()
                )

            self.assertTrue(completed)
            process.assert_called_once()
            self.assertEqual(
                Path(process.call_args.kwargs["cwd"]), runner.REPOSITORY_ROOT
            )
            self.assertEqual(
                process.call_args.kwargs["env"]["PYTHONPATH"],
                str(runner.REPOSITORY_ROOT),
            )
            self.assertTrue((job.output_dir / "command.json").is_file())
            self.assertTrue((job.output_dir / "train.log").is_file())
            status = json.loads(
                (job.output_dir / "status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(status["identity"], job.identity)
            self.assertEqual(status["returncode"], 0)
            self.assertEqual(status["code_revision"], TEST_REVISION)
            self.assertEqual(status["git_status"], TEST_GIT_STATUS)
            self.assertEqual(
                status["source_contract_sha256"], job.source_contract_sha256
            )
            self.assertEqual(
                status["command_contract_sha256"], _contract_sha256(job)
            )
            for key in (
                "manifest_path",
                "manifest_sha256",
                "epoch_diagnostics_path",
                "epoch_diagnostics_sha256",
                "python_executable",
                "attempt_started_at_unix",
                "fold_metrics_path",
                "fold_metrics_sha256",
                "result_archive_path",
                "result_archive_sha256",
            ):
                self.assertIn(key, status)

    def test_source_mutation_during_subprocess_rejects_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            job = runner.build_jobs(
                Path(directory) / "outputs", "/official/python", 3, epochs=2
            )[0]
            manifest_box: dict[str, dict] = {}

            def launch(*args, **kwargs):
                _, manifest, _ = _write_completion_evidence(
                    job, epoch_records=_valid_epoch_records(job.epochs)
                )
                manifest_box["value"] = manifest
                return subprocess.CompletedProcess(job.command, 0)

            with mock.patch.object(
                runner.subprocess, "run", side_effect=launch
            ), mock.patch.object(
                runner,
                "load_manifest",
                side_effect=lambda path: manifest_box["value"],
            ), mock.patch.object(
                runner, "_code_revision", return_value=TEST_REVISION
            ), mock.patch.object(
                runner, "_git_status", return_value=TEST_GIT_STATUS
            ), mock.patch.object(
                runner,
                "_source_contract_sha256",
                side_effect=[job.source_contract_sha256, "b" * 64],
            ):
                completed = runner.run_job(
                    job, runner.REPOSITORY_ROOT, threading.Event()
                )

            self.assertFalse(completed)
            status = json.loads(
                (job.output_dir / "status.json").read_text(encoding="utf-8")
            )
            self.assertIn("source contract changed", status["error"])

    def test_rc0_rejects_stale_artifacts_from_before_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            job = runner.build_jobs(
                Path(directory) / "outputs", "/official/python", 3, epochs=2
            )[0]
            _, manifest, status = _write_completion_evidence(
                job, epoch_records=_valid_epoch_records(job.epochs)
            )
            status["returncode"] = 1
            (job.output_dir / "status.json").write_text(
                json.dumps(status) + "\n", encoding="utf-8"
            )

            with mock.patch.object(
                runner.subprocess,
                "run",
                return_value=subprocess.CompletedProcess(job.command, 0),
            ) as process, mock.patch.object(
                runner, "load_manifest", return_value=manifest
            ), mock.patch.object(
                runner, "_code_revision", return_value=TEST_REVISION
            ), mock.patch.object(
                runner, "_git_status", return_value=TEST_GIT_STATUS
            ):
                completed = runner.run_job(
                    job, runner.REPOSITORY_ROOT, threading.Event()
                )

            self.assertFalse(completed)
            process.assert_called_once()
            failure = json.loads(
                (job.output_dir / "status.json").read_text(encoding="utf-8")
            )
            self.assertIn("new", failure["error"])

    def test_rc0_rejects_multiple_new_run_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            job = runner.build_jobs(
                Path(directory) / "outputs", "/official/python", 3, epochs=2
            )[0]
            manifest_box: dict[str, dict] = {}

            def launch(*args, **kwargs):
                manifest_path, manifest, _ = _write_completion_evidence(
                    job, epoch_records=_valid_epoch_records(job.epochs)
                )
                second = (
                    job.output_dir
                    / "run_records"
                    / "2"
                    / "run_manifest_fold_1.json"
                )
                second.parent.mkdir(parents=True)
                second.write_bytes(manifest_path.read_bytes())
                manifest_box["value"] = manifest
                return subprocess.CompletedProcess(job.command, 0)

            with mock.patch.object(
                runner.subprocess, "run", side_effect=launch
            ), mock.patch.object(
                runner,
                "load_manifest",
                side_effect=lambda path: manifest_box["value"],
            ), mock.patch.object(
                runner, "_code_revision", return_value=TEST_REVISION
            ), mock.patch.object(
                runner, "_git_status", return_value=TEST_GIT_STATUS
            ):
                completed = runner.run_job(
                    job, runner.REPOSITORY_ROOT, threading.Event()
                )

            self.assertFalse(completed)
            failure = json.loads(
                (job.output_dir / "status.json").read_text(encoding="utf-8")
            )
            self.assertIn("exactly one new", failure["error"])

    def test_bound_completion_rejects_hash_path_and_artifact_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            job = runner.build_jobs(
                Path(directory), "/official/python", 3, epochs=2
            )[0]
            _, manifest, valid_status = _write_completion_evidence(
                job, epoch_records=_valid_epoch_records(job.epochs)
            )
            self.assertTrue(_bound_is_complete(job, manifest))

            cases = {
                "manifest_hash": {"manifest_sha256": "0" * 64},
                "epoch_hash": {"epoch_diagnostics_sha256": "0" * 64},
                "manifest_path": {
                    "manifest_path": str(Path(directory) / "outside.json")
                },
                "contract_hash": {"command_contract_sha256": "0" * 64},
                "source_hash": {"source_contract_sha256": "0" * 64},
                "metrics_hash": {"fold_metrics_sha256": "0" * 64},
                "metrics_path": {"fold_metrics_path": "/wrong/metrics.json"},
                "result_hash": {"result_archive_sha256": "0" * 64},
                "result_path": {"result_archive_path": "/wrong/result.npz"},
                "revision": {"code_revision": "b" * 40},
                "git_status": {"git_status": "clean"},
                "python": {"python_executable": "/wrong/python"},
            }
            for name, changes in cases.items():
                with self.subTest(case=name):
                    status = {**valid_status, **changes}
                    (job.output_dir / "status.json").write_text(
                        json.dumps(status) + "\n", encoding="utf-8"
                    )
                    self.assertFalse(_bound_is_complete(job, manifest))

            (job.output_dir / "status.json").write_text(
                json.dumps(valid_status) + "\n", encoding="utf-8"
            )
            Path(valid_status["manifest_path"]).write_text(
                "{\"tampered\": true}\n", encoding="utf-8"
            )
            self.assertFalse(_bound_is_complete(job, manifest))

            _, manifest, valid_status = _write_completion_evidence(
                job, epoch_records=_valid_epoch_records(job.epochs)
            )
            Path(valid_status["fold_metrics_path"]).write_text(
                "[]\n", encoding="utf-8"
            )
            self.assertFalse(_bound_is_complete(job, manifest))

            _, manifest, valid_status = _write_completion_evidence(
                job, epoch_records=_valid_epoch_records(job.epochs)
            )
            Path(valid_status["result_archive_path"]).write_bytes(b"substitute")
            self.assertFalse(_bound_is_complete(job, manifest))

    def test_completion_rejects_manifest_epoch_and_provenance_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            job = runner.build_jobs(
                Path(directory), "/official/python", 3, epochs=2
            )[0]
            _, manifest, valid_status = _write_completion_evidence(
                job, epoch_records=_valid_epoch_records(job.epochs)
            )
            cases = {}
            wrong_epochs = copy.deepcopy(manifest)
            wrong_epochs["lifecycle"]["epochs_completed"] = 100
            cases["epochs"] = wrong_epochs
            wrong_cwd = copy.deepcopy(manifest)
            wrong_cwd["provenance"]["cwd"] = "/wrong/root"
            cases["cwd"] = wrong_cwd
            wrong_revision = copy.deepcopy(manifest)
            wrong_revision["provenance"]["git_revision"] = "b" * 40
            cases["revision"] = wrong_revision
            wrong_status = copy.deepcopy(manifest)
            wrong_status["provenance"]["git_status"] = "clean"
            cases["git_status"] = wrong_status
            wrong_command = copy.deepcopy(manifest)
            wrong_command["provenance"]["command"].append("--unexpected")
            cases["command"] = wrong_command

            for name, candidate in cases.items():
                with self.subTest(case=name):
                    (job.output_dir / "status.json").write_text(
                        json.dumps(valid_status) + "\n", encoding="utf-8"
                    )
                    self.assertFalse(_bound_is_complete(job, candidate))

            _, manifest, valid_status = _write_completion_evidence(
                job, epoch_records=_valid_epoch_records(job.epochs)
            )
            epoch_path = Path(valid_status["epoch_diagnostics_path"])
            epoch_path.write_text(
                epoch_path.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            self.assertFalse(_bound_is_complete(job, manifest))

    def test_physical_gpu_lease_is_exclusive_and_owner_released(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lease_root = Path(directory)
            lease = runner.acquire_gpu_lease(3, lease_root=lease_root)
            payload = json.loads(lease.path.read_text(encoding="utf-8"))
            self.assertEqual(payload["token"], lease.token)
            self.assertEqual(payload["host"], lease.host)
            self.assertEqual(payload["pid"], lease.pid)

            with self.assertRaisesRegex(RuntimeError, "GPU 3.*lease"):
                runner.acquire_gpu_lease(3, lease_root=lease_root)
            self.assertTrue(runner.release_gpu_lease(lease))
            self.assertTrue(lease.path.exists())

            successor = runner.acquire_gpu_lease(3, lease_root=lease_root)
            self.assertNotEqual(successor.token, lease.token)
            self.assertTrue(runner.release_gpu_lease(successor))
            self.assertTrue(successor.path.exists())

    def test_main_holds_lease_across_two_idle_checks_and_all_work(self) -> None:
        events: list[str] = []
        lease = object()
        future = mock.Mock()
        future.result.return_value = True

        class ImmediateExecutor:
            def __init__(self, max_workers):
                self.max_workers = max_workers

            def __enter__(self):
                events.append("workers")
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def submit(self, *args, **kwargs):
                return future

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner.sys,
            "argv",
            [
                "runner",
                "--output-root",
                str(Path(directory) / "outputs"),
                "--gpu",
                "3",
            ],
        ), mock.patch.object(
            runner,
            "acquire_gpu_lease",
            side_effect=lambda gpu: events.append("lease") or lease,
        ), mock.patch.object(
            runner,
            "release_gpu_lease",
            side_effect=lambda value: events.append("release") or True,
        ), mock.patch.object(
            runner,
            "assert_gpu_available",
            side_effect=lambda gpu: events.append("idle"),
        ), mock.patch.object(
            runner, "ThreadPoolExecutor", ImmediateExecutor
        ), mock.patch.object(
            runner,
            "audit_completed_pairs",
            side_effect=lambda *args: events.append("audit") or (3, 0),
        ), mock.patch.object(runner, "is_complete", return_value=True):
            result = runner.main()

        self.assertEqual(result, 0)
        self.assertEqual(
            events,
            ["lease", "idle", "idle", "workers", "audit", "release"],
        )

    def test_repository_root_is_anchored_and_mismatch_is_rejected(self) -> None:
        environment = runner._environment(runner.REPOSITORY_ROOT, 3)
        self.assertEqual(environment["PYTHONPATH"], str(runner.REPOSITORY_ROOT))
        self.assertEqual(
            environment["GCNET_DATASET_ROOT"],
            str(runner.REPOSITORY_ROOT / "dataset"),
        )

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner.subprocess,
            "run",
            return_value=subprocess.CompletedProcess([], 1),
        ) as launch:
            job = runner.build_jobs(
                Path(directory) / "outputs", "/official/python", 3
            )[0]
            with self.assertRaisesRegex(ValueError, "repository root"):
                runner.run_job(job, Path(directory), threading.Event())
            launch.assert_not_called()

    def test_three_completed_pairs_are_audited(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "outputs"
            jobs = runner.build_jobs(output_root, "/official/python", 3)
            manifests = {
                job.identity: job.output_dir / "manifest.json" for job in jobs
            }
            for manifest in manifests.values():
                manifest.parent.mkdir(parents=True, exist_ok=True)
                manifest.write_text("{}\n", encoding="utf-8")

            with mock.patch.object(runner, "is_complete", return_value=True), mock.patch.object(
                runner,
                "_bound_manifest_path",
                side_effect=lambda current_job: manifests[current_job.identity],
            ), mock.patch.object(
                runner.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0, stdout="ok\n"),
            ):
                audited, failures = runner.audit_completed_pairs(
                    jobs, runner.REPOSITORY_ROOT, "/official/python"
                )

            self.assertEqual((audited, failures), (3, 0))
            audit_logs = list(output_root.glob("**/paired_audit.log"))
            self.assertEqual(len(audit_logs), 3)


if __name__ == "__main__":
    unittest.main()
