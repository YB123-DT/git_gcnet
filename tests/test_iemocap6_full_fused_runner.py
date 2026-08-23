from __future__ import annotations

import importlib
import json
import subprocess
import threading
from pathlib import Path
from unittest import mock

import pytest

from gcnet_modality_jepa.run_manifest import MANIFEST_NAME, MANIFEST_VERSION


RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_iemocap6_full_fused_sweep.py"
)
EXPECTED_BASELINE_ROOT = Path(
    "/data2/yb/experiments/gcnet_official_4dataset_10seed_20260820/IEMOCAPSix"
)


def _runner():
    assert RUNNER_PATH.is_file(), "the isolated full-fused runner must exist"
    return importlib.import_module("scripts.run_iemocap6_full_fused_sweep")


def _manifest_for(
    job,
    metrics_path: Path,
    *,
    reconstruction_target: str = "full_fused",
    epochs: int | None = None,
) -> dict:
    epochs = job.epochs if epochs is None else epochs
    return {
        "schema": {"name": MANIFEST_NAME, "version": MANIFEST_VERSION},
        "run": {
            "dataset": "IEMOCAPSix",
            "fold": 5,
            "master_seed": job.seed,
        },
        "environment": {
            "python": "3.10.14",
            "torch": "2.2.2",
            "cuda": "12.1",
            "cudnn": 8902,
            "pyg": "2.5.3",
            "numpy": "1.26.4",
            "sklearn": "1.4.2",
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
        "split": {
            "indices": {
                "train": [0, 1],
                "validation": [2, 3],
                "test": [2, 3],
            },
            "hash": "d" * 64,
        },
        "samplers": {
            "train": {"seed": 101, "signature": "e" * 64},
            "validation": {"seed": 102, "signature": "f" * 64},
            "test": {"seed": 103, "signature": "0" * 64},
        },
        "masks": {
            "requested_missing_rate": job.rate,
            "config_hashes": {
                "train": "1" * 64,
                "validation": "2" * 64,
                "test": "3" * 64,
            },
            "realized_missing_rates": {
                "train": [job.rate],
                "validation": [job.rate],
                "test": [job.rate],
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
            "reconstruction_target": reconstruction_target,
        },
        "lifecycle": {
            "evaluation_protocol": "official",
            "checkpoint_metric": "validation_weighted_f1",
            "best_epoch": 1,
            "best_validation_f1": 0.71,
            "test_call_count": epochs,
            "epochs_completed": epochs,
        },
        "metrics": {"weighted_f1": 0.70, "accuracy": 0.69},
        "outputs": {
            "result_archive": str(metrics_path.parent / "result.npz"),
            "fold_metrics": str(metrics_path.resolve()),
            "archive_fold_index": 0,
        },
    }


def _write_run_evidence(
    job,
    output_dir: Path,
    *,
    reconstruction_target: str,
    status_identity: str,
    omit_manifest_target: bool = False,
    omit_metrics_target: bool = False,
    status_returncode: int = 0,
    manifest: dict | None = None,
) -> Path:
    record_root = output_dir / "run_records" / "123"
    metrics_path = record_root / "fold_metrics.json"
    record_root.mkdir(parents=True, exist_ok=True)
    metric = {"fold": 5}
    if not omit_metrics_target:
        metric["reconstruction_target"] = reconstruction_target
    metrics_path.write_text(json.dumps([metric]) + "\n", encoding="utf-8")
    if manifest is None:
        manifest = _manifest_for(
            job,
            metrics_path,
            reconstruction_target=reconstruction_target,
        )
    if omit_manifest_target:
        manifest["method"].pop("reconstruction_target", None)
    manifest_path = record_root / "run_manifest_fold_5.json"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    (output_dir / "status.json").write_text(
        json.dumps({"identity": status_identity, "returncode": status_returncode})
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _write_ffr_evidence(job, manifest: dict | None = None) -> Path:
    return _write_run_evidence(
        job,
        job.output_dir,
        reconstruction_target="full_fused",
        status_identity=job.identity,
        manifest=manifest,
    )


def _write_baseline_evidence(
    job,
    *,
    omit_manifest_target: bool = True,
    omit_metrics_target: bool = True,
    status_returncode: int = 0,
    manifest: dict | None = None,
) -> Path:
    return _write_run_evidence(
        job,
        job.baseline_dir,
        reconstruction_target="missing",
        status_identity="IEMOCAPSix:baseline:{:.1f}:{}".format(
            job.rate, job.seed
        ),
        omit_manifest_target=omit_manifest_target,
        omit_metrics_target=omit_metrics_target,
        status_returncode=status_returncode,
        manifest=manifest,
    )


def _job(
    tmp_path: Path,
    *,
    rate: float = 0.4,
    seed: int = 66,
    epochs: int = 100,
):
    sweep = _runner()
    return sweep.build_jobs(
        tmp_path / "outputs",
        python="/official/python",
        baseline_root=tmp_path / "baseline",
        gpus=(0,),
        jobs_per_gpu=2,
        rates=(rate,),
        seeds=(seed,),
        epochs=epochs,
    )[0]


def _tree_bytes(root: Path) -> dict[Path, bytes]:
    return {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_default_matrix_has_80_full_fused_jobs_and_80_pair_identities(tmp_path):
    sweep = _runner()
    jobs = sweep.build_jobs(tmp_path, python="/official/python")

    assert len(jobs) == 80
    assert len({job.identity for job in jobs}) == 80
    assert len({(job.rate, job.seed) for job in jobs}) == 80
    assert {job.condition for job in jobs} == {"full_fused"}
    assert {job.rate for job in jobs} == set(sweep.RATES)
    assert {job.seed for job in jobs} == set(range(66, 76))
    assert {job.gpu for job in jobs} == {0, 1, 2}
    assert {job.baseline_dir.parents[2] for job in jobs} == {
        EXPECTED_BASELINE_ROOT
    }


def test_subset_filters_produce_one_new_job_per_selected_pair(tmp_path):
    sweep = _runner()
    jobs = sweep.build_jobs(
        tmp_path / "outputs",
        python="python",
        baseline_root=tmp_path / "baseline",
        gpus=(2, 3),
        jobs_per_gpu=1,
        rates=(0.0, 0.4),
        seeds=(66, 70),
        epochs=2,
    )

    assert len(jobs) == 4
    assert {(job.rate, job.seed) for job in jobs} == {
        (0.0, 66),
        (0.0, 70),
        (0.4, 66),
        (0.4, 70),
    }
    assert {job.epochs for job in jobs} == {2}


def test_every_command_launches_only_the_full_fused_condition(tmp_path):
    job = _job(tmp_path, rate=0.7, seed=75)
    expected_options = {
        "--audio-feature": "wav2vec-large-c-UTT",
        "--text-feature": "deberta-large-4-UTT",
        "--video-feature": "manet_UTT",
        "--dataset": "IEMOCAPSix",
        "--base-model": "LSTM",
        "--windowp": "2",
        "--windowf": "2",
        "--hidden": "200",
        "--lr": "0.001",
        "--dropout": "0.5",
        "--batch-size": "32",
        "--num-threads": "4",
        "--epochs": "100",
        "--seed": "75",
        "--mask-type": "constant-0.7",
        "--evaluation-protocol": "official",
        "--fold": "5",
        "--stability-aux-mask-rate": "0.1",
        "--stability-recon-weight": "0.01",
        "--jepa-weight": "0",
        "--model-variant": "addon",
        "--reconstruction-target": "full_fused",
    }

    for option, expected in expected_options.items():
        assert job.command[job.command.index(option) + 1] == expected
    assert "--loss-recon" in job.command
    assert "--reccls" not in job.command
    assert "baseline" not in job.command
    assert job.output_dir.name == "full_fused"


def test_completion_accepts_only_matching_full_fused_evidence(tmp_path):
    sweep = _runner()
    job = _job(tmp_path)
    _write_ffr_evidence(job)
    assert sweep.is_complete(job)

    manifest_path = sweep._latest_manifest(job.output_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["method"]["reconstruction_target"] = "missing"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    assert not sweep.is_complete(job)


def test_status_without_a_matching_latest_fold5_manifest_is_rejected(tmp_path):
    sweep = _runner()
    job = _job(tmp_path)
    job.output_dir.mkdir(parents=True)
    (job.output_dir / "status.json").write_text(
        json.dumps({"identity": job.identity, "returncode": 0}) + "\n",
        encoding="utf-8",
    )
    assert not sweep.is_complete(job)


def test_legacy_baseline_omitted_targets_are_interpreted_as_missing(tmp_path):
    sweep = _runner()
    job = _job(tmp_path)
    _write_baseline_evidence(
        job, omit_manifest_target=True, omit_metrics_target=True
    )

    assert sweep.baseline_is_complete(job)
    assert sweep.validate_baselines([job]) == []


@pytest.mark.parametrize("failure", ["status", "manifest_target", "metrics"])
def test_baseline_preflight_rejects_incomplete_or_wrong_evidence(tmp_path, failure):
    sweep = _runner()
    job = _job(tmp_path)
    if failure == "status":
        _write_baseline_evidence(job, status_returncode=1)
    elif failure == "manifest_target":
        _write_baseline_evidence(
            job, omit_manifest_target=False, omit_metrics_target=True
        )
        manifest_path = sweep._latest_manifest(job.baseline_dir)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["method"]["reconstruction_target"] = "full_fused"
        manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    else:
        _write_baseline_evidence(job)
        (job.baseline_dir / "run_records" / "123" / "fold_metrics.json").unlink()

    assert not sweep.baseline_is_complete(job)
    assert sweep.validate_baselines([job]) == [job.identity]


def test_pair_audit_accepts_legacy_baseline_target_omission(tmp_path):
    sweep = _runner()
    job = _job(tmp_path)
    baseline_metrics = job.baseline_dir / "run_records" / "1" / "fold_metrics.json"
    fused_metrics = job.output_dir / "run_records" / "1" / "fold_metrics.json"
    baseline = _manifest_for(job, baseline_metrics, reconstruction_target="missing")
    baseline["method"].pop("reconstruction_target")
    full_fused = _manifest_for(job, fused_metrics)

    assert sweep.audit_pair_manifests(baseline, full_fused) == []


def test_pair_audit_rejects_initialization_hash_and_mask_mismatches(tmp_path):
    sweep = _runner()
    job = _job(tmp_path)
    baseline = _manifest_for(
        job,
        job.baseline_dir / "run_records" / "1" / "fold_metrics.json",
        reconstruction_target="missing",
    )
    full_fused = _manifest_for(
        job, job.output_dir / "run_records" / "1" / "fold_metrics.json"
    )
    full_fused["initialization"]["shared_hash"] = "9" * 64
    full_fused["masks"]["config_hashes"]["train"] = "8" * 64

    mismatches = sweep.audit_pair_manifests(baseline, full_fused)

    assert any("initialization.shared_hash" in item for item in mismatches)
    assert any("masks.config_hashes.train" in item for item in mismatches)


def test_pair_audit_rejects_wrong_condition_targets(tmp_path):
    sweep = _runner()
    job = _job(tmp_path)
    baseline = _manifest_for(
        job,
        job.baseline_dir / "run_records" / "1" / "fold_metrics.json",
        reconstruction_target="full_fused",
    )
    full_fused = _manifest_for(
        job,
        job.output_dir / "run_records" / "1" / "fold_metrics.json",
        reconstruction_target="missing",
    )

    mismatches = sweep.audit_pair_manifests(baseline, full_fused)

    assert any("baseline method.reconstruction_target" in item for item in mismatches)
    assert any("full_fused method.reconstruction_target" in item for item in mismatches)


def test_smoke_pair_audit_allows_only_lifecycle_epoch_count_difference(tmp_path):
    sweep = _runner()
    job = _job(tmp_path, epochs=1)
    baseline = _manifest_for(
        job,
        job.baseline_dir / "run_records" / "1" / "fold_metrics.json",
        reconstruction_target="missing",
        epochs=100,
    )
    full_fused = _manifest_for(
        job,
        job.output_dir / "run_records" / "1" / "fold_metrics.json",
        epochs=1,
    )

    assert sweep.audit_pair_manifests(
        baseline, full_fused, allow_epoch_mismatch=True
    ) == []
    full_fused["split"]["hash"] = "9" * 64
    mismatches = sweep.audit_pair_manifests(
        baseline, full_fused, allow_epoch_mismatch=True
    )
    assert any("split.hash" in item for item in mismatches)


def test_completed_pair_audit_never_writes_into_baseline_root(tmp_path):
    sweep = _runner()
    job = _job(tmp_path)
    _write_baseline_evidence(job)
    _write_ffr_evidence(job)
    before = _tree_bytes(job.baseline_dir.parents[2])

    audited, failures = sweep.audit_completed_pairs([job])

    assert (audited, failures) == (1, 0)
    assert _tree_bytes(job.baseline_dir.parents[2]) == before
    audit = json.loads(
        (job.output_dir.parent / "paired_audit.json").read_text(encoding="utf-8")
    )
    assert audit["passed"] is True


def test_gpu_and_lane_validation_rejects_unsafe_configurations(tmp_path):
    sweep = _runner()
    invalid = [
        ((4,), 1, "GPU 4"),
        ((0, 1, 2, 3), 1, "at most 3 GPUs"),
        ((0, 0), 1, "duplicates"),
        ((0,), 0, "jobs_per_gpu"),
        ((0,), 4, "jobs_per_gpu"),
    ]
    for gpus, jobs_per_gpu, message in invalid:
        with pytest.raises(ValueError, match=message):
            sweep.build_jobs(
                tmp_path,
                "python",
                gpus=gpus,
                jobs_per_gpu=jobs_per_gpu,
            )


def test_occupied_gpu_is_refused_without_process_termination():
    sweep = _runner()
    with mock.patch.object(sweep, "_gpu_memory_mb", side_effect=[100, 769]):
        with pytest.raises(RuntimeError, match=r"1.*769"):
            sweep.assert_gpus_available((0, 1))


def test_atomic_job_claim_is_exclusive_and_reusable(tmp_path):
    sweep = _runner()
    job = _job(tmp_path)

    first = sweep._acquire_claim(job)
    assert first is not None
    assert sweep._acquire_claim(job) is None
    sweep._release_claim(first)
    second = sweep._acquire_claim(job)
    assert second is not None
    sweep._release_claim(second)


def test_scheduler_success_uses_selected_dynamic_counts():
    sweep = _runner()
    assert sweep._scheduler_succeeded(True, 4, 4, 4, 4, 0)
    rejected = [
        (False, 4, 4, 4, 4, 0),
        (True, 3, 4, 4, 4, 0),
        (True, 4, 3, 4, 4, 0),
        (True, 4, 4, 3, 4, 0),
        (True, 4, 4, 4, 3, 0),
        (True, 4, 4, 4, 4, 1),
    ]
    assert not any(sweep._scheduler_succeeded(*arguments) for arguments in rejected)


def test_scheduler_writes_dynamic_selected_counts(tmp_path, monkeypatch):
    sweep = _runner()
    output_root = tmp_path / "outputs"
    monkeypatch.setattr(sweep, "validate_baselines", lambda _jobs: [])
    monkeypatch.setattr(sweep, "assert_gpus_available", lambda _gpus: None)
    monkeypatch.setattr(sweep, "_run_lane", lambda *_args: True)
    monkeypatch.setattr(sweep, "is_complete", lambda _job: True)
    monkeypatch.setattr(sweep, "audit_completed_pairs", lambda *_args: (4, 0))

    returncode = sweep.main(
        [
            "--output-root",
            str(output_root),
            "--baseline-root",
            str(tmp_path / "baseline"),
            "--gpus",
            "0,1",
            "--jobs-per-gpu",
            "1",
            "--rates",
            "0.0,0.4",
            "--seeds",
            "66,67",
            "--epochs",
            "1",
        ]
    )

    status = json.loads(
        (output_root / "scheduler_status.json").read_text(encoding="utf-8")
    )
    assert returncode == 0
    assert status["complete_jobs"] == status["total_jobs"] == 4
    assert status["paired_audits"] == status["expected_pair_audits"] == 4
    assert status["baseline_preflight_failures"] == []


def test_scheduler_refuses_invalid_baseline_before_gpu_or_training(
    tmp_path, monkeypatch
):
    sweep = _runner()
    output_root = tmp_path / "outputs"
    monkeypatch.setattr(sweep, "validate_baselines", lambda jobs: [jobs[0].identity])
    monkeypatch.setattr(
        sweep,
        "assert_gpus_available",
        lambda _gpus: pytest.fail("GPU checks must follow baseline preflight"),
    )

    returncode = sweep.main(
        [
            "--output-root",
            str(output_root),
            "--baseline-root",
            str(tmp_path / "baseline"),
            "--gpus",
            "0",
            "--rates",
            "0.4",
            "--seeds",
            "66",
        ]
    )

    status = json.loads(
        (output_root / "scheduler_status.json").read_text(encoding="utf-8")
    )
    assert returncode == 1
    assert len(status["baseline_preflight_failures"]) == 1
    assert not any(path.name == "full_fused" for path in output_root.rglob("*"))


def test_resume_preserves_valid_completed_job_evidence(tmp_path):
    sweep = _runner()
    job = _job(tmp_path)
    manifest_path = _write_ffr_evidence(job)
    status_path = job.output_dir / "status.json"
    before = (status_path.read_bytes(), manifest_path.read_bytes())

    with mock.patch.object(
        sweep.subprocess, "run", side_effect=AssertionError("must not rerun")
    ) as run:
        assert sweep.run_job(job, tmp_path, threading.Event())

    assert run.call_count == 0
    assert before == (status_path.read_bytes(), manifest_path.read_bytes())
    assert not (job.output_dir / "command.json").exists()


def test_successful_job_writes_command_and_matching_status_json(tmp_path):
    sweep = _runner()
    job = _job(tmp_path, epochs=1)
    captured = {}

    def fake_run(command, cwd, env, stdout, stderr):
        captured.update(command=command, cwd=cwd, env=env)
        _write_ffr_evidence(job)
        return subprocess.CompletedProcess(command, 0)

    with mock.patch.object(sweep.subprocess, "run", side_effect=fake_run):
        assert sweep.run_job(job, tmp_path, threading.Event())

    command = json.loads(
        (job.output_dir / "command.json").read_text(encoding="utf-8")
    )
    status = json.loads(
        (job.output_dir / "status.json").read_text(encoding="utf-8")
    )
    assert command["identity"] == status["identity"] == job.identity
    assert status["returncode"] == 0
    assert captured["cwd"] == str(tmp_path)
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == str(job.gpu)
    assert captured["env"]["GCNET_DATASET_ROOT"] == str(tmp_path / "dataset")
    assert captured["env"]["GCNET_CACHE_ROOT"] == "/data2/yb/gcnet_unified_cache"
    assert captured["env"]["PYTHONPATH"] == str(tmp_path)


def test_dry_run_writes_only_one_ffr_task_per_pair_under_output_root(
    tmp_path, monkeypatch
):
    sweep = _runner()
    output_root = tmp_path / "dry-run"
    baseline_root = tmp_path / "baseline"
    monkeypatch.setattr(
        sweep,
        "validate_baselines",
        lambda _jobs: pytest.fail("dry-run must not inspect baseline evidence"),
    )
    monkeypatch.setattr(
        sweep,
        "assert_gpus_available",
        lambda _gpus: pytest.fail("dry-run must not inspect GPUs"),
    )

    returncode = sweep.main(
        [
            "--output-root",
            str(output_root),
            "--baseline-root",
            str(baseline_root),
            "--gpus",
            "0",
            "--rates",
            "0.4",
            "--seeds",
            "66",
            "--epochs",
            "1",
            "--dry-run",
        ]
    )

    assert returncode == 0
    assert [path.relative_to(output_root) for path in output_root.rglob("*")] == [
        Path("task_manifest.json")
    ]
    assert not baseline_root.exists()
    tasks = json.loads(
        (output_root / "task_manifest.json").read_text(encoding="utf-8")
    )
    assert len(tasks) == 1
    assert tasks[0]["condition"] == "full_fused"
    assert tasks[0]["baseline_dir"].startswith(str(baseline_root))
    assert tasks[0]["command"][0] == (
        "/data2/yb/reproduction_envs/gcnet-official/bin/python"
    )
