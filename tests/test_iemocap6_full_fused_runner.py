from __future__ import annotations

import importlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
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


def _tree_state(root: Path) -> list[tuple[Path, bool, bytes | None]]:
    if not root.exists():
        return []
    return [
        (
            path.relative_to(root),
            path.is_dir(),
            None if path.is_dir() else path.read_bytes(),
        )
        for path in sorted(root.rglob("*"))
    ]


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
    for split_name in ("train", "validation", "test"):
        baseline["masks"]["realized_missing_rates"][split_name] = [0.4] * 100
        full_fused["masks"]["realized_missing_rates"][split_name] = [0.4]

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


def test_gpu_reservation_falls_back_and_holds_selected_lock_set(
    tmp_path, monkeypatch
):
    sweep = _runner()
    snapshot = {0: (), 1: (7001,), 2: (), 3: (), 4: (), 5: (7005,)}
    monkeypatch.setattr(
        sweep,
        "_gpu_process_snapshot",
        mock.Mock(side_effect=[snapshot, snapshot]),
    )

    reservation = sweep.acquire_gpu_reservation(
        (0, 1, 2),
        output_root=tmp_path / "output-a",
        lock_root=tmp_path / "gpu-locks",
    )

    assert reservation.gpus == (0, 3, 2)
    assert {claim.gpu for claim in reservation.claims} == {0, 2, 3}
    assert all(claim.path.exists() for claim in reservation.claims)
    assert sweep._try_acquire_gpu_claims(
        reservation.gpus,
        output_root=tmp_path / "output-b",
        lock_root=tmp_path / "gpu-locks",
    ) is None
    sweep.release_gpu_reservation(reservation)
    assert all(
        claim.path.read_text(encoding="utf-8") == ""
        for claim in reservation.claims
    )


def test_gpu_fallback_fails_when_too_few_idle_non_gpu4_devices(
    tmp_path, monkeypatch
):
    sweep = _runner()
    monkeypatch.setattr(
        sweep,
        "_gpu_process_snapshot",
        lambda: {0: (7000,), 1: (7001,), 2: (), 3: (7003,), 4: ()},
    )

    with pytest.raises(RuntimeError, match="insufficient idle GPUs"):
        sweep.acquire_gpu_reservation(
            (0, 1, 2),
            output_root=tmp_path / "output",
            lock_root=tmp_path / "gpu-locks",
        )


def test_gpu_reservation_releases_and_retries_when_occupancy_changes(
    tmp_path, monkeypatch
):
    sweep = _runner()
    changed = {0: (9000,), 1: (), 4: ()}
    snapshots = mock.Mock(
        side_effect=[
            {0: (), 1: (), 4: ()},
            changed,
            changed,
            changed,
        ]
    )
    monkeypatch.setattr(sweep, "_gpu_process_snapshot", snapshots)

    reservation = sweep.acquire_gpu_reservation(
        (0,),
        output_root=tmp_path / "output",
        lock_root=tmp_path / "gpu-locks",
    )

    assert reservation.gpus == (1,)
    gpu0_lock = tmp_path / "gpu-locks" / "gpu-0.lock"
    assert gpu0_lock.exists()
    assert gpu0_lock.read_text(encoding="utf-8") == ""
    sweep.release_gpu_reservation(reservation)


def test_different_output_roots_contend_on_shared_gpu_lock(tmp_path, monkeypatch):
    sweep = _runner()
    lock_root = tmp_path / "gpu-locks"
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    script = """
import sys
import time
from pathlib import Path
from scripts import run_iemocap6_full_fused_sweep as sweep
sweep._gpu_process_snapshot = lambda: {0: (), 1: (), 4: ()}
reservation = sweep.acquire_gpu_reservation(
    (0,), output_root=Path(sys.argv[1]), lock_root=Path(sys.argv[2])
)
print(reservation.gpus[0], flush=True)
time.sleep(60)
"""
    first = subprocess.Popen(
        [sys.executable, "-c", script, str(output_a), str(lock_root)],
        cwd=str(Path(__file__).resolve().parents[1]),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert first.stdout is not None
    assert first.stdout.readline().strip() == "0"
    monkeypatch.setattr(
        sweep, "_gpu_process_snapshot", lambda: {0: (), 1: (), 4: ()}
    )

    second = sweep.acquire_gpu_reservation(
        (0,), output_root=output_b, lock_root=lock_root
    )

    assert second.gpus == (1,)
    gpu0 = json.loads((lock_root / "gpu-0.lock").read_text(encoding="utf-8"))
    gpu1 = json.loads((lock_root / "gpu-1.lock").read_text(encoding="utf-8"))
    assert gpu0["output_root"] == str(output_a.resolve())
    assert gpu1["output_root"] == str(output_b.resolve())
    sweep.release_gpu_reservation(second)
    first.terminate()
    first.wait(timeout=10)


def test_gpu_process_snapshot_maps_compute_pids_to_indexes(monkeypatch):
    sweep = _runner()
    outputs = iter(
        [
            "0, GPU-a\n1, GPU-b\n4, GPU-four\n",
            "GPU-b, 7001\nGPU-b, 7002\n",
        ]
    )
    monkeypatch.setattr(
        sweep.subprocess, "check_output", lambda *_args, **_kwargs: next(outputs)
    )

    assert sweep._gpu_process_snapshot() == {
        0: (),
        1: (7001, 7002),
        4: (),
    }


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
    assert first.path.exists()
    assert first.path.read_text(encoding="utf-8") == ""


def test_job_claim_recovers_after_claiming_process_dies(tmp_path):
    sweep = _runner()
    job = _job(tmp_path)
    script = """
import sys
import time
from pathlib import Path
from scripts import run_iemocap6_full_fused_sweep as sweep
job = sweep.build_jobs(
    Path(sys.argv[1]), "python", baseline_root=Path(sys.argv[2]),
    gpus=(0,), rates=(0.4,), seeds=(66,),
)[0]
assert sweep._acquire_claim(job) is not None
print("claimed", flush=True)
time.sleep(60)
"""
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            script,
            str(tmp_path / "outputs"),
            str(tmp_path / "baseline"),
        ],
        cwd=str(Path(__file__).resolve().parents[1]),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert process.stdout is not None
    assert process.stdout.readline().strip() == "claimed"
    process.terminate()
    process.wait(timeout=10)

    recovered = sweep._acquire_claim(job)

    assert recovered is not None
    sweep._release_claim(recovered)
    assert recovered.path.exists()
    assert recovered.path.read_text(encoding="utf-8") == ""


def test_output_root_claim_is_process_locked_and_records_owner(tmp_path):
    sweep = _runner()
    output_root = tmp_path / "outputs"

    first = sweep._acquire_output_root_claim(output_root)
    payload = json.loads(first.path.read_text(encoding="utf-8"))

    assert payload["pid"] == os.getpid()
    assert payload["host"] == socket.gethostname()
    assert payload["token"] == first.token
    with pytest.raises(RuntimeError, match="already claimed"):
        sweep._acquire_output_root_claim(output_root)

    sweep._release_output_root_claim(first)
    assert first.path.exists()
    assert first.path.read_text(encoding="utf-8") == ""


def test_unlocked_stale_output_root_claim_is_safely_recovered(tmp_path):
    sweep = _runner()
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    claim_path = output_root / sweep.ROOT_CLAIM_FILE
    claim_path.write_text(
        json.dumps(
            {
                "host": socket.gethostname(),
                "pid": 999_999_999,
                "token": "stale-token",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    claim = sweep._acquire_output_root_claim(output_root)
    payload = json.loads(claim_path.read_text(encoding="utf-8"))

    assert payload["token"] == claim.token
    assert payload["token"] != "stale-token"
    sweep._release_output_root_claim(claim)
    assert claim_path.exists()
    assert claim_path.read_text(encoding="utf-8") == ""


def test_cross_process_waiter_acquires_same_root_lock_inode_after_release(tmp_path):
    sweep = _runner()
    output_root = tmp_path / "outputs"
    first = sweep._acquire_output_root_claim(output_root)
    first_inode = first.path.stat().st_ino
    ready_path = tmp_path / "waiter-ready"
    acquired_path = tmp_path / "waiter-acquired"
    script = """
import fcntl
import os
import sys
from pathlib import Path
path, ready, acquired = map(Path, sys.argv[1:])
descriptor = os.open(str(path), os.O_RDWR)
ready.write_text(str(os.fstat(descriptor).st_ino), encoding="utf-8")
fcntl.flock(descriptor, fcntl.LOCK_EX)
acquired.write_text(str(os.fstat(descriptor).st_ino), encoding="utf-8")
fcntl.flock(descriptor, fcntl.LOCK_UN)
os.close(descriptor)
"""
    waiter = subprocess.Popen(
        [
            sys.executable,
            "-c",
            script,
            str(first.path),
            str(ready_path),
            str(acquired_path),
        ]
    )
    for _ in range(100):
        if ready_path.exists():
            break
        time.sleep(0.01)
    assert ready_path.exists()
    assert int(ready_path.read_text(encoding="utf-8")) == first_inode

    sweep._release_output_root_claim(first)
    waiter.wait(timeout=10)

    assert waiter.returncode == 0
    assert int(acquired_path.read_text(encoding="utf-8")) == first_inode
    assert first.path.exists()
    assert first.path.stat().st_ino == first_inode


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
    reservation = sweep.GPUReservation(gpus=(0, 3), claims=())
    acquire = mock.Mock(return_value=reservation)
    release = mock.Mock()
    final_idle = mock.Mock(return_value=True)
    monkeypatch.setattr(sweep, "acquire_gpu_reservation", acquire)
    monkeypatch.setattr(sweep, "release_gpu_reservation", release)
    monkeypatch.setattr(
        sweep, "_gpu_reservation_is_idle", final_idle, raising=False
    )
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
    assert status["requested_gpus"] == [0, 1]
    assert status["selected_gpus"] == [0, 3]
    assert acquire.call_count == 1
    final_idle.assert_called_once_with(reservation)
    release.assert_called_once_with(reservation)
    tasks = json.loads(
        (output_root / "task_manifest.json").read_text(encoding="utf-8")
    )
    assert {task["gpu"] for task in tasks} == {0, 3}
    root_claim_path = output_root / sweep.ROOT_CLAIM_FILE
    assert root_claim_path.exists()
    assert root_claim_path.read_text(encoding="utf-8") == ""


def test_scheduler_reacquires_fallback_if_final_gpu_check_changes(
    tmp_path, monkeypatch
):
    sweep = _runner()
    output_root = tmp_path / "outputs"
    first = sweep.GPUReservation(gpus=(0,), claims=())
    fallback = sweep.GPUReservation(gpus=(1,), claims=())
    acquire = mock.Mock(side_effect=[first, fallback])
    release = mock.Mock()
    final_idle = mock.Mock(side_effect=[False, True])
    monkeypatch.setattr(sweep, "validate_baselines", lambda _jobs: [])
    monkeypatch.setattr(sweep, "acquire_gpu_reservation", acquire)
    monkeypatch.setattr(sweep, "release_gpu_reservation", release)
    monkeypatch.setattr(
        sweep, "_gpu_reservation_is_idle", final_idle, raising=False
    )
    monkeypatch.setattr(sweep, "_run_lane", lambda *_args: True)
    monkeypatch.setattr(sweep, "is_complete", lambda _job: True)
    monkeypatch.setattr(sweep, "audit_completed_pairs", lambda *_args: (1, 0))

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
    assert returncode == 0
    assert status["selected_gpus"] == [1]
    assert acquire.call_count == 2
    assert final_idle.call_count == 2
    assert release.call_args_list == [mock.call(first), mock.call(fallback)]


def test_scheduler_refuses_invalid_baseline_before_gpu_or_training(
    tmp_path, monkeypatch
):
    sweep = _runner()
    output_root = tmp_path / "outputs"
    monkeypatch.setattr(sweep, "validate_baselines", lambda jobs: [jobs[0].identity])
    monkeypatch.setattr(
        sweep,
        "acquire_gpu_reservation",
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
    root_claim_path = output_root / sweep.ROOT_CLAIM_FILE
    assert root_claim_path.exists()
    assert root_claim_path.read_text(encoding="utf-8") == ""


def test_concurrent_invocation_fails_without_mutating_root_evidence(
    tmp_path, monkeypatch
):
    sweep = _runner()
    output_root = tmp_path / "outputs"
    first = sweep._acquire_output_root_claim(output_root)
    task_path = output_root / "task_manifest.json"
    task_path.write_text('{"owner":"first"}\n', encoding="utf-8")
    before = _tree_bytes(output_root)
    monkeypatch.setattr(
        sweep,
        "acquire_gpu_reservation",
        lambda _gpus: pytest.fail("second invocation must stop at root claim"),
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

    assert returncode == 1
    assert _tree_bytes(output_root) == before
    sweep._release_output_root_claim(first)


def test_output_root_claim_remains_held_through_pair_audit(tmp_path, monkeypatch):
    sweep = _runner()
    output_root = tmp_path / "outputs"
    gpu_lock_root = tmp_path / "gpu-locks"
    monkeypatch.setattr(sweep, "GPU_LOCK_ROOT", gpu_lock_root)
    monkeypatch.setattr(sweep, "validate_baselines", lambda _jobs: [])
    monkeypatch.setattr(sweep, "_gpu_process_snapshot", lambda: {0: (), 4: ()})
    monkeypatch.setattr(sweep, "_run_lane", lambda *_args: True)
    monkeypatch.setattr(sweep, "is_complete", lambda _job: True)

    def audit_while_locked(jobs):
        with pytest.raises(RuntimeError, match="already claimed"):
            sweep._acquire_output_root_claim(output_root)
        assert sweep._try_acquire_gpu_claims(
            (0,),
            output_root=tmp_path / "other-output",
            lock_root=gpu_lock_root,
        ) is None
        return len(jobs), 0

    monkeypatch.setattr(sweep, "audit_completed_pairs", audit_while_locked)

    assert sweep.main(
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
    ) == 0
    root_claim_path = output_root / sweep.ROOT_CLAIM_FILE
    assert root_claim_path.exists()
    assert root_claim_path.read_text(encoding="utf-8") == ""
    gpu_claim_path = gpu_lock_root / "gpu-0.lock"
    assert gpu_claim_path.exists()
    assert gpu_claim_path.read_text(encoding="utf-8") == ""


def test_scheduler_aborts_if_gpu_reservation_fails_before_workers(
    tmp_path, monkeypatch
):
    sweep = _runner()
    output_root = tmp_path / "outputs"
    monkeypatch.setattr(sweep, "validate_baselines", lambda _jobs: [])
    acquire = mock.Mock(
        side_effect=RuntimeError("insufficient idle GPUs after lock revalidation")
    )
    monkeypatch.setattr(sweep, "acquire_gpu_reservation", acquire)
    monkeypatch.setattr(
        sweep,
        "_run_lane",
        lambda *_args: pytest.fail(
            "workers must not start after GPU revalidation fails"
        ),
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
    assert acquire.call_count == 1
    assert "lock revalidation" in status["gpu_selection_error"]
    assert (output_root / sweep.ROOT_CLAIM_FILE).exists()


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


def test_environment_preserves_explicit_dataset_root(tmp_path, monkeypatch):
    sweep = _runner()
    explicit_root = tmp_path / "shared-dataset"
    monkeypatch.setenv("GCNET_DATASET_ROOT", str(explicit_root))

    environment = sweep._environment(tmp_path, gpu=0)

    assert environment["GCNET_DATASET_ROOT"] == str(explicit_root)


@pytest.mark.parametrize("relationship", ["equal", "output_inside", "baseline_inside"])
def test_overlapping_output_and_baseline_roots_are_rejected_before_writes(
    tmp_path, monkeypatch, relationship
):
    sweep = _runner()
    if relationship == "equal":
        output_root = baseline_root = tmp_path / "shared"
    elif relationship == "output_inside":
        baseline_root = tmp_path / "baseline"
        output_root = baseline_root / "new-output"
    else:
        output_root = tmp_path / "output"
        baseline_root = output_root / "baseline"
    baseline_root.mkdir(parents=True, exist_ok=True)
    (baseline_root / "sentinel.txt").write_text("unchanged\n", encoding="utf-8")
    before = _tree_state(tmp_path)
    monkeypatch.setattr(
        sweep,
        "_acquire_output_root_claim",
        lambda _root: pytest.fail("overlap must fail before root claim"),
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
        ]
    )

    assert returncode == 1
    assert _tree_state(tmp_path) == before


def test_dry_run_prints_plan_without_mutating_live_output_root(
    tmp_path, monkeypatch, capsys
):
    sweep = _runner()
    output_root = tmp_path / "dry-run"
    baseline_root = tmp_path / "baseline"
    output_root.mkdir()
    (output_root / "task_manifest.json").write_text(
        '{"live":"task"}\n', encoding="utf-8"
    )
    (output_root / "scheduler_status.json").write_text(
        '{"live":"status"}\n', encoding="utf-8"
    )
    (output_root / sweep.ROOT_CLAIM_FILE).write_text(
        "permanent-lock-inode\n", encoding="utf-8"
    )
    before = _tree_state(output_root)
    monkeypatch.setattr(
        sweep,
        "validate_baselines",
        lambda _jobs: pytest.fail("dry-run must not inspect baseline evidence"),
    )
    monkeypatch.setattr(
        sweep,
        "acquire_gpu_reservation",
        lambda _gpus: pytest.fail("dry-run must not inspect GPUs"),
        raising=False,
    )
    monkeypatch.setattr(
        sweep,
        "_acquire_output_root_claim",
        lambda _root: pytest.fail("dry-run must not claim output root"),
        raising=False,
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
    assert _tree_state(output_root) == before
    assert not baseline_root.exists()
    printed = capsys.readouterr().out
    plan = json.loads(printed)
    assert len(plan["jobs"]) == 1
    assert plan["jobs"][0]["condition"] == "full_fused"
    assert plan["jobs"][0]["baseline_dir"].startswith(str(baseline_root))
    assert plan["jobs"][0]["command"][0] == (
        "/data2/yb/reproduction_envs/gcnet-official/bin/python"
    )
