from collections import Counter
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest


RATES = tuple(format(index / 10, ".1f") for index in range(8))


def test_runner_locks_counts_observed_with_formal_512_1024_1024_features():
    from gcnet_missing_m3_sdr_backbone import run_mosi

    assert run_mosi.EXPECTED_PARAMETER_COUNTS == {
        "sdr-public": {
            "registered_parameters": 12_486_434,
            "trainable_parameters": 11_626_274,
            "registered_backbone_parameters": 9_444_901,
            "trainable_backbone_parameters": 9_444_901,
        },
        "sdr-paper": {
            "registered_parameters": 21_079_835,
            "trainable_parameters": 20_219_675,
            "registered_backbone_parameters": 18_038_302,
            "trainable_backbone_parameters": 18_038_302,
        },
    }


def test_runner_builds_exactly_two_variants_by_five_seeds(tmp_path):
    from gcnet_missing_m3_sdr_backbone import run_mosi

    jobs = run_mosi.build_jobs(output_root=tmp_path)

    assert len(jobs) == 10
    assert {(job.variant, job.seed) for job in jobs} == {
        (variant, seed)
        for variant in ("sdr-public", "sdr-paper")
        for seed in (66, 67, 68, 69, 70)
    }
    assert Counter(job.gpu for job in jobs) == {2: 2, 3: 2, 5: 2, 6: 2, 7: 2}
    for seed in (66, 67, 68, 69, 70):
        assert len({job.gpu for job in jobs if job.seed == seed}) == 1
    assert [job.output_dir for job in jobs[:5]] == [
        tmp_path / "sdr-public" / "seed_{}".format(seed)
        for seed in (66, 67, 68, 69, 70)
    ]


@pytest.mark.parametrize("gpus", [(2, 3, 4, 6, 7), (0, 2, 3, 5, 6)])
def test_runner_rejects_gpu4_and_any_gpu_outside_healthy_set(tmp_path, gpus):
    from gcnet_missing_m3_sdr_backbone import run_mosi

    with pytest.raises(ValueError):
        run_mosi.build_jobs(output_root=tmp_path, gpus=gpus)


def test_runner_accepts_an_explicit_nonempty_unique_healthy_gpu_subset(tmp_path):
    from gcnet_missing_m3_sdr_backbone import run_mosi

    jobs = run_mosi.build_jobs(output_root=tmp_path, gpus=(2, 5, 7))

    assert {job.gpu for job in jobs} == {2, 5, 7}
    for seed in run_mosi.SEEDS:
        assert len({job.gpu for job in jobs if job.seed == seed}) == 1

    for invalid in ((), (2, 2), (2, True)):
        with pytest.raises(ValueError):
            run_mosi.build_jobs(output_root=tmp_path, gpus=invalid)


def test_command_is_fixed_to_official_python_features_and_protocol(tmp_path):
    from gcnet_missing_m3_sdr_backbone import run_mosi

    job = run_mosi.build_jobs(output_root=tmp_path)[0]
    command = run_mosi.build_command(job, feature_root=Path("/features"))
    joined = " ".join(command)

    assert command[:3] == [
        "/data2/yb/reproduction_envs/gcnet-official/bin/python",
        "-m",
        "gcnet_missing_m3_sdr_backbone.train_gcnet",
    ]
    for expected in (
        "--dataset CMUMOSI",
        "--feature-root /features",
        "--audio-feature wav2vec-large-c-UTT",
        "--text-feature deberta-large-4-UTT",
        "--video-feature manet_UTT",
        "--epochs 100",
        "--train-rate-mode all",
        "--fusion-type slot",
        "--lr 0.0005",
        "--device cuda",
        "--sdr-variant sdr-public",
        "--seed 66",
    ):
        assert expected in joined
    assert str(job.output_dir) in command
    assert "gcnet_missing_m3.train_gcnet" not in command
    assert "original" not in joined.lower()
    assert "control" not in joined.lower()
    assert "--skip-test" not in command


def _write_complete_result(
    job,
    *,
    validation_offset=0.0,
    test_offset=0.0,
    validation_only=False,
):
    from gcnet_missing_m3 import train_gcnet as base_train
    from gcnet_missing_m3_sdr_backbone import run_mosi
    from gcnet_missing_m3_sdr_backbone.train_gcnet import SDRTrainConfig

    job.output_dir.mkdir(parents=True, exist_ok=True)
    config = asdict(
        SDRTrainConfig(
            seed=job.seed,
            sdr_variant=job.variant,
            evaluate_test=not validation_only,
        )
    )
    history = []
    best_validation = None
    for epoch in range(1, 101):
        epoch_offset = 0.1 if epoch == 37 else 0.0
        validation = {
            rate: {
                "weighted_f1": (
                    0.60
                    + validation_offset
                    + epoch_offset
                    - float(rate) / 100.0
                ),
                "loss": 0.2,
            }
            for rate in RATES
        }
        mean = sum(
            value["weighted_f1"] for value in validation.values()
        ) / len(RATES)
        history.append(
            {
                "epoch": epoch,
                "train": {"weighted_f1": 0.5},
                "validation": validation,
                "validation_mean_weighted_f1": mean,
            }
        )
        if epoch == 37:
            best_validation = validation
            best_mean = mean
    assert best_validation is not None
    test = {}
    mask_hashes = {}
    if not validation_only:
        for index, rate in enumerate(RATES, start=1):
            labels = np.array(
                [0.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
                dtype=np.float32,
            )
            if test_offset < 0.0:
                predictions = np.array(
                    [100.0 + index, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0],
                    dtype=np.float32,
                )
            else:
                predictions = np.array(
                    [100.0 + index, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
                    dtype=np.float32,
                )
            code = (job.seed - 66) * len(RATES) + index
            availability = np.zeros((labels.size, 3), dtype=np.float32)
            for bit in range(availability.size):
                availability.reshape(-1)[bit] = float((code >> bit) & 1)
            mask_hash = hashlib.sha256(
                np.ascontiguousarray(availability).tobytes()
            ).hexdigest()
            recomputed = base_train._metrics(
                "CMUMOSI",
                labels,
                predictions,
                "regression",
            )
            test[rate] = {
                "weighted_f1": recomputed["weighted_f1"],
                "loss": 0.3,
                "prediction_std": recomputed["prediction_std"],
                "predicted_sign_count": recomputed["predicted_sign_count"],
                "mask_sha256": mask_hash,
            }
            mask_hashes[rate] = mask_hash
            np.savez_compressed(
                job.output_dir
                / "predictions_miss_{}.npz".format(rate.replace(".", "p")),
                predictions=predictions,
                labels=labels,
                availability=availability,
            )
    counts = run_mosi.EXPECTED_PARAMETER_COUNTS[job.variant]
    metrics = {
        "backbone": "sdr-gnn-whole-backbone",
        "variant": job.variant,
        "sdr_variant": job.variant,
        "best_epoch": 37,
        "best_validation": best_validation,
        "best_validation_mean_weighted_f1": best_mean,
        "selection_missing_rates": [index / 10 for index in range(8)],
        "evaluation_stage": (
            "train-validation-only"
            if validation_only
            else "train-validation-test"
        ),
        "test": test,
        "mask_sha256": mask_hashes,
        "registered_parameters": counts["registered_parameters"],
        "trainable_parameters": counts["trainable_parameters"],
        "registered_backbone_parameters": counts[
            "registered_backbone_parameters"
        ],
        "trainable_backbone_parameters": counts[
            "trainable_backbone_parameters"
        ],
        "ema_steps": 100,
        "wall_time_seconds": 123.0,
        "peak_memory_bytes": 1_000_000,
    }
    for name, payload in (
        ("config.json", config),
        ("history.json", history),
        ("metrics.json", metrics),
    ):
        (job.output_dir / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    (job.output_dir / "train.log").touch()


def _make_rate_collapsed(job, rate="0.4"):
    from gcnet_missing_m3 import train_gcnet as base_train

    archive_path = job.output_dir / "predictions_miss_{}.npz".format(
        rate.replace(".", "p")
    )
    with np.load(archive_path, allow_pickle=False) as archive:
        labels = archive["labels"]
        availability = archive["availability"]
    predictions = np.ones_like(labels)
    np.savez_compressed(
        archive_path,
        predictions=predictions,
        labels=labels,
        availability=availability,
    )
    recomputed = base_train._metrics(
        "CMUMOSI", labels, predictions, "regression"
    )
    metrics = json.loads((job.output_dir / "metrics.json").read_text())
    metrics["test"][rate].update(
        weighted_f1=recomputed["weighted_f1"],
        prediction_std=recomputed["prediction_std"],
        predicted_sign_count=recomputed["predicted_sign_count"],
    )
    (job.output_dir / "metrics.json").write_text(json.dumps(metrics))


def test_result_completion_requires_exact_semantics_and_all_prediction_archives(
    tmp_path,
):
    from gcnet_missing_m3_sdr_backbone import run_mosi

    jobs = run_mosi.build_jobs(output_root=tmp_path)
    job = jobs[0]
    assert run_mosi.inspect_result(job).complete is False
    _write_complete_result(job)
    inspection = run_mosi.inspect_result(job)
    assert inspection.complete is True
    assert inspection.reason == "complete-test-8-rates"

    required = ("config.json", "history.json", "metrics.json", "train.log")
    for name in required:
        path = job.output_dir / name
        saved = path.read_bytes()
        path.unlink()
        assert run_mosi.inspect_result(job).complete is False
        path.write_bytes(saved)

    archive = job.output_dir / "predictions_miss_0p7.npz"
    saved_archive = archive.read_bytes()
    archive.unlink()
    assert "prediction" in run_mosi.inspect_result(job).reason
    archive.write_bytes(saved_archive)
    archive.write_bytes(b"half-written")
    assert "prediction" in run_mosi.inspect_result(job).reason


def test_collapse_is_complete_scientific_state_and_is_not_rescheduled(tmp_path):
    from gcnet_missing_m3_sdr_backbone import run_mosi

    job = run_mosi.build_jobs(output_root=tmp_path)[0]
    _write_complete_result(job)
    rate = "0.4"
    _make_rate_collapsed(job, rate)

    inspection = run_mosi.inspect_result(job)

    assert inspection.complete is True
    assert inspection.collapsed is True
    assert inspection.collapse_rates == (rate,)
    assert run_mosi.pending_jobs([job]) == []


def test_npz_metrics_are_independently_recomputed_with_mosi_nonzero_rule(tmp_path):
    from gcnet_missing_m3 import train_gcnet as base_train
    from gcnet_missing_m3_sdr_backbone import run_mosi

    job = run_mosi.build_jobs(output_root=tmp_path)[0]
    _write_complete_result(job)
    archive_path = job.output_dir / "predictions_miss_0p2.npz"
    with np.load(archive_path, allow_pickle=False) as archive:
        labels = archive["labels"]
        predictions = archive["predictions"]
    assert labels[0] == 0.0 and predictions[0] > 0.0
    assert base_train._metrics(
        "CMUMOSI", labels, predictions, "regression"
    )["weighted_f1"] == pytest.approx(1.0)

    metrics = json.loads((job.output_dir / "metrics.json").read_text())
    metrics["test"]["0.2"]["weighted_f1"] = 0.123
    (job.output_dir / "metrics.json").write_text(json.dumps(metrics))

    inspection = run_mosi.inspect_result(job)
    assert inspection.complete is False
    assert "recomputed" in inspection.reason


def test_npz_availability_sha_is_independent_and_rate_or_job_swaps_fail(tmp_path):
    from gcnet_missing_m3_sdr_backbone import run_mosi

    jobs = run_mosi.build_jobs(output_root=tmp_path)
    first, second = jobs[0], jobs[1]
    _write_complete_result(first)
    _write_complete_result(second)

    metrics = json.loads((first.output_dir / "metrics.json").read_text())
    metrics["test"]["0.3"]["mask_sha256"] = "f" * 64
    metrics["mask_sha256"]["0.3"] = "f" * 64
    (first.output_dir / "metrics.json").write_text(json.dumps(metrics))
    assert run_mosi.inspect_result(first).complete is False

    _write_complete_result(first)
    rate_a = first.output_dir / "predictions_miss_0p1.npz"
    rate_b = first.output_dir / "predictions_miss_0p2.npz"
    bytes_a, bytes_b = rate_a.read_bytes(), rate_b.read_bytes()
    rate_a.write_bytes(bytes_b)
    rate_b.write_bytes(bytes_a)
    assert run_mosi.inspect_result(first).complete is False

    _write_complete_result(first)
    job_a = first.output_dir / "predictions_miss_0p5.npz"
    job_b = second.output_dir / "predictions_miss_0p5.npz"
    bytes_a, bytes_b = job_a.read_bytes(), job_b.read_bytes()
    job_a.write_bytes(bytes_b)
    job_b.write_bytes(bytes_a)
    assert run_mosi.inspect_result(first).complete is False
    assert run_mosi.inspect_result(second).complete is False


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda c, h, m: c.update(seed=999), "config"),
        (lambda c, h, m: h.pop(), "100 epochs"),
        (lambda c, h, m: m.update(variant="sdr-paper"), "variant"),
        (lambda c, h, m: m["test"].pop("0.7"), "8 test rates"),
        (
            lambda c, h, m: m["test"]["0.4"].update(prediction_std=0.0),
            "semantic",
        ),
        (
            lambda c, h, m: m["test"]["0.4"].update(predicted_sign_count=1),
            "semantic",
        ),
        (
            lambda c, h, m: m["mask_sha256"].update({"0.4": "f" * 64}),
            "semantic",
        ),
        (lambda c, h, m: m.update(best_epoch=38), "validation-selected"),
        (
            lambda c, h, m: m.update(selection_missing_rates=[0.0]),
            "selection",
        ),
    ],
)
def test_result_completion_rejects_half_written_or_semantically_wrong_outputs(
    tmp_path,
    mutation,
    reason,
):
    from gcnet_missing_m3_sdr_backbone import run_mosi

    job = run_mosi.build_jobs(output_root=tmp_path)[0]
    _write_complete_result(job)
    config = json.loads((job.output_dir / "config.json").read_text())
    history = json.loads((job.output_dir / "history.json").read_text())
    metrics = json.loads((job.output_dir / "metrics.json").read_text())
    mutation(config, history, metrics)
    (job.output_dir / "config.json").write_text(json.dumps(config))
    (job.output_dir / "history.json").write_text(json.dumps(history))
    (job.output_dir / "metrics.json").write_text(json.dumps(metrics))

    inspection = run_mosi.inspect_result(job)
    assert inspection.complete is False
    assert reason in inspection.reason


def test_validation_only_is_not_resumable_and_pending_uses_content(tmp_path):
    from gcnet_missing_m3_sdr_backbone import run_mosi

    jobs = run_mosi.build_jobs(output_root=tmp_path)
    _write_complete_result(jobs[0])
    _write_complete_result(jobs[1], validation_only=True)

    assert run_mosi.inspect_result(jobs[1]).has_test_metrics is False
    assert run_mosi.pending_jobs(jobs) == jobs[1:]


def test_waves_keep_variants_of_each_seed_on_the_same_gpu_in_two_safe_waves(
    tmp_path,
):
    from gcnet_missing_m3_sdr_backbone import run_mosi

    jobs = run_mosi.build_jobs(output_root=tmp_path)
    one_per_gpu = run_mosi.build_waves(jobs, jobs_per_gpu=1)
    two_per_gpu = run_mosi.build_waves(jobs, jobs_per_gpu=2)

    assert len(one_per_gpu) == 2
    assert all(len(wave) == 5 for wave in one_per_gpu)
    assert {job.variant for job in one_per_gpu[0]} == {"sdr-public"}
    assert {job.variant for job in one_per_gpu[1]} == {"sdr-paper"}
    assert len(two_per_gpu) == 1
    assert Counter(job.gpu for job in two_per_gpu[0]) == {
        2: 2,
        3: 2,
        5: 2,
        6: 2,
        7: 2,
    }
    for invalid in (0, 3, True):
        with pytest.raises(ValueError, match="jobs_per_gpu"):
            run_mosi.build_waves(jobs, jobs_per_gpu=invalid)


def test_manifest_is_atomic_and_binds_source_results_environment_and_status(
    tmp_path,
):
    from gcnet_missing_m3_sdr_backbone import run_mosi

    jobs = run_mosi.build_jobs(output_root=tmp_path / "formal")
    _write_complete_result(jobs[0])
    (jobs[0].output_dir / "status.json").write_text(
        json.dumps({"state": "complete", "returncode": 0})
    )
    manifest_path = tmp_path / "formal" / "manifest.json"
    source_commit = "a" * 40
    repo_root = Path(run_mosi.__file__).resolve().parents[1]

    run_mosi.write_manifest(
        manifest_path,
        jobs,
        source_commit=source_commit,
        feature_root=Path("/features"),
        repo_root=repo_root,
    )
    first = json.loads(manifest_path.read_text())
    run_mosi.write_manifest(
        manifest_path,
        jobs,
        source_commit=source_commit,
        feature_root=Path("/features"),
        repo_root=repo_root,
    )
    assert json.loads(manifest_path.read_text()) == first

    assert first["source_commit"] == source_commit
    assert first["variants"] == ["sdr-public", "sdr-paper"]
    assert first["seeds"] == [66, 67, 68, 69, 70]
    assert first["features"] == {
        "root": "/features",
        "audio": "/features/wav2vec-large-c-UTT",
        "text": "/features/deberta-large-4-UTT",
        "video": "/features/manet_UTT",
    }
    assert first["runtime"]["training_python_executable"] == str(
        run_mosi.DEFAULT_PYTHON
    )
    assert first["runtime"]["training"]["python_version"].startswith("3.8")
    assert first["runtime"]["training"]["torch_version"] == "1.8.0"
    assert "cuda_version" in first["runtime"]["training"]
    assert "gpu_names" in first["runtime"]["training"]
    assert first["runtime"]["runner"]["python_version"].startswith("3.10")
    assert set(first["source_files_sha256"]) == set(run_mosi.SOURCE_FILES)
    assert all(len(value) == 64 for value in first["source_files_sha256"].values())
    assert len(first["jobs"]) == 10
    complete = first["jobs"][0]
    incomplete = first["jobs"][1]
    assert complete["complete"] is True
    assert len(complete["config_sha256"]) == 64
    expected_config = asdict(
        run_mosi.SDRTrainConfig(seed=66, sdr_variant="sdr-public")
    )
    expected_config_sha = hashlib.sha256(
        (json.dumps(expected_config, indent=2, sort_keys=True) + "\n").encode()
    ).hexdigest()
    assert complete["config_sha256"] == expected_config_sha
    assert complete["config_file_sha256"] == expected_config_sha
    assert len(incomplete["config_sha256"]) == 64
    assert incomplete["config_file_sha256"] is None
    assert len(complete["metrics_sha256"]) == 64
    assert complete["status"] == {"state": "complete", "returncode": 0}
    assert set(complete["mask_sha256"]) == set(RATES)
    assert not list(manifest_path.parent.glob("*.tmp"))
    encoded = manifest_path.read_text().lower()
    assert "original" not in encoded
    assert "control" not in encoded

    with pytest.raises(ValueError, match="full 40-character"):
        run_mosi.write_manifest(
            manifest_path,
            jobs,
            source_commit="abc",
            feature_root=Path("/features"),
            repo_root=repo_root,
        )


def test_resume_skips_only_semantically_complete_jobs(monkeypatch, tmp_path):
    from gcnet_missing_m3_sdr_backbone import run_mosi

    jobs = run_mosi.build_jobs(output_root=tmp_path)
    complete, incomplete = jobs[:2]
    _write_complete_result(complete)
    _write_complete_result(incomplete)
    (incomplete.output_dir / "history.json").write_text("[]")
    launched = []

    class FakeProcess:
        pid = 12345

        def wait(self, timeout=None):
            assert timeout > 0
            _write_complete_result(incomplete)
            return 0

    def popen(command, **kwargs):
        launched.append((command, kwargs))
        return FakeProcess()

    monkeypatch.setattr(run_mosi.subprocess, "Popen", popen)
    failures = run_mosi.run_jobs(
        [complete, incomplete],
        feature_root=Path("/features"),
        repo_root=tmp_path,
        jobs_per_gpu=1,
        timeout_seconds=123,
    )

    assert failures == 0
    assert len(launched) == 1
    assert launched[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == str(incomplete.gpu)
    assert launched[0][1]["start_new_session"] is True
    assert run_mosi.inspect_result(incomplete).complete is True
    status = json.loads((incomplete.output_dir / "status.json").read_text())
    assert status["state"] == "complete"
    assert not list(incomplete.output_dir.glob("*.tmp"))


def test_rerun_clears_half_written_artifacts_before_child_failure(
    monkeypatch,
    tmp_path,
):
    from gcnet_missing_m3_sdr_backbone import run_mosi

    job = run_mosi.build_jobs(output_root=tmp_path)[0]
    _write_complete_result(job)
    config = json.loads((job.output_dir / "config.json").read_text())
    config["hidden"] = 100
    (job.output_dir / "config.json").write_text(json.dumps(config))

    class FakeProcess:
        pid = 12345

        def wait(self, timeout=None):
            return 2

    monkeypatch.setattr(
        run_mosi.subprocess,
        "Popen",
        lambda *args, **kwargs: FakeProcess(),
    )
    failures = run_mosi.run_jobs(
        [job],
        feature_root=Path("/features"),
        repo_root=tmp_path,
        jobs_per_gpu=1,
        timeout_seconds=123,
    )

    assert failures == 1
    assert not (job.output_dir / "config.json").exists()
    assert not (job.output_dir / "history.json").exists()
    assert not (job.output_dir / "metrics.json").exists()
    assert not list(job.output_dir.glob("predictions_miss_*.npz"))
    assert json.loads((job.output_dir / "status.json").read_text())["state"] == "failed"


def test_timeout_and_launch_exception_terminate_processes_instead_of_hanging(
    monkeypatch,
    tmp_path,
):
    from gcnet_missing_m3_sdr_backbone import run_mosi

    jobs = run_mosi.build_jobs(output_root=tmp_path)[:2]
    terminated = []

    class TimeoutProcess:
        pid = 12345

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired("train", timeout)

    monkeypatch.setattr(
        run_mosi.subprocess,
        "Popen",
        lambda *args, **kwargs: TimeoutProcess(),
    )
    monkeypatch.setattr(
        run_mosi,
        "_terminate_process_tree",
        lambda process: terminated.append(process.pid),
    )
    failures = run_mosi.run_jobs(
        [jobs[0]],
        feature_root=Path("/features"),
        repo_root=tmp_path,
        jobs_per_gpu=1,
        timeout_seconds=1,
    )
    assert failures == 1
    assert terminated == [12345]
    assert json.loads((jobs[0].output_dir / "status.json").read_text())[
        "state"
    ] == "timeout"

    first = TimeoutProcess()
    calls = {"count": 0}

    def launch(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return first
        raise RuntimeError("launch failed")

    monkeypatch.setattr(run_mosi.subprocess, "Popen", launch)
    with pytest.raises(RuntimeError, match="launch failed"):
        run_mosi.run_jobs(
            run_mosi.build_jobs(output_root=tmp_path / "launch")[:2],
            feature_root=Path("/features"),
            repo_root=tmp_path,
            jobs_per_gpu=1,
            timeout_seconds=1,
        )
    assert terminated[-1] == 12345


def test_dry_run_prints_ten_treatment_commands_and_atomic_manifest(
    monkeypatch,
    capsys,
    tmp_path,
):
    from gcnet_missing_m3_sdr_backbone import run_mosi

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_mosi.py",
            "--dry-run",
            "--output-root",
            str(tmp_path),
            "--repo-root",
            str(Path(run_mosi.__file__).resolve().parents[1]),
            "--feature-root",
            "/features",
            "--source-commit",
            "a" * 40,
        ],
    )
    assert run_mosi.main() == 0
    output = capsys.readouterr().out
    commands = [line for line in output.splitlines() if line.startswith("COMMAND ")]
    assert len(commands) == 10
    assert sum("--sdr-variant sdr-public" in line for line in commands) == 5
    assert sum("--sdr-variant sdr-paper" in line for line in commands) == 5
    assert "gcnet_missing_m3.train_gcnet" not in output
    assert "original" not in output.lower()
    assert "control" not in output.lower()
    assert len(json.loads((tmp_path / "manifest.json").read_text())["jobs"]) == 10


def test_aggregate_reports_per_rate_high_missing_paired_deltas_and_collapse(
    tmp_path,
):
    from gcnet_missing_m3_sdr_backbone import run_mosi

    jobs = run_mosi.build_jobs(output_root=tmp_path)
    for job in jobs:
        if job.variant == "sdr-public":
            _write_complete_result(
                job,
                validation_offset=0.02,
                test_offset=-0.05,
            )
        else:
            _write_complete_result(
                job,
                validation_offset=-0.01,
                test_offset=0.10,
            )
    _make_rate_collapsed(jobs[0], "0.4")
    control = {
        seed: {rate: 0.70 - float(rate) / 100.0 for rate in RATES}
        for seed in (66, 67, 68, 69, 70)
    }
    summary = run_mosi.aggregate(jobs, control_validation=control)

    assert summary["selection_basis"] == "validation-eight-rate-mean-weighted-f1"
    assert summary["test_used_for_selection"] is False
    assert summary["validation_order"] == ["sdr-public", "sdr-paper"]
    public = summary["variants"]["sdr-public"]
    paper = summary["variants"]["sdr-paper"]
    assert set(public["seeds"]) == {"66", "67", "68", "69", "70"}
    assert set(public["rates"]) == set(RATES)
    assert public["positive_seed_count"] == 5
    assert public["positive_seeds"] == [66, 67, 68, 69, 70]
    assert paper["positive_seed_count"] == 0
    assert public["validation_high_missing_mean"] == pytest.approx(
        np.mean([0.72 - rate / 100.0 for rate in (0.4, 0.5, 0.6, 0.7)])
    )
    assert public["collapse"] == {"any": True, "seeds": [66]}
    assert public["seeds"]["66"]["collapse_rates"] == ["0.4"]
    assert public["seeds"]["66"]["paired_validation_delta"] == pytest.approx(0.02)
    assert public["seeds"]["66"]["validation_mean"] > paper["seeds"]["66"][
        "validation_mean"
    ]
    # Paper has deliberately better test metrics, which must not alter ordering.
    assert public["test_mean"] < paper["test_mean"]
