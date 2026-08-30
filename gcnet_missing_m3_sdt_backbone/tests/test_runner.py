from __future__ import annotations

import importlib
import importlib.util
import json
import subprocess
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pytest

from gcnet_missing_m3_sdt_backbone.train_gcnet import SDTTrainConfig


MODULE_NAME = "gcnet_missing_m3_sdt_backbone.run_mosi"
RATES = tuple(format(index / 10, ".1f") for index in range(8))


def _runner():
    spec = importlib.util.find_spec(MODULE_NAME)
    assert spec is not None, "the independent SDT runner is not implemented"
    return importlib.import_module(MODULE_NAME)


def _write_complete_result(job, *, validation_only=False, returncode=None):
    job.output_dir.mkdir(parents=True, exist_ok=True)
    config = asdict(
        SDTTrainConfig(
            seed=job.seed,
            evaluate_test=not validation_only,
        )
    )
    history = [{"epoch": epoch} for epoch in range(1, 101)]
    metrics = {
        "backbone": "sdt-style-full-context",
        "best_epoch": 37,
        "best_validation_mean_weighted_f1": 0.79,
        "evaluation_stage": (
            "train-validation-only"
            if validation_only
            else "train-validation-test"
        ),
        "test": (
            {}
            if validation_only
            else {
                rate: {
                    "weighted_f1": 0.8,
                    "prediction_std": 0.25,
                    "predicted_sign_count": 2,
                    "mask_sha256": format(index, "064x"),
                }
                for index, rate in enumerate(RATES, start=1)
            }
        ),
        "mask_sha256": (
            {}
            if validation_only
            else {
                rate: format(index, "064x")
                for index, rate in enumerate(RATES, start=1)
            }
        ),
        "registered_parameters": 8_847_037,
        "trainable_parameters": 7_986_877,
        "registered_backbone_parameters": 5_869_754,
        "active_backbone_parameters": 5_869_370,
        "control_active_backbone_parameters": 5_864_700,
    }
    for name, payload in (
        ("config.json", config),
        ("history.json", history),
        ("metrics.json", metrics),
    ):
        (job.output_dir / name).write_text(
            json.dumps(payload), encoding="utf-8"
        )
    (job.output_dir / "train.log").write_text("epoch=100\n", encoding="utf-8")
    if returncode is not None:
        (job.output_dir / "status.json").write_text(
            json.dumps({"returncode": returncode}), encoding="utf-8"
        )


def test_build_jobs_is_exactly_the_locked_five_seed_candidate_matrix(tmp_path):
    module = _runner()

    jobs = module.build_jobs(output_root=tmp_path, gpus=(0, 1, 2))

    assert [(job.seed, job.gpu) for job in jobs] == [
        (66, 0),
        (67, 1),
        (68, 2),
        (69, 0),
        (70, 1),
    ]
    assert [job.output_dir for job in jobs] == [
        tmp_path / "seed_66",
        tmp_path / "seed_67",
        tmp_path / "seed_68",
        tmp_path / "seed_69",
        tmp_path / "seed_70",
    ]
    assert Counter(job.gpu for job in jobs) == {0: 2, 1: 2, 2: 1}


@pytest.mark.parametrize("gpus", [(), (0, 1), (0, 1, 4), (2, 1, 0)])
def test_build_jobs_rejects_any_gpu_contract_drift(tmp_path, gpus):
    module = _runner()

    with pytest.raises(ValueError, match="exactly"):
        module.build_jobs(output_root=tmp_path, gpus=gpus)


def test_command_invokes_only_the_locked_sdt_candidate(tmp_path):
    module = _runner()
    job = module.build_jobs(output_root=tmp_path, gpus=(0, 1, 2))[0]

    command = module.build_command(
        job,
        python_executable=Path("/env/bin/python"),
        feature_root=Path("/features"),
    )
    joined = " ".join(command)

    assert command[:3] == [
        "/env/bin/python",
        "-m",
        "gcnet_missing_m3_sdt_backbone.train_gcnet",
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
    ):
        assert expected in joined
    assert "--seed 66" in joined
    assert str(job.output_dir) in command
    assert "gcnet_missing_m3.train_gcnet" not in command
    assert "original" not in joined.lower()
    assert "control" not in joined.lower()
    assert "--skip-test-evaluation" not in command


def test_complete_result_requires_all_four_valid_artifact_classes(tmp_path):
    module = _runner()
    job = module.build_jobs(output_root=tmp_path, gpus=(0, 1, 2))[0]

    assert module.inspect_result(job).complete is False
    _write_complete_result(job)
    inspection = module.inspect_result(job)
    assert inspection.complete is True
    assert inspection.reason == "complete-test-8-rates"

    required = ("config.json", "history.json", "metrics.json", "train.log")
    for name in required:
        path = job.output_dir / name
        saved = path.read_bytes()
        path.unlink()
        inspection = module.inspect_result(job)
        assert inspection.complete is False
        assert name in inspection.reason
        path.write_bytes(saved)


def test_completion_rejects_half_written_and_semantically_wrong_results(tmp_path):
    module = _runner()
    jobs = module.build_jobs(output_root=tmp_path, gpus=(0, 1, 2))

    malformed = jobs[0]
    _write_complete_result(malformed)
    (malformed.output_dir / "metrics.json").write_text("{", encoding="utf-8")
    assert module.inspect_result(malformed).complete is False

    short_history = jobs[1]
    _write_complete_result(short_history)
    history = json.loads(
        (short_history.output_dir / "history.json").read_text(encoding="utf-8")
    )
    (short_history.output_dir / "history.json").write_text(
        json.dumps(history[:-1]), encoding="utf-8"
    )
    assert "100 epochs" in module.inspect_result(short_history).reason

    wrong_seed = jobs[2]
    _write_complete_result(wrong_seed)
    config = json.loads(
        (wrong_seed.output_dir / "config.json").read_text(encoding="utf-8")
    )
    config["seed"] = 999
    (wrong_seed.output_dir / "config.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    assert "config" in module.inspect_result(wrong_seed).reason

    wrong_locked_field = jobs[4]
    _write_complete_result(wrong_locked_field)
    config = json.loads(
        (wrong_locked_field.output_dir / "config.json").read_text(
            encoding="utf-8"
        )
    )
    config["hidden"] = 200
    (wrong_locked_field.output_dir / "config.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    assert "config" in module.inspect_result(wrong_locked_field).reason

    seven_rates = jobs[3]
    _write_complete_result(seven_rates)
    metrics = json.loads(
        (seven_rates.output_dir / "metrics.json").read_text(encoding="utf-8")
    )
    metrics["test"].pop("0.7")
    (seven_rates.output_dir / "metrics.json").write_text(
        json.dumps(metrics), encoding="utf-8"
    )
    assert "8 test rates" in module.inspect_result(seven_rates).reason

    _write_complete_result(seven_rates)
    metrics = json.loads(
        (seven_rates.output_dir / "metrics.json").read_text(encoding="utf-8")
    )
    del metrics["test"]["0.4"]["prediction_std"]
    (seven_rates.output_dir / "metrics.json").write_text(
        json.dumps(metrics), encoding="utf-8"
    )
    assert "semantic" in module.inspect_result(seven_rates).reason


def test_validation_only_result_is_not_complete_for_the_formal_runner(tmp_path):
    module = _runner()
    job = module.build_jobs(output_root=tmp_path, gpus=(0, 1, 2))[0]
    _write_complete_result(job, validation_only=True)

    inspection = module.inspect_result(job)

    assert inspection.complete is False
    assert "config" in inspection.reason or "test" in inspection.reason
    assert inspection.has_test_metrics is False


def test_pending_jobs_inherits_only_content_complete_runs(tmp_path):
    module = _runner()
    jobs = module.build_jobs(output_root=tmp_path, gpus=(0, 1, 2))
    _write_complete_result(jobs[0])
    _write_complete_result(jobs[1])
    (jobs[1].output_dir / "history.json").write_text("[]", encoding="utf-8")

    assert module.pending_jobs(jobs) == jobs[1:]


def test_pending_rerun_cannot_combine_new_config_with_old_completion_files(
    monkeypatch, tmp_path
):
    module = _runner()
    job = module.build_jobs(output_root=tmp_path, gpus=(0, 1, 2))[0]
    _write_complete_result(job)
    stale_config = json.loads(
        (job.output_dir / "config.json").read_text(encoding="utf-8")
    )
    stale_config["hidden"] = 200
    (job.output_dir / "config.json").write_text(
        json.dumps(stale_config), encoding="utf-8"
    )
    assert module.inspect_result(job).complete is False

    class FakeProcess:
        pid = 12345

        def wait(self, timeout=None):
            fresh_config = asdict(SDTTrainConfig(seed=job.seed))
            (job.output_dir / "config.json").write_text(
                json.dumps(fresh_config), encoding="utf-8"
            )
            return 2

    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        lambda *args, **kwargs: FakeProcess(),
    )

    failures = module.run_jobs(
        [job],
        python_executable=Path("/env/bin/python"),
        feature_root=Path("/features"),
        repo_root=tmp_path,
        jobs_per_gpu=1,
        timeout_seconds=123,
    )

    assert failures == 1
    assert not (job.output_dir / "history.json").exists()
    assert not (job.output_dir / "metrics.json").exists()
    status = json.loads(
        (job.output_dir / "status.json").read_text(encoding="utf-8")
    )
    assert status["state"] == "failed"


def test_manifest_is_atomic_and_records_recomputed_completion(tmp_path):
    module = _runner()
    jobs = module.build_jobs(output_root=tmp_path / "formal", gpus=(0, 1, 2))
    _write_complete_result(jobs[0])
    manifest_path = tmp_path / "formal" / "manifest.json"
    source_commit = "a" * 40
    repo_root = Path(module.__file__).resolve().parents[1]

    module.write_manifest(
        manifest_path,
        jobs,
        source_commit=source_commit,
        feature_root=Path("/features"),
        python_executable=Path("/env/bin/python"),
        repo_root=repo_root,
    )
    first = json.loads(manifest_path.read_text(encoding="utf-8"))
    module.write_manifest(
        manifest_path,
        jobs,
        source_commit=source_commit,
        feature_root=Path("/features"),
        python_executable=Path("/env/bin/python"),
        repo_root=repo_root,
    )
    second = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert first == second
    assert first["treatment"] == "sdt-style-full-context"
    assert first["seeds"] == [66, 67, 68, 69, 70]
    assert first["gpu_mapping"] == {
        "66": 0,
        "67": 1,
        "68": 2,
        "69": 0,
        "70": 1,
    }
    assert first["source_commit"] == source_commit
    assert first["features"] == {
        "root": "/features",
        "audio": "/features/wav2vec-large-c-UTT",
        "text": "/features/deberta-large-4-UTT",
        "video": "/features/manet_UTT",
    }
    assert first["runtime"]["python_executable"] == "/env/bin/python"
    assert first["runtime"]["python_version"]
    assert first["runtime"]["torch_version"]
    assert "cuda_version" in first["runtime"]
    assert "gpu_names" in first["runtime"]
    assert set(first["source_files_sha256"]) == {
        "gcnet_missing_m3_sdt_backbone/model.py",
        "gcnet_missing_m3_sdt_backbone/train_gcnet.py",
        "gcnet_missing_m3_sdt_backbone/run_mosi.py",
    }
    assert all(
        len(value) == 64 for value in first["source_files_sha256"].values()
    )
    assert [entry["complete"] for entry in first["jobs"]] == [
        True,
        False,
        False,
        False,
        False,
    ]
    complete_entry = first["jobs"][0]
    assert len(complete_entry["config_sha256"]) == 64
    assert len(complete_entry["metrics_sha256"]) == 64
    assert set(complete_entry["mask_sha256"]) == set(RATES)
    assert complete_entry["parameter_counts"] == {
        "registered": 8_847_037,
        "trainable": 7_986_877,
        "backbone_registered": 5_869_754,
        "backbone_active": 5_869_370,
        "control_backbone_active": 5_864_700,
    }
    assert not list(manifest_path.parent.glob("*.tmp"))
    encoded = manifest_path.read_text(encoding="utf-8").lower()
    assert "original" not in encoded
    assert all(entry["seed"] in range(66, 71) for entry in first["jobs"])


def test_waves_enforce_per_gpu_concurrency_without_changing_mapping(tmp_path):
    module = _runner()
    jobs = module.build_jobs(output_root=tmp_path, gpus=(0, 1, 2))

    one_per_gpu = module.build_waves(jobs, jobs_per_gpu=1)
    two_per_gpu = module.build_waves(jobs, jobs_per_gpu=2)

    assert len(one_per_gpu) == 2
    assert len(two_per_gpu) == 1
    assert {job for wave in one_per_gpu for job in wave} == set(jobs)
    for wave in one_per_gpu:
        assert max(Counter(job.gpu for job in wave).values()) <= 1
    assert Counter(job.gpu for job in two_per_gpu[0]) == {0: 2, 1: 2, 2: 1}

    with pytest.raises(ValueError, match="positive"):
        module.build_waves(jobs, jobs_per_gpu=0)


def test_nonzero_child_exit_is_inherited_when_outputs_are_complete_and_pruned(
    monkeypatch, tmp_path
):
    module = _runner()
    job = module.build_jobs(output_root=tmp_path, gpus=(0, 1, 2))[0]

    class FakeProcess:
        pid = 12345

        def wait(self, timeout=None):
            assert 0 < timeout <= 123
            _write_complete_result(job)
            (job.output_dir / "best.pt").write_bytes(b"checkpoint")
            (job.output_dir / "predictions_miss_0p0.npz").write_bytes(b"npz")
            return 9

    popen_calls = []

    def popen(command, **kwargs):
        popen_calls.append((command, kwargs))
        return FakeProcess()

    monkeypatch.setattr(module.subprocess, "Popen", popen)

    failures = module.run_jobs(
        [job],
        python_executable=Path("/env/bin/python"),
        feature_root=Path("/features"),
        repo_root=tmp_path,
        jobs_per_gpu=1,
        timeout_seconds=123,
    )

    assert failures == 0
    assert len(popen_calls) == 1
    _, kwargs = popen_calls[0]
    assert kwargs["cwd"] == tmp_path
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert kwargs["env"]["PYTHONHASHSEED"] == "0"
    assert kwargs["start_new_session"] is True
    assert not (job.output_dir / "best.pt").exists()
    assert not list(job.output_dir.glob("*.npz"))
    status = json.loads(
        (job.output_dir / "status.json").read_text(encoding="utf-8")
    )
    assert status["state"] == "complete"
    assert status["returncode"] == 9
    assert status["completion_reason"] == "complete-test-8-rates"
    assert status["log_path"] == str(job.output_dir / "train.log")


def test_incomplete_nonzero_child_exit_is_a_failure(monkeypatch, tmp_path):
    module = _runner()
    job = module.build_jobs(output_root=tmp_path, gpus=(0, 1, 2))[0]

    class FakeProcess:
        pid = 12345

        def wait(self, timeout=None):
            return 2

    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        lambda *args, **kwargs: FakeProcess(),
    )

    failures = module.run_jobs(
        [job],
        python_executable=Path("/env/bin/python"),
        feature_root=Path("/features"),
        repo_root=tmp_path,
        jobs_per_gpu=1,
        timeout_seconds=123,
    )

    assert failures == 1
    status = json.loads(
        (job.output_dir / "status.json").read_text(encoding="utf-8")
    )
    assert status["state"] == "failed"
    assert status["returncode"] == 2
    assert status["log_path"] == str(job.output_dir / "train.log")


def test_dry_run_prints_five_candidate_commands_and_no_control(
    monkeypatch, capsys, tmp_path
):
    module = _runner()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_mosi.py",
            "--dry-run",
            "--output-root",
            str(tmp_path),
            "--repo-root",
            str(Path(module.__file__).resolve().parents[1]),
            "--python-executable",
            "/env/bin/python",
            "--feature-root",
            "/features",
            "--source-commit",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        ],
    )

    assert module.main() == 0
    output = capsys.readouterr().out

    commands = [line for line in output.splitlines() if line.startswith("COMMAND ")]
    assert len(commands) == 5
    assert all("gcnet_missing_m3_sdt_backbone.train_gcnet" in line for line in commands)
    assert "gcnet_missing_m3.train_gcnet" not in output
    assert "original" not in output.lower()
    assert "control" not in output.lower()
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["jobs"]) == 5


def test_timeout_is_reported_as_failure_instead_of_waiting_forever(
    monkeypatch, tmp_path
):
    module = _runner()
    job = module.build_jobs(output_root=tmp_path, gpus=(0, 1, 2))[0]

    class FakeProcess:
        pid = 98765

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired(cmd="candidate", timeout=timeout)

    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        lambda *args, **kwargs: FakeProcess(),
    )
    monkeypatch.setattr(module, "_terminate_process_tree", lambda process: None)

    failures = module.run_jobs(
        [job],
        python_executable=Path("/env/bin/python"),
        feature_root=Path("/features"),
        repo_root=tmp_path,
        jobs_per_gpu=1,
        timeout_seconds=1,
    )

    assert failures == 1
    status = json.loads(
        (job.output_dir / "status.json").read_text(encoding="utf-8")
    )
    assert status["state"] == "timeout"
    assert status["returncode"] is None
    assert status["timeout_seconds"] == 1


def test_each_process_timeout_is_measured_from_its_own_start(
    monkeypatch, tmp_path
):
    module = _runner()
    jobs = module.build_jobs(output_root=tmp_path, gpus=(0, 1, 2))[:2]
    clock = {"now": 0.0}
    observed_timeouts = []

    class FakeProcess:
        def __init__(self, pid, elapsed):
            self.pid = pid
            self.elapsed = elapsed

        def wait(self, timeout=None):
            observed_timeouts.append(timeout)
            clock["now"] += self.elapsed
            return 2

    processes = iter((FakeProcess(101, 80.0), FakeProcess(102, 0.0)))
    monkeypatch.setattr(module.time, "time", lambda: clock["now"])
    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        lambda *args, **kwargs: next(processes),
    )

    failures = module.run_jobs(
        jobs,
        python_executable=Path("/env/bin/python"),
        feature_root=Path("/features"),
        repo_root=tmp_path,
        jobs_per_gpu=1,
        timeout_seconds=100,
    )

    assert failures == 2
    assert observed_timeouts[0] == pytest.approx(100.0)
    assert observed_timeouts[1] == pytest.approx(20.0)


def test_wave_launch_failure_terminates_started_children_and_closes_logs(
    monkeypatch, tmp_path
):
    module = _runner()
    jobs = module.build_jobs(output_root=tmp_path, gpus=(0, 1, 2))[:2]
    log_handles = []
    calls = {"count": 0}

    class RunningProcess:
        pid = 101

    def popen(*args, **kwargs):
        calls["count"] += 1
        log_handles.append(kwargs["stdout"])
        if calls["count"] == 1:
            return RunningProcess()
        raise OSError("synthetic launch failure")

    terminated = []
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    monkeypatch.setattr(
        module,
        "_terminate_process_tree",
        lambda process: terminated.append(process.pid),
    )

    with pytest.raises(OSError, match="synthetic launch failure"):
        module.run_jobs(
            jobs,
            python_executable=Path("/env/bin/python"),
            feature_root=Path("/features"),
            repo_root=tmp_path,
            jobs_per_gpu=1,
            timeout_seconds=100,
        )

    assert terminated == [101]
    assert log_handles
    assert all(handle.closed for handle in log_handles)
