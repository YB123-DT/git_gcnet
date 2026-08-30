from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest


RATES = tuple(format(index / 10.0, ".1f") for index in range(8))
COUNTS = {
    "registered_parameters": 101,
    "trainable_parameters": 89,
    "registered_backbone_parameters": 61,
    "trainable_backbone_parameters": 61,
}


def _availability(seed, rate_index):
    values = np.zeros((7, 3), dtype=np.float32)
    code = (seed - 65) * 17 + rate_index
    for index in range(values.size):
        values.reshape(-1)[index] = float((code >> (index % 12)) & 1)
    return values


def _fake_provenance(run_mosi, job, source_commit="a" * 40):
    from gcnet_missing_m3_raw_sdr.train_gcnet import RawSDRTrainConfig

    return {
        "schema_version": 1,
        "treatment": run_mosi.TREATMENT,
        "training_module": run_mosi.TRAINING_MODULE,
        "variant": run_mosi.VARIANT,
        "seed": job.seed,
        "source_commit": source_commit,
        "source_files_sha256": {
            name: hashlib.sha256(name.encode()).hexdigest()
            for name in run_mosi.SOURCE_FILES
        },
        "features": {
            "root": "/features",
            "audio": "/features/wav2vec-large-c-UTT",
            "text": "/features/deberta-large-4-UTT",
            "video": "/features/manet_UTT",
        },
        "training_runtime": {"python_version": "3.8.10"},
        "training_python_executable": str(run_mosi.DEFAULT_PYTHON),
        "canonical_config_sha256": run_mosi._canonical_json_sha256(
            asdict(RawSDRTrainConfig(seed=job.seed))
        ),
        "command": run_mosi.build_command(job, feature_root=Path("/features")),
    }


def _write_complete_result(
    run_mosi,
    job,
    *,
    validation_offset=0.0,
    test_offset=0.0,
    parameter_counts=COUNTS,
    returncode_provenance=True,
):
    from gcnet_missing_m3 import train_gcnet as base_train
    from gcnet_missing_m3_raw_sdr.train_gcnet import RawSDRTrainConfig

    job.output_dir.mkdir(parents=True, exist_ok=True)
    config = asdict(RawSDRTrainConfig(seed=job.seed))
    history = []
    selected = None
    for epoch in range(1, 101):
        bonus = 0.03 if epoch == 37 else 0.0
        validation = {}
        for index, rate in enumerate(RATES):
            schedule_hash = hashlib.sha256(
                "schedule:{}:{}".format(job.seed, rate).encode()
            ).hexdigest()
            validation[rate] = {
                "weighted_f1": 0.70 + validation_offset + bonus - index / 1000,
                "loss": 0.2,
                "prediction_std": 0.4,
                "predicted_sign_count": 2,
                "mask_sha256": schedule_hash,
            }
        validation_mean = np.mean(
            [validation[rate]["weighted_f1"] for rate in RATES]
        ).item()
        history.append(
            {
                "epoch": epoch,
                "train": {"weighted_f1": 0.5},
                "validation": validation,
                "validation_mean_weighted_f1": validation_mean,
            }
        )
        if epoch == 37:
            selected = (validation, validation_mean)
    best_validation, best_validation_mean = selected

    test = {}
    schedule_hashes = {}
    artifact_hashes = {}
    for index, rate in enumerate(RATES):
        labels = np.asarray([0, -1, -1, -1, 1, 1, 1], dtype=np.float32)
        if test_offset < 0:
            predictions = np.asarray(
                [99 + index, -1, -1, 1, -1, 1, 1], dtype=np.float32
            )
        else:
            predictions = np.asarray(
                [99 + index, -1, -1, -1, 1, 1, 1], dtype=np.float32
            )
        availability = _availability(job.seed, index)
        archive_hash = hashlib.sha256(
            np.ascontiguousarray(availability).tobytes()
        ).hexdigest()
        schedule_hash = best_validation[rate]["mask_sha256"]
        recomputed = base_train._metrics(
            "CMUMOSI", labels, predictions, "regression"
        )
        test[rate] = {
            "weighted_f1": recomputed["weighted_f1"],
            "loss": 0.3,
            "prediction_std": recomputed["prediction_std"],
            "predicted_sign_count": recomputed["predicted_sign_count"],
            "mask_sha256": schedule_hash,
            "prediction_availability_sha256": archive_hash,
        }
        schedule_hashes[rate] = schedule_hash
        artifact_hashes[rate] = archive_hash
        np.savez_compressed(
            job.output_dir / "predictions_miss_{}.npz".format(rate.replace(".", "p")),
            predictions=predictions,
            labels=labels,
            availability=availability,
        )

    metrics = {
        "variant": run_mosi.VARIANT,
        "sdr_variant": "sdr-public",
        "sdr_input_type": "raw-residual",
        "backbone": run_mosi.VARIANT,
        "best_epoch": 37,
        "best_validation": best_validation,
        "best_validation_mean_weighted_f1": best_validation_mean,
        "selection_missing_rates": [index / 10 for index in range(8)],
        "evaluation_stage": "train-validation-test",
        "test": test,
        "mask_sha256": schedule_hashes,
        "prediction_availability_sha256": artifact_hashes,
        "ema_steps": 100,
        "wall_time_seconds": 123.0,
        "peak_memory_bytes": 1000,
        **parameter_counts,
    }
    for name, value in (
        ("config.json", config),
        ("history.json", history),
        ("metrics.json", metrics),
    ):
        (job.output_dir / name).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    (job.output_dir / "train.log").touch()
    if returncode_provenance:
        (job.output_dir / run_mosi.PRODUCER_PROVENANCE_NAME).write_text(
            json.dumps(_fake_provenance(run_mosi, job), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )


def _reference_from_jobs(run_mosi, jobs, *, offset):
    reference = {}
    for job in jobs:
        metrics = json.loads((job.output_dir / "metrics.json").read_text())
        copied = json.loads(json.dumps(metrics))
        for rate in RATES:
            copied["best_validation"][rate]["weighted_f1"] += offset
        copied["best_validation_mean_weighted_f1"] += offset
        reference[job.seed] = copied
    return reference


def test_builds_exactly_five_registered_treatments_on_one_healthy_gpu_each(tmp_path):
    from gcnet_missing_m3_raw_sdr import run_mosi

    jobs = run_mosi.build_jobs(output_root=tmp_path)

    assert len(jobs) == 5
    assert [job.seed for job in jobs] == [66, 67, 68, 69, 70]
    assert [job.gpu for job in jobs] == [2, 3, 5, 6, 7]
    assert all(job.variant == "raw-residual-sdr-public" for job in jobs)
    assert len({job.gpu for job in jobs}) == 5
    assert [job.output_dir for job in jobs] == [
        tmp_path / "seed_{}".format(seed) for seed in range(66, 71)
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gpus": (2, 3, 4, 6, 7)},
        {"gpus": (2, 2, 5, 6, 7)},
        {"gpus": (2, 3, 5, 6)},
        {"seeds": (66, 67, 68, 69, 71)},
        {"seeds": (66, 66, 68, 69, 70)},
        {"variants": ("raw-residual-sdr-public", "sdr-paper")},
        {"variants": ("slot",)},
    ],
)
def test_rejects_gpu4_duplicates_unknown_seed_or_any_extra_variant(tmp_path, kwargs):
    from gcnet_missing_m3_raw_sdr import run_mosi

    with pytest.raises(ValueError):
        run_mosi.build_jobs(output_root=tmp_path, **kwargs)


def test_rejects_manual_job_relabeling_and_command_is_raw_only(tmp_path):
    from gcnet_missing_m3_raw_sdr import run_mosi

    job = run_mosi.build_jobs(output_root=tmp_path)[0]
    command = run_mosi.build_command(job, feature_root=Path("/features"))
    joined = " ".join(command)
    assert command[:3] == [
        "/data2/yb/reproduction_envs/gcnet-official/bin/python",
        "-m",
        "gcnet_missing_m3_raw_sdr.train_gcnet",
    ]
    for expected in (
        "--train-rate-mode all",
        "--lr 0.0005",
        "--epochs 100",
        "--seed 66",
        "--device cuda",
    ):
        assert expected in joined
    for forbidden in ("original", "control", "slot", "sdr-paper"):
        assert forbidden not in joined.lower()
    with pytest.raises(ValueError):
        run_mosi.build_command(replace(job, variant="sdr-paper"))
    with pytest.raises(ValueError):
        run_mosi.build_command(replace(job, seed=99))
    with pytest.raises(ValueError):
        run_mosi.build_command(replace(job, gpu=4))


def test_parameter_count_lock_is_fail_closed_until_smoke_records_exact_counts(
    tmp_path, monkeypatch
):
    from gcnet_missing_m3_raw_sdr import run_mosi

    job = run_mosi.build_jobs(output_root=tmp_path)[0]
    _write_complete_result(run_mosi, job)
    monkeypatch.setattr(run_mosi, "EXPECTED_PARAMETER_COUNTS", None)
    inspection = run_mosi.inspect_result(job)
    assert inspection.complete is False
    assert "parameter count" in inspection.reason.lower()
    assert run_mosi.inspect_result(job, expected_parameter_counts=COUNTS).complete


def test_completion_requires_provenance_config_100_epochs_metrics_and_8_npz(
    tmp_path,
):
    from gcnet_missing_m3_raw_sdr import run_mosi

    job = run_mosi.build_jobs(output_root=tmp_path)[0]
    _write_complete_result(run_mosi, job)
    assert run_mosi.inspect_result(job, expected_parameter_counts=COUNTS).complete

    required = [
        run_mosi.PRODUCER_PROVENANCE_NAME,
        "config.json",
        "history.json",
        "metrics.json",
        "train.log",
        "predictions_miss_0p7.npz",
    ]
    for name in required:
        path = job.output_dir / name
        saved = path.read_bytes()
        path.unlink()
        assert not run_mosi.inspect_result(
            job, expected_parameter_counts=COUNTS
        ).complete
        path.write_bytes(saved)

    archive = job.output_dir / "predictions_miss_0p7.npz"
    archive.write_bytes(b"half-written")
    inspection = run_mosi.inspect_result(job, expected_parameter_counts=COUNTS)
    assert not inspection.complete
    assert "archive" in inspection.reason.lower()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c, h, m: c.update(seed=999),
        lambda c, h, m: h.pop(),
        lambda c, h, m: m.update(variant="slot"),
        lambda c, h, m: m.update(sdr_input_type="slot"),
        lambda c, h, m: m["test"].pop("0.7"),
        lambda c, h, m: m["mask_sha256"].update({"0.3": "f" * 64}),
        lambda c, h, m: m["prediction_availability_sha256"].update(
            {"0.3": "f" * 64}
        ),
        lambda c, h, m: m.update(best_epoch=38),
        lambda c, h, m: m.update(registered_parameters=999),
        lambda c, h, m: h[36]["validation"]["0.2"].update(loss=float("nan")),
        lambda c, h, m: m["test"]["0.2"].update(loss=float("nan")),
    ],
)
def test_completion_rejects_semantic_mismatch_or_old_directory_relabel(
    tmp_path, mutation
):
    from gcnet_missing_m3_raw_sdr import run_mosi

    job = run_mosi.build_jobs(output_root=tmp_path)[0]
    _write_complete_result(run_mosi, job)
    config = json.loads((job.output_dir / "config.json").read_text())
    history = json.loads((job.output_dir / "history.json").read_text())
    metrics = json.loads((job.output_dir / "metrics.json").read_text())
    mutation(config, history, metrics)
    (job.output_dir / "config.json").write_text(json.dumps(config))
    (job.output_dir / "history.json").write_text(json.dumps(history))
    (job.output_dir / "metrics.json").write_text(json.dumps(metrics))
    assert not run_mosi.inspect_result(
        job, expected_parameter_counts=COUNTS
    ).complete


def test_nonzero_child_exit_is_inherited_when_all_durable_outputs_are_complete(
    tmp_path, monkeypatch
):
    from gcnet_missing_m3_raw_sdr import run_mosi

    job = run_mosi.build_jobs(output_root=tmp_path)[0]
    expected = _fake_provenance(run_mosi, job)
    monkeypatch.setattr(
        run_mosi,
        "_producer_provenance",
        lambda *args, **kwargs: expected,
    )

    class Process:
        pid = 12345

        def wait(self, timeout=None):
            assert (job.output_dir / run_mosi.PRODUCER_PROVENANCE_NAME).is_file()
            _write_complete_result(
                run_mosi, job, returncode_provenance=False
            )
            return 17

    monkeypatch.setattr(run_mosi.subprocess, "Popen", lambda *a, **k: Process())
    failures = run_mosi.run_jobs(
        [job],
        feature_root=Path("/features"),
        repo_root=Path(run_mosi.__file__).resolve().parents[1],
        timeout_seconds=30,
        source_commit="a" * 40,
        expected_parameter_counts=COUNTS,
    )
    assert failures == 0
    status = json.loads((job.output_dir / "status.json").read_text())
    assert status["state"] == "complete"
    assert status["returncode"] == 17


def test_provenance_is_written_before_launch_and_existing_output_cannot_relabel(
    tmp_path, monkeypatch
):
    from gcnet_missing_m3_raw_sdr import run_mosi

    job = run_mosi.build_jobs(output_root=tmp_path)[0]
    expected = _fake_provenance(run_mosi, job)
    monkeypatch.setattr(
        run_mosi,
        "_producer_provenance",
        lambda *args, **kwargs: expected,
    )
    launched = []

    class Process:
        pid = 12345

        def wait(self, timeout=None):
            assert json.loads(
                (job.output_dir / run_mosi.PRODUCER_PROVENANCE_NAME).read_text()
            ) == expected
            return 2

    def popen(*args, **kwargs):
        launched.append(args)
        return Process()

    monkeypatch.setattr(run_mosi.subprocess, "Popen", popen)
    assert run_mosi.run_jobs(
        [job],
        feature_root=Path("/features"),
        repo_root=Path(run_mosi.__file__).resolve().parents[1],
        timeout_seconds=30,
        source_commit="a" * 40,
        expected_parameter_counts=COUNTS,
    ) == 1
    assert len(launched) == 1

    monkeypatch.setattr(
        run_mosi,
        "_producer_provenance",
        lambda *args, **kwargs: {**expected, "source_commit": "b" * 40},
    )
    with pytest.raises(ValueError, match="provenance"):
        run_mosi.run_jobs(
            [job],
            feature_root=Path("/features"),
            repo_root=Path(run_mosi.__file__).resolve().parents[1],
            timeout_seconds=30,
            source_commit="b" * 40,
            expected_parameter_counts=COUNTS,
        )
    assert len(launched) == 1


def test_timeout_terminates_bounded_process_tree(tmp_path, monkeypatch):
    from gcnet_missing_m3_raw_sdr import run_mosi

    job = run_mosi.build_jobs(output_root=tmp_path)[0]
    expected = _fake_provenance(run_mosi, job)
    monkeypatch.setattr(
        run_mosi, "_producer_provenance", lambda *args, **kwargs: expected
    )
    terminated = []

    class Process:
        pid = 12345

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired("raw-sdr", timeout)

    monkeypatch.setattr(run_mosi.subprocess, "Popen", lambda *a, **k: Process())
    monkeypatch.setattr(
        run_mosi,
        "_terminate_process_tree",
        lambda process: terminated.append(process.pid),
    )
    failures = run_mosi.run_jobs(
        [job],
        feature_root=Path("/features"),
        repo_root=Path(run_mosi.__file__).resolve().parents[1],
        timeout_seconds=1,
        source_commit="a" * 40,
        expected_parameter_counts=COUNTS,
    )
    assert failures == 1
    assert terminated == [12345]
    assert json.loads((job.output_dir / "status.json").read_text())["state"] == "timeout"


def test_aggregate_inherits_slot_and_control_and_gate_never_reads_test(tmp_path):
    from gcnet_missing_m3_raw_sdr import run_mosi

    jobs = run_mosi.build_jobs(output_root=tmp_path)
    for job in jobs:
        _write_complete_result(run_mosi, job, validation_offset=0.03)
    slot = _reference_from_jobs(run_mosi, jobs, offset=-0.02)
    control = _reference_from_jobs(run_mosi, jobs, offset=-0.01)

    first = run_mosi.aggregate(
        jobs,
        slot_reference=slot,
        control_reference=control,
        expected_parameter_counts=COUNTS,
    )
    assert first["selection_basis"] == "validation-only"
    assert first["test_used_for_selection"] is False
    assert first["primary_gate"]["status"] == "pass"
    assert first["formal_gate"]["status"] == "pass"
    assert first["primary_gate"]["positive_seed_count"] == 5
    assert first["primary_gate"]["mean_validation_delta"] == pytest.approx(0.02)
    assert first["primary_gate"]["mean_high_missing_delta"] == pytest.approx(0.02)
    assert first["control_comparison"]["mean_validation_delta"] == pytest.approx(0.01)

    for job in jobs:
        metrics_path = job.output_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text())
        metrics["test"] = {
            rate: {**value, "weighted_f1": 1.0 - value["weighted_f1"]}
            for rate, value in metrics["test"].items()
        }
        metrics_path.write_text(json.dumps(metrics))
    second = run_mosi.aggregate(
        jobs,
        slot_reference=slot,
        control_reference=control,
        expected_parameter_counts=COUNTS,
        require_complete_results=False,
    )
    assert second["primary_gate"] == first["primary_gate"]
    assert second["formal_gate"] == first["formal_gate"]


def test_aggregate_is_not_assessable_for_unpaired_masks_or_incomplete_matrix(
    tmp_path,
):
    from gcnet_missing_m3_raw_sdr import run_mosi

    jobs = run_mosi.build_jobs(output_root=tmp_path)
    for job in jobs:
        _write_complete_result(run_mosi, job, validation_offset=0.03)
    slot = _reference_from_jobs(run_mosi, jobs, offset=-0.02)
    control = _reference_from_jobs(run_mosi, jobs, offset=-0.01)
    slot[66]["best_validation"]["0.7"]["mask_sha256"] = "f" * 64
    summary = run_mosi.aggregate(
        jobs,
        slot_reference=slot,
        control_reference=control,
        expected_parameter_counts=COUNTS,
    )
    assert summary["primary_gate"]["status"] == "not-assessable"
    with pytest.raises(ValueError, match="five"):
        run_mosi.aggregate(
            jobs[:-1],
            slot_reference=slot,
            control_reference=control,
            expected_parameter_counts=COUNTS,
        )


def test_reference_loader_reads_metrics_only_and_never_builds_control_commands(
    tmp_path,
):
    from gcnet_missing_m3_raw_sdr import run_mosi

    paths = {}
    for seed in run_mosi.SEEDS:
        root = tmp_path / "seed_{}".format(seed)
        root.mkdir()
        (root / "metrics.json").write_text(json.dumps({"seed": seed}))
        paths[seed] = root
    loaded = run_mosi.load_inherited_reference(paths)
    assert loaded == {seed: {"seed": seed} for seed in run_mosi.SEEDS}
    assert all("command" not in value for value in loaded.values())


def test_dry_run_prints_five_and_only_five_raw_commands(tmp_path, monkeypatch, capsys):
    from gcnet_missing_m3_raw_sdr import run_mosi

    monkeypatch.setattr(
        run_mosi,
        "_producer_provenance",
        lambda job, **kwargs: _fake_provenance(run_mosi, job),
    )
    monkeypatch.setattr(
        run_mosi,
        "EXPECTED_PARAMETER_COUNTS",
        COUNTS,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_mosi.py",
            "--dry-run",
            "--output-root",
            str(tmp_path),
            "--feature-root",
            "/features",
            "--repo-root",
            str(Path(run_mosi.__file__).resolve().parents[1]),
            "--source-commit",
            "a" * 40,
        ],
    )
    assert run_mosi.main() == 0
    output = capsys.readouterr().out
    commands = [line for line in output.splitlines() if line.startswith("COMMAND ")]
    assert len(commands) == 5
    assert all("gcnet_missing_m3_raw_sdr.train_gcnet" in line for line in commands)
    for forbidden in ("original", "control", "slot", "sdr-paper"):
        assert forbidden not in output.lower()
