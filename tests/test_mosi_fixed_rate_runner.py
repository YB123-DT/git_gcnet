from __future__ import annotations

import importlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np


def _runner():
    name = "scripts.run_mosi_fixed_rate"
    assert importlib.util.find_spec(name) is not None, "runner is not implemented"
    return importlib.import_module(name)


def _write_complete_result(job) -> None:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    rate_key = format(job.rate, ".1f")
    rate_token = rate_key.replace(".", "p")
    config = {
        "dataset": "CMUMOSI",
        "fold": 1,
        "seed": job.seed,
        "epochs": 100,
        "batch_size": 32,
        "train_rate_mode": "fixed",
        "fixed_missing_rate": job.rate,
        "hidden": 200,
        "latent_dim": 256,
        "window_past": 2,
        "window_future": 2,
        "time_attention": False,
        "fusion_type": "slot",
        "representation_type": "slot",
        "mmoe_variant": "dual-gate",
        "mosi_task_mode": "regression",
        "graph_branch_mode": "both",
        "learning_rate": 0.0005,
        "weight_decay": 0.00001,
        "jepa_weight": 0.1,
        "jepa_regression_aggregation": "target",
        "recurrent_padding_mode": "legacy",
        "task_regression_loss": "mse",
        "postgraph_sequence_mode": "independent",
        "jepa_rate_weighting": "uniform",
        "graph_message_calibration": "none",
        "evaluation_protocol": "official",
        "evaluate_test": True,
    }
    history = [{"epoch": epoch} for epoch in range(1, 101)]
    digest = "a" * 64
    metrics = {
        "best_epoch": 42,
        "evaluation_stage": "train-validation-test",
        "train_missing_rate": job.rate,
        "selection_missing_rates": [job.rate],
        "test": {
            rate_key: {
                "weighted_f1": 0.8,
                "mask_sha256": digest,
            }
        },
        "mask_sha256": {rate_key: digest},
    }
    (job.output_dir / "config.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    (job.output_dir / "history.json").write_text(
        json.dumps(history), encoding="utf-8"
    )
    (job.output_dir / "metrics.json").write_text(
        json.dumps(metrics), encoding="utf-8"
    )
    (job.output_dir / "train.log").write_text("complete\n", encoding="utf-8")
    np.savez_compressed(
        job.output_dir / ("predictions_miss_" + rate_token + ".npz"),
        predictions=np.array([0.2, -0.1], dtype=np.float32),
        labels=np.array([0.3, -0.2], dtype=np.float32),
        availability=np.ones((2, 3), dtype=np.float32),
    )


def test_default_matrix_has_forty_unique_fixed_rate_jobs(tmp_path):
    module = _runner()
    jobs = module.build_jobs(output_root=tmp_path)

    assert len(jobs) == 40
    assert {(job.rate, job.seed) for job in jobs} == {
        (rate, seed)
        for rate in module.MISSING_RATES
        for seed in module.SEEDS
    }
    assert {job.gpu for job in jobs} <= {0, 1, 2}
    assert 4 not in {job.gpu for job in jobs}
    sample = next(job for job in jobs if job.rate == 0.5 and job.seed == 66)
    assert sample.output_dir == tmp_path / "rate_0p5" / "seed_66"


def test_command_locks_slot_missing_m3_and_same_rate(tmp_path):
    module = _runner()
    job = next(
        job
        for job in module.build_jobs(output_root=tmp_path)
        if job.rate == 0.5 and job.seed == 66
    )
    command = " ".join(
        module.build_command(
            job,
            python_executable=Path("/env/bin/python"),
            feature_root=Path("/features"),
        )
    )

    assert "-m gcnet_missing_m3.train_gcnet" in command
    assert "--train-rate-mode fixed" in command
    assert "--train-missing-rate 0.5" in command
    assert "--fusion-type slot" in command
    assert "--representation-type slot" in command
    assert "--mmoe-variant dual-gate" in command
    assert "--mosi-task-mode regression" in command
    assert "--graph-branch-mode both" in command
    assert "--hidden 200" in command
    assert "--latent-dim 256" in command
    assert "--windowp 2" in command
    assert "--windowf 2" in command
    assert "--lr 0.0005" in command
    assert "--jepa-weight 0.1" in command
    assert "original" not in command.lower()


def test_result_audit_requires_only_the_registered_rate(tmp_path):
    module = _runner()
    job = next(
        job
        for job in module.build_jobs(output_root=tmp_path)
        if job.rate == 0.5 and job.seed == 66
    )
    _write_complete_result(job)

    inspection = module.inspect_result(job)
    assert inspection.complete, inspection.reason

    metrics_path = job.output_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["test"]["0.7"] = dict(metrics["test"]["0.5"])
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    inspection = module.inspect_result(job)
    assert not inspection.complete
    assert "registered rate" in inspection.reason


def test_result_audit_rejects_partial_history_and_extra_prediction_file(tmp_path):
    module = _runner()
    job = next(
        job
        for job in module.build_jobs(output_root=tmp_path)
        if job.rate == 0.7 and job.seed == 70
    )
    _write_complete_result(job)

    history_path = job.output_dir / "history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    history_path.write_text(json.dumps(history[:-1]), encoding="utf-8")
    assert not module.inspect_result(job).complete

    history_path.write_text(json.dumps(history), encoding="utf-8")
    np.savez_compressed(
        job.output_dir / "predictions_miss_0p6.npz",
        predictions=np.array([0.1]),
        labels=np.array([0.1]),
        availability=np.ones((1, 3)),
    )
    inspection = module.inspect_result(job)
    assert not inspection.complete
    assert "prediction artifact" in inspection.reason


def test_waves_limit_each_gpu_to_five_concurrent_jobs(tmp_path):
    module = _runner()
    jobs = module.build_jobs(output_root=tmp_path)
    waves = module.build_waves(jobs, jobs_per_gpu=5)

    assert sum(len(wave) for wave in waves) == 40
    assert len({(job.rate, job.seed) for wave in waves for job in wave}) == 40
    for wave in waves:
        per_gpu = Counter(job.gpu for job in wave)
        assert max(per_gpu.values()) <= 5
        assert set(per_gpu) <= {0, 1, 2}
