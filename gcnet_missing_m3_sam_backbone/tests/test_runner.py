import json

import pytest

from gcnet_missing_m3_sam_backbone.run_mosi import (
    SEEDS,
    build_jobs,
    inspect_result,
    summarize,
)


def test_jobs_are_exactly_five_seeds_without_control_reruns(tmp_path):
    jobs = build_jobs(tmp_path, gpus=(0, 1, 2))

    assert [job.seed for job in jobs] == list(SEEDS)
    assert [job.gpu for job in jobs] == [0, 1, 2, 0, 1]
    assert all("Original" not in " ".join(job.command) for job in jobs)
    assert all("gcnet_missing_m3_sam_backbone.train_mosi" in job.command for job in jobs)


def test_gate_requires_mean_and_three_paired_wins():
    candidate = {66: 0.88, 67: 0.87, 68: 0.85, 69: 0.88, 70: 0.86}
    control = {66: 0.86, 67: 0.86, 68: 0.86, 69: 0.86, 70: 0.86}

    result = summarize(candidate, control, collapsed_seeds=())

    assert result["positive_seed_count"] == 3
    assert result["mean_delta"] > 0
    assert result["passed"]


def test_gate_fails_when_mean_is_positive_but_only_two_seeds_win():
    candidate = {66: 0.95, 67: 0.95, 68: 0.85, 69: 0.85, 70: 0.85}
    control = {seed: 0.86 for seed in SEEDS}

    result = summarize(candidate, control, collapsed_seeds=())

    assert result["mean_delta"] > 0
    assert result["positive_seed_count"] == 2
    assert not result["passed"]


def test_gate_fails_if_any_formal_seed_collapses():
    candidate = {seed: 0.90 for seed in SEEDS}
    control = {seed: 0.86 for seed in SEEDS}

    result = summarize(candidate, control, collapsed_seeds=(68,))

    assert not result["passed"]
    assert result["collapsed_seeds"] == [68]


def test_inspect_result_rejects_test_oracle_and_half_written_run(tmp_path):
    output = tmp_path / "seed_66"
    output.mkdir()
    (output / "metrics.json").write_text(
        json.dumps(
            {
                "variant": "mask-aware-sam-backbone",
                "seed": 66,
                "selection_split": "test",
                "history_epochs": 100,
                "collapsed": False,
                "test": {"weighted_f1": 0.9},
            }
        )
    )

    inspection = inspect_result(output, expected_seed=66, expected_epochs=100)

    assert not inspection.complete
    assert "selection" in inspection.reason


@pytest.mark.parametrize("missing_name", ["history.json", "predictions.npz"])
def test_inspect_result_requires_complete_artifact_set(tmp_path, missing_name):
    output = tmp_path / "seed_66"
    output.mkdir()
    metrics = {
        "variant": "mask-aware-sam-backbone",
        "seed": 66,
        "selection_split": "validation",
        "history_epochs": 100,
        "collapsed": False,
        "test": {"weighted_f1": 0.9},
    }
    (output / "metrics.json").write_text(json.dumps(metrics))
    for name in ("config.json", "history.json", "best_checkpoint.pt", "predictions.npz"):
        if name != missing_name:
            (output / name).write_bytes(b"complete")

    inspection = inspect_result(output, expected_seed=66, expected_epochs=100)

    assert not inspection.complete
    assert missing_name in inspection.reason
