import json
from pathlib import Path

import pytest

from scripts import run_missing_m3_iemocap_current as runner


def _argument(command: tuple[str, ...], name: str) -> str:
    index = command.index(name)
    return command[index + 1]


def test_default_job_matrix_is_unchanged(tmp_path):
    jobs = runner._build_jobs(
        repo_root=tmp_path,
        output_root=tmp_path / "results",
        python=Path("/env/bin/python"),
        gpus=(2, 3, 7),
    )

    assert len(jobs) == 10
    assert {(job.dataset, job.seed) for job in jobs} == {
        (dataset, seed)
        for dataset in runner.DATASETS
        for seed in runner.SEEDS
    }
    assert {_argument(job.command, "--jepa-weight") for job in jobs} == {"0.1"}
    assert {_argument(job.command, "--train-rate-mode") for job in jobs} == {"all"}
    assert {job.model_arm for job in jobs} == {"missing-m3"}
    assert {
        job.command[job.command.index("-m") + 1] for job in jobs
    } == {"gcnet_missing_m3.train_gcnet"}


def test_stratified_iemocap6_matrix_forwards_train_rate_mode(tmp_path):
    jobs = runner._build_jobs(
        repo_root=tmp_path,
        output_root=tmp_path / "results",
        python=Path("/env/bin/python"),
        gpus=(0, 1, 2),
        datasets=("IEMOCAPSix",),
        jepa_weight=0.1,
        train_rate_mode="stratified",
    )

    assert len(jobs) == 5
    assert all(
        _argument(job.command, "--train-rate-mode") == "stratified" for job in jobs
    )


def test_no_jepa_iemocap6_matrix_contains_only_five_control_jobs(tmp_path):
    jobs = runner._build_jobs(
        repo_root=tmp_path,
        output_root=tmp_path / "results",
        python=Path("/env/bin/python"),
        gpus=(0, 1, 2),
        datasets=("IEMOCAPSix",),
        jepa_weight=0.0,
    )

    assert len(jobs) == 5
    assert {(job.dataset, job.seed) for job in jobs} == {
        ("IEMOCAPSix", seed) for seed in runner.SEEDS
    }
    assert {_argument(job.command, "--jepa-weight") for job in jobs} == {"0.0"}
    assert all(job.output_dir == tmp_path / "results" / "IEMOCAPSix" / f"seed_{job.seed}" for job in jobs)


def test_original_gcnet_jobs_use_matched_trainer_without_missing_m3_flags(tmp_path):
    jobs = runner._build_jobs(
        repo_root=tmp_path,
        output_root=tmp_path / "results",
        python=Path("/env/bin/python"),
        gpus=(0, 1, 2),
        datasets=("IEMOCAPSix",),
        jepa_weight=0.0,
        train_rate_mode="stratified",
        model_arm="original-gcnet",
    )

    assert len(jobs) == 5
    assert {job.model_arm for job in jobs} == {"original-gcnet"}
    assert {
        job.command[job.command.index("-m") + 1] for job in jobs
    } == {"gcnet_original_stratified.train_gcnet"}
    assert all("--jepa-weight" not in job.command for job in jobs)
    assert all("--fusion-type" not in job.command for job in jobs)
    assert all("--mmoe-variant" not in job.command for job in jobs)
    assert {
        _argument(job.command, "--train-rate-mode") for job in jobs
    } == {"stratified"}


def test_original_gcnet_jobs_reject_nonzero_jepa_weight(tmp_path):
    with pytest.raises(ValueError, match="jepa_weight=0"):
        runner._build_jobs(
            repo_root=tmp_path,
            output_root=tmp_path / "results",
            python=Path("/env/bin/python"),
            gpus=(0,),
            datasets=("IEMOCAPSix",),
            jepa_weight=0.1,
            train_rate_mode="stratified",
            model_arm="original-gcnet",
        )


@pytest.mark.parametrize("mode", ("fixed", "cyclic", "all"))
def test_original_gcnet_jobs_reject_non_stratified_modes(tmp_path, mode):
    with pytest.raises(ValueError, match="stratified"):
        runner._build_jobs(
            repo_root=tmp_path,
            output_root=tmp_path / "results",
            python=Path("/env/bin/python"),
            gpus=(0,),
            datasets=("IEMOCAPSix",),
            jepa_weight=0.0,
            train_rate_mode=mode,
            model_arm="original-gcnet",
        )


def test_original_completion_requires_control_provenance(tmp_path):
    job = runner._build_jobs(
        repo_root=tmp_path,
        output_root=tmp_path / "results",
        python=Path("/env/bin/python"),
        gpus=(0,),
        datasets=("IEMOCAPSix",),
        jepa_weight=0.0,
        train_rate_mode="stratified",
        model_arm="original-gcnet",
    )[0]
    job.output_dir.mkdir(parents=True)
    (job.output_dir / "history.json").write_text(
        json.dumps([{}] * 100), encoding="utf-8"
    )
    provenance = {
        "best_epoch": 10,
        "test": {str(index): {} for index in range(8)},
        "model_arm": "original-gcnet",
        "training_objective": "classification-plus-masked-reconstruction",
        "reconstruction_loss_variant": "corrected-formal-repo",
        "reconstruction_weight": 1.0,
        "ema_steps": 0,
        "selection_split": "validation",
    }
    metrics_path = job.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(provenance), encoding="utf-8")

    assert runner._is_complete(job)
    for field in (
        "model_arm",
        "training_objective",
        "reconstruction_loss_variant",
        "reconstruction_weight",
        "ema_steps",
        "selection_split",
    ):
        invalid = dict(provenance)
        invalid.pop(field)
        metrics_path.write_text(json.dumps(invalid), encoding="utf-8")
        assert not runner._is_complete(job), field


def test_original_classification_only_jobs_change_only_the_objective_flag(tmp_path):
    jobs = runner._build_jobs(
        repo_root=tmp_path,
        output_root=tmp_path / "results",
        python=Path("/env/bin/python"),
        gpus=(0, 1),
        datasets=("IEMOCAPSix",),
        jepa_weight=0.0,
        train_rate_mode="stratified",
        model_arm="original-gcnet-cls-only",
    )

    assert len(jobs) == 5
    assert {job.model_arm for job in jobs} == {"original-gcnet-cls-only"}
    assert {
        job.command[job.command.index("-m") + 1] for job in jobs
    } == {"gcnet_original_stratified.train_gcnet"}
    assert {
        _argument(job.command, "--reconstruction-weight") for job in jobs
    } == {"0.0"}
    assert all("--jepa-weight" not in job.command for job in jobs)
    assert all("--fusion-type" not in job.command for job in jobs)


def test_classification_only_completion_rejects_full_original_metadata(tmp_path):
    job = runner._build_jobs(
        repo_root=tmp_path,
        output_root=tmp_path / "results",
        python=Path("/env/bin/python"),
        gpus=(0,),
        datasets=("IEMOCAPSix",),
        jepa_weight=0.0,
        train_rate_mode="stratified",
        model_arm="original-gcnet-cls-only",
    )[0]
    job.output_dir.mkdir(parents=True)
    (job.output_dir / "history.json").write_text(
        json.dumps([{}] * 100), encoding="utf-8"
    )
    metrics = {
        "best_epoch": 10,
        "test": {str(index): {} for index in range(8)},
        "model_arm": "original-gcnet",
        "training_objective": "classification-plus-masked-reconstruction",
        "reconstruction_loss_variant": "corrected-formal-repo",
        "reconstruction_weight": 1.0,
        "ema_steps": 0,
        "selection_split": "validation",
    }
    metrics_path = job.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    assert not runner._is_complete(job)

    metrics.update(
        {
            "model_arm": "original-gcnet-cls-only",
            "training_objective": "classification-only",
            "reconstruction_weight": 0.0,
        }
    )
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    assert runner._is_complete(job)
