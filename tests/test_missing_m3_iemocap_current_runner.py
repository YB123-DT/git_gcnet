from pathlib import Path

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
