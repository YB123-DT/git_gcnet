from pathlib import Path

import pytest

from scripts.run_plci_single_view_iemocap6 import _parse_args, build_jobs


def test_stage1_builds_only_fifteen_single_view_jobs():
    jobs = build_jobs(
        output_root=Path("/outputs"),
        python="/env/bin/python",
        gpus=(0, 1, 2),
        jobs_per_gpu=3,
        rates=(0.0, 0.5, 0.7),
        seeds=(66, 67, 68, 69, 70),
        epochs=100,
    )

    assert len(jobs) == 15
    assert len({job.identity for job in jobs}) == 15
    assert len({job.output_dir for job in jobs}) == 15
    assert {job.method for job in jobs} == {"plci-single"}
    assert {job.rate for job in jobs} == {0.0, 0.5, 0.7}
    assert {job.seed for job in jobs} == {66, 67, 68, 69, 70}
    assert all("--jepa-architecture" in job.command for job in jobs)
    assert all("plci-single" in job.command for job in jobs)
    assert all("--loss-recon" in job.command for job in jobs)
    assert all("--jepa-weight" in job.command for job in jobs)
    assert all("original" not in job.command for job in jobs)


def test_stage1_matches_dual_view_protocol_except_architecture_and_output():
    job = build_jobs(
        output_root=Path("/outputs"),
        python="/env/bin/python",
        gpus=(0,),
        jobs_per_gpu=1,
        rates=(0.5,),
        seeds=(66,),
        epochs=100,
    )[0]
    command = list(job.command)

    assert command[command.index("--dataset") + 1] == "IEMOCAPSix"
    assert command[command.index("--fold") + 1] == "5"
    assert command[command.index("--seed") + 1] == "66"
    assert command[command.index("--mask-type") + 1] == "constant-0.5"
    assert command[command.index("--evaluation-protocol") + 1] == "official"
    assert command[command.index("--stability-recon-weight") + 1] == "0"
    assert command[command.index("--model-variant") + 1] == "addon"
    assert command[command.index("--jepa-weight") + 1] == "0.1"
    assert job.dual_view_dir == Path(
        "/data2/yb/paper/experiments/plci_jepa_iemocap6_20260826"
        "/formal/miss_0.5/seed_66"
    )


def test_continuation_accepts_all_unrun_rates_without_repeating_completed_rates():
    jobs = build_jobs(
        output_root=Path("/outputs"),
        python="/env/bin/python",
        gpus=(0, 1, 2),
        jobs_per_gpu=3,
        rates=(0.1, 0.2, 0.3, 0.4, 0.6),
        seeds=(66, 67, 68, 69, 70),
        epochs=100,
    )

    assert len(jobs) == 25
    assert {job.rate for job in jobs} == {0.1, 0.2, 0.3, 0.4, 0.6}
    assert {job.seed for job in jobs} == {66, 67, 68, 69, 70}


def test_cli_selects_continuation_rates_and_seeds_without_control_audit():
    args = _parse_args(
        [
            "--output-root",
            "/outputs",
            "--python",
            "/env/bin/python",
            "--missing-rates",
            "0.1",
            "0.2",
            "0.3",
            "0.4",
            "0.6",
            "--seeds",
            "66",
            "67",
            "68",
            "69",
            "70",
        ]
    )

    assert args.missing_rates == [0.1, 0.2, 0.3, 0.4, 0.6]
    assert args.seeds == [66, 67, 68, 69, 70]
    assert args.audit_dual_view_controls is False


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"gpus": ()}, "GPU"),
        ({"gpus": (4,)}, "GPU 4"),
        ({"jobs_per_gpu": 4}, "jobs_per_gpu"),
        ({"rates": (0.8,)}, "rate"),
        ({"seeds": (71,)}, "seed"),
        ({"epochs": 0}, "epochs"),
    ],
)
def test_stage1_rejects_unsafe_or_out_of_scope_matrix(kwargs, message):
    options = dict(
        output_root=Path("/outputs"),
        python="/env/bin/python",
        gpus=(0, 1, 2),
        jobs_per_gpu=3,
        rates=(0.0, 0.5, 0.7),
        seeds=(66, 67, 68, 69, 70),
        epochs=100,
    )
    options.update(kwargs)

    with pytest.raises(ValueError, match=message):
        build_jobs(**options)
