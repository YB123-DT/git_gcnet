import json
from pathlib import Path

import pytest

from scripts.run_mosi_target_private_ab import RATES, _build_jobs, _is_complete, main, summarize_jobs


def _normalized(command):
    values = list(command)
    for option in ("--output-dir", "--target-private-rank"):
        index = values.index(option)
        del values[index : index + 2]
    return tuple(values)


def test_jobs_are_strict_five_seed_pairs(tmp_path):
    jobs = _build_jobs(tmp_path, tmp_path / "out", Path("python"), Path("/features"), (0, 1, 2, 3))
    assert len(jobs) == 10
    assert {job.arm for job in jobs} == {"shared", "target-private"}
    assert {job.seed for job in jobs} == {66, 67, 68, 69, 70}
    assert all(job.gpu != 4 for job in jobs)
    for seed in range(66, 71):
        pair = [job for job in jobs if job.seed == seed]
        assert _normalized(pair[0].command) == _normalized(pair[1].command)
        assert {job.target_private_rank for job in pair} == {0, 32}


def _complete(job, delta=0.0, wrong_rank=None):
    job.output_dir.mkdir(parents=True, exist_ok=True)
    (job.output_dir / "config.json").write_text(json.dumps({
        "dataset": "CMUMOSI", "seed": job.seed, "train_rate_mode": "stratified",
        "target_private_rank": job.target_private_rank if wrong_rank is None else wrong_rank,
    }))
    (job.output_dir / "history.json").write_text(json.dumps([{}] * 100))
    (job.output_dir / "best.pt").write_bytes(b"checkpoint")
    (job.output_dir / "metrics.json").write_text(json.dumps({
        "best_epoch": 10, "selection_split": "validation", "parameter_count": 100 + job.target_private_rank,
        "test": {str(rate): {"weighted_f1": 0.70 + delta, "mask_sha256": f"{job.seed}-{rate}"} for rate in RATES},
    }))


def test_completion_rejects_wrong_rank_and_partial_json(tmp_path):
    job = _build_jobs(tmp_path, tmp_path / "out", Path("python"), Path("/features"), (0,))[0]
    _complete(job)
    assert _is_complete(job)
    _complete(job, wrong_rank=99)
    assert not _is_complete(job)
    (job.output_dir / "metrics.json").write_text("{")
    assert not _is_complete(job)


def test_summary_computes_paired_gate(tmp_path):
    jobs = _build_jobs(tmp_path, tmp_path / "out", Path("python"), Path("/features"), (0, 1))
    for job in jobs:
        _complete(job, delta=0.01 if job.arm == "target-private" else 0.0)
    summary = summarize_jobs(jobs)
    assert summary["complete"] is True
    assert summary["rate_summary"]["0.5"]["mean_delta"] == pytest.approx(0.01)
    assert summary["nonzero_macro"]["positive_seed_count"] == 5
    assert summary["verdict"]["passes_predefined_gate"] is True


def test_resume_with_complete_jobs_writes_terminal_status(tmp_path, monkeypatch):
    output_root = tmp_path / "out"
    jobs = _build_jobs(tmp_path, output_root, Path("/python"), Path("/features"), (0,))
    for job in jobs:
        _complete(job, delta=0.01 if job.arm == "target-private" else 0.0)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_mosi_target_private_ab.py",
            "--repo-root", str(tmp_path),
            "--output-root", str(output_root),
            "--python", "/python",
            "--feature-root", "/features",
            "--gpus", "0",
        ],
    )
    assert main() == 0
    status = json.loads((output_root / "runner_status.json").read_text())
    assert status == {"pending": 0, "running": [], "failures": []}
