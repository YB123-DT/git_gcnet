from collections import Counter
import builtins
from pathlib import Path

from scripts.run_mosi_hidden_window_sweep import (
    build_command,
    build_jobs,
    pending_jobs,
    write_manifest,
)


def test_stage1_matrix_assigns_twelve_unique_configs_to_each_gpu(tmp_path):
    jobs = build_jobs(
        seeds=(66, 67, 68),
        gpus=(0, 1, 2),
        output_root=tmp_path,
    )

    assert len(jobs) == 36
    assert len({job.output_dir for job in jobs}) == 36
    assert Counter(job.gpu for job in jobs) == {0: 12, 1: 12, 2: 12}
    assert {(job.seed, job.gpu) for job in jobs} == {
        (66, 0),
        (67, 1),
        (68, 2),
    }
    expected_grid = {
        (hidden, window)
        for hidden in (50, 100, 200)
        for window in (1, 2, 3, 4)
    }
    for seed in (66, 67, 68):
        actual_grid = {
            (job.hidden, job.window) for job in jobs if job.seed == seed
        }
        assert actual_grid == expected_grid


def test_training_command_locks_formal_slot_protocol_without_time_attention(
    tmp_path,
):
    job = build_jobs(
        seeds=(66,),
        gpus=(0,),
        output_root=tmp_path,
    )[0]
    command = build_command(
        job,
        python_executable=Path("/env/bin/python"),
    )

    joined = " ".join(command)
    assert command[:3] == [
        "/env/bin/python",
        "-m",
        "gcnet_missing_m3.train_gcnet",
    ]
    assert "--time-attn" not in command
    for expected in (
        "--dataset CMUMOSI",
        "--feature-root /data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset/CMUMOSI/features",
        "--fold 1",
        "--train-rate-mode all",
        "--fusion-type slot",
        "--representation-type slot",
        "--mosi-task-mode regression",
        "--graph-branch-mode both",
        "--lr 0.0005",
        "--l2 1e-05",
        "--jepa-weight 0.1",
        "--num-threads 2",
    ):
        assert expected in joined
    assert f"--hidden {job.hidden}" in joined
    assert f"--windowp {job.window}" in joined
    assert f"--windowf {job.window}" in joined
    assert str(job.output_dir) in command


def test_completed_metrics_are_skipped_and_manifest_is_deterministic(tmp_path):
    jobs = build_jobs(
        seeds=(66,),
        gpus=(0,),
        output_root=tmp_path / "results",
    )
    completed = jobs[0]
    completed.output_dir.mkdir(parents=True)
    (completed.output_dir / "metrics.json").write_text("{}", encoding="utf-8")

    assert pending_jobs(jobs) == jobs[1:]

    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest_path, jobs)
    first = manifest_path.read_text(encoding="utf-8")
    write_manifest(manifest_path, jobs)
    second = manifest_path.read_text(encoding="utf-8")
    assert first == second
    assert '"time_attention": false' in first
    assert '"num_threads": 2' in first


def test_matrix_generation_is_compatible_with_legacy_python_zip(
    monkeypatch, tmp_path
):
    original_zip = builtins.zip

    def legacy_zip(*args, **kwargs):
        if kwargs:
            raise TypeError("zip() takes no keyword arguments")
        return original_zip(*args)

    monkeypatch.setattr(builtins, "zip", legacy_zip)
    jobs = build_jobs(seeds=(66,), gpus=(5,), output_root=tmp_path)
    assert len(jobs) == 12
