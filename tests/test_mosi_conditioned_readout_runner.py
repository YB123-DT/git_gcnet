from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path

import pytest


def _runner():
    name = "scripts.run_mosi_conditioned_readout"
    assert importlib.util.find_spec(name) is not None, "runner is not implemented"
    return importlib.import_module(name)


def _write_compatible_run(job) -> None:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "dataset": "CMUMOSI",
        "fold": 1,
        "seed": job.seed,
        "window_past": 1,
        "window_future": 1,
        "hidden": 100,
        "batch_size": 32,
        "epochs": 100,
        "learning_rate": 0.0005,
        "weight_decay": 0.00001,
        "fusion_type": "slot",
        "representation_type": "slot",
        "train_rate_mode": "all",
        "mosi_task_mode": "regression",
        "graph_branch_mode": "both",
        "jepa_weight": 0.1,
        "jepa_regression_aggregation": job.jepa_regression_aggregation,
        "time_attention": False,
        "readout_type": job.readout_type,
        "readout_rank": job.rank,
        "recurrent_padding_mode": job.recurrent_padding_mode,
        "task_regression_loss": job.task_regression_loss,
        "task_smooth_l1_beta": job.task_smooth_l1_beta,
        "postgraph_sequence_mode": job.postgraph_sequence_mode,
        "jepa_rate_weighting": job.jepa_rate_weighting,
        "graph_message_calibration": job.graph_message_calibration,
        "evaluate_test": False,
    }
    metrics = {
        "evaluation_stage": "train-validation-only",
        "readout_type": job.readout_type,
        "readout_rank": job.rank,
        "jepa_regression_aggregation": job.jepa_regression_aggregation,
        "recurrent_padding_mode": job.recurrent_padding_mode,
        "task_regression_loss": job.task_regression_loss,
        "task_smooth_l1_beta": job.task_smooth_l1_beta,
        "postgraph_sequence_mode": job.postgraph_sequence_mode,
        "jepa_rate_weighting": job.jepa_rate_weighting,
        "graph_message_calibration": job.graph_message_calibration,
    }
    history = [{"epoch": epoch} for epoch in range(1, 101)]
    (job.output_dir / "config.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    (job.output_dir / "metrics.json").write_text(
        json.dumps(metrics), encoding="utf-8"
    )
    (job.output_dir / "history.json").write_text(
        json.dumps(history), encoding="utf-8"
    )


def test_screen_jobs_train_only_the_three_candidate_seeds(tmp_path):
    module = _runner()

    jobs = module.build_jobs(
        seeds=(66, 67, 68),
        gpus=(7, 7, 7),
        output_root=tmp_path,
        readout_type="availability-low-rank",
        rank=8,
    )

    assert len(jobs) == 3
    assert [job.seed for job in jobs] == [66, 67, 68]
    assert all(job.gpu == 7 for job in jobs)
    assert all(job.readout_type == "availability-low-rank" for job in jobs)
    assert all(job.rank == 8 for job in jobs)
    assert [job.output_dir for job in jobs] == [
        tmp_path / "availability-low-rank_rank8" / "seed_66",
        tmp_path / "availability-low-rank_rank8" / "seed_67",
        tmp_path / "availability-low-rank_rank8" / "seed_68",
    ]


def test_job_matrix_rejects_gpu4_and_mismatched_seed_gpu_counts(tmp_path):
    module = _runner()

    with pytest.raises(ValueError, match="same length"):
        module.build_jobs(
            seeds=(66, 67),
            gpus=(7,),
            output_root=tmp_path,
            readout_type="availability-low-rank",
            rank=8,
        )
    with pytest.raises(ValueError, match="GPU 4"):
        module.build_jobs(
            seeds=(66,),
            gpus=(4,),
            output_root=tmp_path,
            readout_type="availability-low-rank",
            rank=8,
        )


def test_affine_fallback_uses_a_distinct_rank_free_output_directory(tmp_path):
    module = _runner()
    job = module.build_jobs(
        seeds=(66,),
        gpus=(7,),
        output_root=tmp_path,
        readout_type="availability-affine",
        rank=8,
    )[0]

    assert job.output_dir == tmp_path / "availability-affine" / "seed_66"
    command = " ".join(module.build_command(job, Path("/env/bin/python")))
    assert "--readout-type availability-affine" in command


def test_utterance_balanced_jepa_is_a_shared_readout_single_variable_job(tmp_path):
    module = _runner()
    job = module.build_jobs(
        seeds=(66,),
        gpus=(7,),
        output_root=tmp_path,
        readout_type="shared",
        rank=8,
        jepa_regression_aggregation="utterance",
    )[0]

    assert job.output_dir == tmp_path / "utterance-balanced-jepa" / "seed_66"
    command = " ".join(module.build_command(job, Path("/env/bin/python")))
    assert "--readout-type shared" in command
    assert "--jepa-regression-aggregation utterance" in command

    with pytest.raises(ValueError, match="cannot be combined"):
        module.build_jobs(
            seeds=(66,),
            gpus=(7,),
            output_root=tmp_path,
            readout_type="availability-low-rank",
            rank=8,
            jepa_regression_aggregation="utterance",
        )


def test_packed_recurrent_builds_fresh_deterministic_paired_jobs(tmp_path):
    module = _runner()
    candidates = module.build_jobs(
        seeds=(66, 67, 68),
        gpus=(7, 7, 7),
        output_root=tmp_path,
        readout_type="shared",
        rank=8,
        jepa_regression_aggregation="target",
        recurrent_padding_mode="packed",
    )
    controls = module.build_fresh_legacy_controls(candidates, tmp_path)

    assert [job.output_dir for job in candidates] == [
        tmp_path / "packed-recurrent" / "seed_66",
        tmp_path / "packed-recurrent" / "seed_67",
        tmp_path / "packed-recurrent" / "seed_68",
    ]
    assert [job.output_dir for job in controls] == [
        tmp_path / "deterministic-legacy" / "seed_66",
        tmp_path / "deterministic-legacy" / "seed_67",
        tmp_path / "deterministic-legacy" / "seed_68",
    ]
    assert all(job.recurrent_padding_mode == "packed" for job in candidates)
    assert all(job.recurrent_padding_mode == "legacy" for job in controls)
    candidate_command = " ".join(
        module.build_command(candidates[0], Path("/env/bin/python"))
    )
    control_command = " ".join(
        module.build_command(controls[0], Path("/env/bin/python"))
    )
    assert "--recurrent-padding-mode packed" in candidate_command
    assert "--recurrent-padding-mode legacy" in control_command


def test_packed_recurrent_cli_requires_fresh_controls_and_no_bundled_change():
    module = _runner()
    parser = module.build_parser()
    common = [
        "--readout-type",
        "shared",
        "--jepa-regression-aggregation",
        "target",
        "--recurrent-padding-mode",
        "packed",
    ]

    with pytest.raises(ValueError, match="requires fresh"):
        module.validate_run_arguments(parser.parse_args(common))

    valid = parser.parse_args([*common, "--fresh-legacy-control"])
    module.validate_run_arguments(valid)

    bundled = parser.parse_args(
        [
            "--readout-type",
            "availability-affine",
            "--recurrent-padding-mode",
            "packed",
            "--fresh-legacy-control",
        ]
    )
    with pytest.raises(ValueError, match="packed-only"):
        module.validate_run_arguments(bundled)


def test_smooth_l1_task_loss_is_a_legacy_shared_single_variable_job(tmp_path):
    module = _runner()
    job = module.build_jobs(
        seeds=(66,),
        gpus=(7,),
        output_root=tmp_path,
        readout_type="shared",
        rank=8,
        task_regression_loss="smooth-l1",
        task_smooth_l1_beta=1.0,
    )[0]

    assert job.output_dir == tmp_path / "smooth-l1-task_beta1" / "seed_66"
    assert job.recurrent_padding_mode == "legacy"
    command = " ".join(module.build_command(job, Path("/env/bin/python")))
    assert "--task-regression-loss smooth-l1" in command
    assert "--task-smooth-l1-beta 1.0" in command

    with pytest.raises(ValueError, match="cannot be combined"):
        module.build_jobs(
            seeds=(66,),
            gpus=(7,),
            output_root=tmp_path,
            readout_type="availability-affine",
            rank=8,
            task_regression_loss="smooth-l1",
        )
    with pytest.raises(ValueError, match="cannot be combined"):
        module.build_jobs(
            seeds=(66,),
            gpus=(7,),
            output_root=tmp_path,
            readout_type="shared",
            rank=8,
            recurrent_padding_mode="packed",
            task_regression_loss="smooth-l1",
        )


def test_shared_postgraph_bilstm_is_the_only_registered_treatment(tmp_path):
    module = _runner()
    job = module.build_jobs(
        seeds=(66,),
        gpus=(7,),
        output_root=tmp_path,
        readout_type="shared",
        rank=8,
        postgraph_sequence_mode="shared-bilstm",
    )[0]

    assert job.output_dir == tmp_path / "shared-postgraph-bilstm" / "seed_66"
    command = " ".join(module.build_command(job, Path("/env/bin/python")))
    assert "--postgraph-sequence-mode shared-bilstm" in command
    assert "--task-regression-loss mse" in command
    assert "--recurrent-padding-mode legacy" in command

    combinations = (
        {"readout_type": "availability-affine"},
        {"jepa_regression_aggregation": "utterance"},
        {"recurrent_padding_mode": "packed"},
        {"task_regression_loss": "smooth-l1"},
    )
    for change in combinations:
        arguments = {
            "seeds": (66,),
            "gpus": (7,),
            "output_root": tmp_path,
            "readout_type": "shared",
            "rank": 8,
            "postgraph_sequence_mode": "shared-bilstm",
            **change,
        }
        with pytest.raises(ValueError, match="cannot be combined"):
            module.build_jobs(**arguments)


def test_sparsity_weighted_jepa_is_a_single_variable_job(tmp_path):
    module = _runner()
    job = module.build_jobs(
        seeds=(66,),
        gpus=(7,),
        output_root=tmp_path,
        readout_type="shared",
        rank=8,
        jepa_rate_weighting="sparsity-budget",
    )[0]

    assert job.output_dir == tmp_path / "sparsity-weighted-jepa" / "seed_66"
    command = " ".join(module.build_command(job, Path("/env/bin/python")))
    assert "--jepa-rate-weighting sparsity-budget" in command
    assert "--task-regression-loss mse" in command
    assert "--postgraph-sequence-mode independent" in command

    with pytest.raises(ValueError, match="cannot be combined"):
        module.build_jobs(
            seeds=(66,),
            gpus=(7,),
            output_root=tmp_path,
            readout_type="shared",
            rank=8,
            task_regression_loss="smooth-l1",
            jepa_rate_weighting="sparsity-budget",
        )


def test_branch_graph_message_calibration_is_a_single_variable_job(tmp_path):
    module = _runner()
    job = module.build_jobs(
        seeds=(66,),
        gpus=(7,),
        output_root=tmp_path,
        readout_type="shared",
        rank=8,
        graph_message_calibration="branch-layernorm-residual",
    )[0]

    assert (
        job.output_dir
        == tmp_path / "branch-graph-message-calibration" / "seed_66"
    )
    command = " ".join(module.build_command(job, Path("/env/bin/python")))
    assert (
        "--graph-message-calibration branch-layernorm-residual" in command
    )
    assert "--jepa-rate-weighting uniform" in command
    assert "--postgraph-sequence-mode independent" in command

    with pytest.raises(ValueError, match="cannot be combined"):
        module.build_jobs(
            seeds=(66,),
            gpus=(7,),
            output_root=tmp_path,
            readout_type="shared",
            rank=8,
            graph_message_calibration="branch-layernorm-residual",
            jepa_rate_weighting="sparsity-budget",
        )


def test_shared_postgraph_confirmation_builds_only_missing_direct_controls(
    tmp_path,
):
    module = _runner()
    candidates = module.build_jobs(
        seeds=(69, 70),
        gpus=(7, 7),
        output_root=tmp_path / "results",
        readout_type="shared",
        rank=8,
        postgraph_sequence_mode="shared-bilstm",
    )

    controls = module.build_confirmation_legacy_controls(
        candidates,
        tmp_path / "results" / "deterministic-legacy",
    )

    assert [job.seed for job in controls] == [69, 70]
    assert [job.output_dir for job in controls] == [
        tmp_path / "results" / "deterministic-legacy" / "seed_69",
        tmp_path / "results" / "deterministic-legacy" / "seed_70",
    ]
    assert all(job.postgraph_sequence_mode == "independent" for job in controls)
    assert all(job.task_regression_loss == "mse" for job in controls)


def test_confirmation_cli_is_locked_to_missing_shared_postgraph_seeds():
    module = _runner()
    parser = module.build_parser()
    valid = parser.parse_args(
        [
            "--stage",
            "confirm",
            "--seeds",
            "69",
            "70",
            "--gpus",
            "7",
            "7",
            "--readout-type",
            "shared",
            "--postgraph-sequence-mode",
            "shared-bilstm",
        ]
    )
    module.validate_run_arguments(valid)

    wrong_seeds = parser.parse_args(
        [
            "--stage",
            "confirm",
            "--seeds",
            "68",
            "69",
            "--gpus",
            "7",
            "7",
            "--readout-type",
            "shared",
            "--postgraph-sequence-mode",
            "shared-bilstm",
        ]
    )
    with pytest.raises(ValueError, match="seeds 69 and 70"):
        module.validate_run_arguments(wrong_seeds)

    wrong_treatment = parser.parse_args(
        [
            "--stage",
            "confirm",
            "--seeds",
            "69",
            "70",
            "--gpus",
            "7",
            "7",
            "--readout-type",
            "shared",
            "--task-regression-loss",
            "smooth-l1",
        ]
    )
    with pytest.raises(ValueError, match="shared-postgraph-only"):
        module.validate_run_arguments(wrong_treatment)


def test_confirmation_rejects_a_truncated_completed_screen_history(tmp_path):
    module = _runner()
    candidates = module.build_jobs(
        seeds=(66, 67, 68),
        gpus=(7, 7, 7),
        output_root=tmp_path,
        readout_type="shared",
        rank=8,
        postgraph_sequence_mode="shared-bilstm",
    )
    controls = module.build_confirmation_legacy_controls(
        candidates, tmp_path / "deterministic-legacy"
    )
    for job in [*candidates, *controls]:
        _write_compatible_run(job)
    variant_root = tmp_path / "shared-postgraph-bilstm"
    (variant_root / "VALIDATION_SUMMARY.json").write_text(
        json.dumps({"gate": {"passed": True}}), encoding="utf-8"
    )
    (candidates[1].output_dir / "history.json").write_text(
        json.dumps([{"epoch": epoch} for epoch in range(1, 11)]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="completed candidate seed 67"):
        module.validate_confirmation_prerequisites(
            variant_root,
            tmp_path / "deterministic-legacy",
        )


def test_validation_audit_rejects_a_truncated_candidate_history(tmp_path):
    module = _runner()
    candidates = module.build_jobs(
        seeds=(66,),
        gpus=(7,),
        output_root=tmp_path,
        readout_type="shared",
        rank=8,
        postgraph_sequence_mode="shared-bilstm",
    )
    controls = module.build_confirmation_legacy_controls(
        candidates, tmp_path / "deterministic-legacy"
    )
    for job in [*candidates, *controls]:
        _write_compatible_run(job)
    (candidates[0].output_dir / "history.json").write_text(
        json.dumps([{"epoch": epoch} for epoch in range(1, 11)]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="completed candidate seed 66"):
        module.write_validation_summary(
            tmp_path / "summary.json",
            candidates,
            control_root=tmp_path / "deterministic-legacy",
            control_jobs=controls,
        )


def test_new_candidate_requires_direct_deterministic_control_before_training(
    tmp_path,
):
    module = _runner()
    jobs = module.build_jobs(
        seeds=(66, 67, 68),
        gpus=(7, 7, 7),
        output_root=tmp_path / "candidate",
        readout_type="shared",
        rank=8,
        postgraph_sequence_mode="shared-bilstm",
    )
    direct_root = tmp_path / "deterministic-legacy"
    for seed in (66, 67, 68):
        run = direct_root / f"seed_{seed}"
        run.mkdir(parents=True)
        (run / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="completed direct deterministic Legacy"):
        module.validate_inherited_control_layout(jobs, direct_root)
    for seed in (66, 67, 68):
        run = direct_root / f"seed_{seed}"
        (run / "history.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="completed direct deterministic Legacy"):
        module.validate_inherited_control_layout(jobs, direct_root)

    controls = module.build_confirmation_legacy_controls(jobs, direct_root)
    for control in controls:
        _write_compatible_run(control)
    module.validate_inherited_control_layout(jobs, direct_root)

    legacy_layout = tmp_path / "old-hidden-window"
    for seed in (66, 67, 68):
        run = legacy_layout / f"seed_{seed}" / "hidden_100_window_1"
        run.mkdir(parents=True)
        (run / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="direct deterministic Legacy"):
        module.validate_inherited_control_layout(jobs, legacy_layout)


def test_direct_deterministic_legacy_control_layout_is_resolved(tmp_path):
    module = _runner()
    direct = tmp_path / "seed_66"
    direct.mkdir()
    (direct / "config.json").write_text("{}", encoding="utf-8")

    assert module._control_run(tmp_path, 66) == direct


def test_training_command_locks_the_single_variable_protocol(tmp_path):
    module = _runner()
    job = module.build_jobs(
        seeds=(66,),
        gpus=(7,),
        output_root=tmp_path,
        readout_type="availability-low-rank",
        rank=8,
    )[0]

    command = module.build_command(job, Path("/env/bin/python"))
    joined = " ".join(command)

    assert command[:3] == [
        "/env/bin/python",
        "-m",
        "gcnet_missing_m3.train_gcnet",
    ]
    for expected in (
        "--dataset CMUMOSI",
        "--fold 1",
        "--epochs 100",
        "--batch-size 32",
        "--train-rate-mode all",
        "--hidden 100",
        "--windowp 1",
        "--windowf 1",
        "--fusion-type slot",
        "--representation-type slot",
        "--mosi-task-mode regression",
        "--graph-branch-mode both",
        "--lr 0.0005",
        "--l2 1e-05",
        "--jepa-weight 0.1",
        "--readout-type availability-low-rank",
        "--readout-rank 8",
        "--skip-test-evaluation",
        "--num-threads 2",
    ):
        assert expected in joined
    assert "--time-attn" not in command
    assert "--classification-completion" not in command
    assert "--node-interaction-residual" not in command


def _history(overall_values, prediction_std=0.2, sign_count=2):
    history = []
    for epoch, values in enumerate(overall_values, start=1):
        validation = {}
        for index, value in enumerate(values):
            validation[f"{index / 10:.1f}"] = {
                "weighted_f1": value,
                "prediction_std": prediction_std,
                "predicted_sign_count": sign_count,
            }
        history.append(
            {
                "epoch": epoch,
                "validation": validation,
                "validation_mean_weighted_f1": sum(values) / len(values),
            }
        )
    return history


def test_validation_snapshot_uses_earliest_tied_best_epoch_and_never_reads_metrics(
    tmp_path, monkeypatch
):
    module = _runner()
    run = tmp_path / "run"
    run.mkdir()
    values = [0.80, 0.79, 0.78, 0.77, 0.76, 0.75, 0.74, 0.73]
    (run / "history.json").write_text(
        json.dumps(_history([values, values])), encoding="utf-8"
    )
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        assert path.name != "metrics.json"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)

    snapshot = module.validation_snapshot(run)

    assert snapshot["best_epoch"] == 1
    assert snapshot["overall"] == pytest.approx(sum(values) / 8)
    assert snapshot["miss0"] == pytest.approx(0.80)
    assert snapshot["high_missing"] == pytest.approx(
        sum(values[4:8]) / 4
    )
    assert snapshot["collapsed"] is False


def test_validation_snapshot_flags_all_predeclared_collapse_modes(tmp_path):
    module = _runner()
    base = [0.80] * 8
    cases = (
        ("low_f1", [0.54] + base[1:], 0.2, 2),
        ("constant", base, 1e-6, 2),
        ("one_sign", base, 0.2, 1),
    )
    for name, values, std, signs in cases:
        run = tmp_path / name
        run.mkdir()
        (run / "history.json").write_text(
            json.dumps(_history([values], std, signs)), encoding="utf-8"
        )
        assert module.validation_snapshot(run)["collapsed"] is True


def test_three_seed_gate_uses_only_paired_validation_and_all_guardrails():
    module = _runner()
    control = {
        seed: {
            "overall": 0.780,
            "miss0": 0.850,
            "high_missing": 0.730,
            "collapsed": False,
        }
        for seed in (66, 67, 68)
    }
    candidate = {
        66: {
            "overall": 0.786,
            "miss0": 0.849,
            "high_missing": 0.736,
            "collapsed": False,
        },
        67: {
            "overall": 0.784,
            "miss0": 0.848,
            "high_missing": 0.733,
            "collapsed": False,
        },
        68: {
            "overall": 0.782,
            "miss0": 0.849,
            "high_missing": 0.731,
            "collapsed": False,
        },
    }

    summary = module.paired_validation_gate(candidate, control)

    assert summary["overall_delta"] == pytest.approx(0.004)
    assert summary["positive_seeds"] == 3
    assert summary["high_missing_delta"] >= 0
    assert summary["miss0_delta"] >= -0.003
    assert summary["passed"] is True

    failed = {seed: dict(record) for seed, record in candidate.items()}
    failed[68]["collapsed"] = True
    assert module.paired_validation_gate(failed, control)["passed"] is False


def test_five_seed_confirmation_gate_requires_four_positive_seeds():
    module = _runner()
    control = {
        seed: {
            "overall": 0.780,
            "miss0": 0.850,
            "high_missing": 0.730,
            "collapsed": False,
        }
        for seed in (66, 67, 68, 69, 70)
    }
    deltas = {66: 0.010, 67: -0.001, 68: 0.003, 69: 0.002, 70: 0.001}
    candidate = {
        seed: {
            "overall": control[seed]["overall"] + delta,
            "miss0": 0.850,
            "high_missing": 0.731,
            "collapsed": False,
        }
        for seed, delta in deltas.items()
    }

    summary = module.paired_confirmation_gate(candidate, control)

    assert summary["stage"] == "five-seed-confirmation"
    assert summary["positive_seeds"] == 4
    assert summary["required_positive_seeds"] == 4
    assert summary["passed"] is True

    candidate[69]["overall"] = 0.779
    assert module.paired_confirmation_gate(candidate, control)["passed"] is False


def test_manifest_is_atomic_and_records_inherited_control_without_original_jobs(
    tmp_path,
):
    module = _runner()
    jobs = module.build_jobs(
        seeds=(66, 67, 68),
        gpus=(7, 7, 7),
        output_root=tmp_path / "candidate",
        readout_type="availability-low-rank",
        rank=8,
    )
    path = tmp_path / "manifest.json"
    control_root = Path("/controls/hidden_window")

    module.write_manifest(path, jobs, control_root=control_root)
    first = path.read_text(encoding="utf-8")
    module.write_manifest(path, jobs, control_root=control_root)

    assert path.read_text(encoding="utf-8") == first
    payload = json.loads(first)
    assert payload["control"]["policy"] == "inherit-no-retrain"
    assert payload["control"]["root"] == str(control_root)
    assert payload["condition"]["test_policy"] == "not-computed-before-gate"
    assert payload["condition"]["feature_root"] == str(module.FEATURE_ROOT)
    assert payload["condition"]["relation_mapping"]["speaker"] == {"00": 0}
    assert payload["control"]["runner_sha256"]
    assert len(payload["jobs"]) == 3
    assert all(job["readout_type"] != "shared" for job in payload["jobs"])

    changed = module.build_jobs(
        seeds=(66, 67, 68),
        gpus=(7, 7, 7),
        output_root=tmp_path / "candidate",
        readout_type="availability-low-rank",
        rank=4,
    )
    with pytest.raises(ValueError, match="immutable"):
        module.write_manifest(path, changed, control_root=control_root)


def test_fresh_control_manifest_allows_only_monotonic_artifact_enrichment(
    tmp_path,
):
    module = _runner()
    candidates = module.build_jobs(
        seeds=(66,),
        gpus=(7,),
        output_root=tmp_path,
        readout_type="shared",
        rank=8,
        recurrent_padding_mode="packed",
    )
    controls = module.build_fresh_legacy_controls(candidates, tmp_path)
    path = tmp_path / "packed-recurrent" / "MANIFEST.json"
    control_root = tmp_path / "deterministic-legacy"

    module.write_manifest(path, [*candidates, *controls], control_root=control_root)
    initial = json.loads(path.read_text(encoding="utf-8"))
    assert initial["control"]["artifact_sha256"] == {"66": {}}

    control = control_root / "seed_66"
    control.mkdir(parents=True)
    (control / "config.json").write_text("{}", encoding="utf-8")
    (control / "history.json").write_text("[]", encoding="utf-8")

    module.write_manifest(path, [*candidates, *controls], control_root=control_root)
    partial_text = path.read_text(encoding="utf-8")
    partial = json.loads(partial_text)
    assert partial["control"]["artifact_sha256"] == {"66": {}}

    _write_compatible_run(controls[0])
    module.write_manifest(path, [*candidates, *controls], control_root=control_root)
    enriched_text = path.read_text(encoding="utf-8")
    enriched = json.loads(enriched_text)
    assert set(enriched["control"]["artifact_sha256"]["66"]) == {
        "config.json",
        "history.json",
    }
    module.write_manifest(path, [*candidates, *controls], control_root=control_root)
    assert path.read_text(encoding="utf-8") == enriched_text

    tampered_history = [{"epoch": epoch} for epoch in range(1, 101)]
    tampered_history[0]["tampered"] = True
    (control / "history.json").write_text(
        json.dumps(tampered_history), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="immutable"):
        module.write_manifest(
            path,
            [*candidates, *controls],
            control_root=control_root,
        )


def test_inherited_control_audit_allows_only_the_preregistered_readout_difference(
    tmp_path,
):
    module = _runner()
    candidate = tmp_path / "candidate"
    control = tmp_path / "control"
    candidate.mkdir()
    control.mkdir()
    common = {
        "dataset": "CMUMOSI",
        "fold": 1,
        "seed": 66,
        "window_past": 1,
        "window_future": 1,
        "hidden": 100,
        "batch_size": 32,
        "epochs": 100,
        "learning_rate": 0.0005,
        "weight_decay": 0.00001,
        "fusion_type": "slot",
        "representation_type": "slot",
        "train_rate_mode": "all",
        "mosi_task_mode": "regression",
        "graph_branch_mode": "both",
        "jepa_weight": 0.1,
        "time_attention": False,
    }
    candidate_config = {
        **common,
        "readout_type": "availability-low-rank",
        "readout_rank": 8,
        "evaluate_test": False,
    }
    (candidate / "config.json").write_text(
        json.dumps(candidate_config), encoding="utf-8"
    )
    (control / "config.json").write_text(json.dumps(common), encoding="utf-8")

    audit = module.audit_inherited_control(candidate, control)

    assert audit["compatible"] is True
    assert audit["control_readout_type"] == "shared"
    assert audit["candidate_readout_type"] == "availability-low-rank"

    mismatched = dict(common)
    mismatched["hidden"] = 200
    (control / "config.json").write_text(
        json.dumps(mismatched), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="hidden"):
        module.audit_inherited_control(candidate, control)


def test_inherited_control_audit_allows_only_smooth_l1_vs_legacy_mse(tmp_path):
    module = _runner()
    candidate = tmp_path / "candidate"
    control = tmp_path / "control"
    candidate.mkdir()
    control.mkdir()
    common = {
        "dataset": "CMUMOSI",
        "fold": 1,
        "seed": 66,
        "window_past": 1,
        "window_future": 1,
        "hidden": 100,
        "batch_size": 32,
        "epochs": 100,
        "learning_rate": 0.0005,
        "weight_decay": 0.00001,
        "fusion_type": "slot",
        "representation_type": "slot",
        "train_rate_mode": "all",
        "mosi_task_mode": "regression",
        "graph_branch_mode": "both",
        "jepa_weight": 0.1,
        "time_attention": False,
        "readout_type": "shared",
        "readout_rank": 8,
        "jepa_regression_aggregation": "target",
        "recurrent_padding_mode": "legacy",
        "evaluate_test": False,
    }
    candidate_config = {
        **common,
        "task_regression_loss": "smooth-l1",
        "task_smooth_l1_beta": 1.0,
    }
    # Fresh deterministic controls predate the explicit task-loss fields. Their
    # missing values must canonicalize to the legacy MSE defaults.
    (candidate / "config.json").write_text(
        json.dumps(candidate_config), encoding="utf-8"
    )
    (control / "config.json").write_text(
        json.dumps(common), encoding="utf-8"
    )

    audit = module.audit_inherited_control(candidate, control)

    assert audit["compatible"] is True
    assert audit["candidate_task_regression_loss"] == "smooth-l1"
    assert audit["control_task_regression_loss"] == "mse"
    assert audit["task_smooth_l1_beta"] == pytest.approx(1.0)


def test_inherited_control_audit_allows_only_shared_postgraph_bilstm(tmp_path):
    module = _runner()
    candidate = tmp_path / "candidate"
    control = tmp_path / "control"
    candidate.mkdir()
    control.mkdir()
    common = {
        "dataset": "CMUMOSI",
        "fold": 1,
        "seed": 66,
        "window_past": 1,
        "window_future": 1,
        "hidden": 100,
        "batch_size": 32,
        "epochs": 100,
        "learning_rate": 0.0005,
        "weight_decay": 0.00001,
        "fusion_type": "slot",
        "representation_type": "slot",
        "train_rate_mode": "all",
        "mosi_task_mode": "regression",
        "graph_branch_mode": "both",
        "jepa_weight": 0.1,
        "time_attention": False,
        "readout_type": "shared",
        "readout_rank": 8,
        "jepa_regression_aggregation": "target",
        "recurrent_padding_mode": "legacy",
        "task_regression_loss": "mse",
        "task_smooth_l1_beta": 1.0,
        "evaluate_test": False,
    }
    (candidate / "config.json").write_text(
        json.dumps(
            {**common, "postgraph_sequence_mode": "shared-bilstm"}
        ),
        encoding="utf-8",
    )
    (control / "config.json").write_text(
        json.dumps(common), encoding="utf-8"
    )

    audit = module.audit_inherited_control(candidate, control)

    assert audit["compatible"] is True
    assert audit["candidate_postgraph_sequence_mode"] == "shared-bilstm"
    assert audit["control_postgraph_sequence_mode"] == "independent"


def test_shared_postgraph_resume_requires_exact_config_and_metrics(tmp_path):
    module = _runner()
    job = module.build_jobs(
        seeds=(66,),
        gpus=(7,),
        output_root=tmp_path,
        readout_type="shared",
        rank=8,
        postgraph_sequence_mode="shared-bilstm",
    )[0]
    job.output_dir.mkdir(parents=True)
    config = {
        "dataset": "CMUMOSI",
        "fold": 1,
        "seed": 66,
        "window_past": 1,
        "window_future": 1,
        "hidden": 100,
        "batch_size": 32,
        "epochs": 100,
        "learning_rate": 0.0005,
        "weight_decay": 0.00001,
        "fusion_type": "slot",
        "representation_type": "slot",
        "train_rate_mode": "all",
        "mosi_task_mode": "regression",
        "graph_branch_mode": "both",
        "jepa_weight": 0.1,
        "jepa_regression_aggregation": "target",
        "time_attention": False,
        "readout_type": "shared",
        "readout_rank": 8,
        "recurrent_padding_mode": "legacy",
        "task_regression_loss": "mse",
        "task_smooth_l1_beta": 1.0,
        "postgraph_sequence_mode": "shared-bilstm",
        "evaluate_test": False,
    }
    metrics = {
        "evaluation_stage": "train-validation-only",
        "readout_type": "shared",
        "readout_rank": 8,
        "jepa_regression_aggregation": "target",
        "recurrent_padding_mode": "legacy",
        "task_regression_loss": "mse",
        "task_smooth_l1_beta": 1.0,
        "postgraph_sequence_mode": "shared-bilstm",
    }
    (job.output_dir / "config.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    (job.output_dir / "metrics.json").write_text(
        json.dumps(metrics), encoding="utf-8"
    )
    (job.output_dir / "history.json").write_text(
        json.dumps([{"epoch": epoch} for epoch in range(1, 101)]),
        encoding="utf-8",
    )

    assert module.completed_job_is_compatible(job) is True
    metrics["postgraph_sequence_mode"] = "independent"
    (job.output_dir / "metrics.json").write_text(
        json.dumps(metrics), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="postgraph_sequence_mode"):
        module.completed_job_is_compatible(job)


def test_successful_runner_keeps_checkpoint_until_post_gate_audit(
    tmp_path, monkeypatch
):
    module = _runner()
    job = module.build_jobs(
        seeds=(66,),
        gpus=(7,),
        output_root=tmp_path,
        readout_type="availability-low-rank",
        rank=8,
    )[0]

    class FakeProcess:
        pid = 123

        def wait(self):
            return 0

    def fake_popen(*args, **kwargs):
        job.output_dir.mkdir(parents=True, exist_ok=True)
        (job.output_dir / "metrics.json").write_text("{}", encoding="utf-8")
        (job.output_dir / "best.pt").write_text("checkpoint", encoding="utf-8")
        return FakeProcess()

    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)

    failures = module.run_jobs(
        [job],
        python_executable=Path("/env/bin/python"),
        repo_root=tmp_path,
        max_concurrent_per_gpu=1,
    )

    assert failures == 0
    assert (job.output_dir / "best.pt").is_file()


def test_resume_accepts_only_matching_validation_only_artifacts(tmp_path):
    module = _runner()
    job = module.build_jobs(
        seeds=(66,),
        gpus=(7,),
        output_root=tmp_path,
        readout_type="availability-low-rank",
        rank=8,
    )[0]
    job.output_dir.mkdir(parents=True)
    config = {
        "dataset": "CMUMOSI",
        "fold": 1,
        "seed": 66,
        "window_past": 1,
        "window_future": 1,
        "hidden": 100,
        "batch_size": 32,
        "epochs": 100,
        "learning_rate": 0.0005,
        "weight_decay": 0.00001,
        "fusion_type": "slot",
        "representation_type": "slot",
        "train_rate_mode": "all",
        "mosi_task_mode": "regression",
        "graph_branch_mode": "both",
        "jepa_weight": 0.1,
        "time_attention": False,
        "readout_type": "availability-low-rank",
        "readout_rank": 8,
        "evaluate_test": False,
    }
    metrics = {
        "evaluation_stage": "train-validation-only",
        "readout_type": "availability-low-rank",
        "readout_rank": 8,
    }
    (job.output_dir / "config.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    (job.output_dir / "metrics.json").write_text(
        json.dumps(metrics), encoding="utf-8"
    )
    (job.output_dir / "history.json").write_text(
        json.dumps([{"epoch": epoch} for epoch in range(1, 101)]),
        encoding="utf-8",
    )

    assert module.completed_job_is_compatible(job) is True

    metrics["evaluation_stage"] = "train-validation-test"
    (job.output_dir / "metrics.json").write_text(
        json.dumps(metrics), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="evaluation_stage"):
        module.completed_job_is_compatible(job)
