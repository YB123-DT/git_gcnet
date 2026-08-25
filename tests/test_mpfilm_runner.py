import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments.mpfilm_iemocap6.run_locked_ab import (
    ARM_TO_BRANCH_FUSION,
    ARM_TO_GRAPH_VARIANT,
    ARM_TO_SECOND_GRAPH_AGGREGATION,
    _claim_job,
    _completed,
    _ensure_run_manifest,
    _job_payload,
    _launch,
    _release_job,
    build_command,
    build_jobs,
    main,
    run_jobs,
)


def _complete_log():
    return "\n".join(
        [f"epoch:{index}; train_fscore:0.0" for index in range(1, 101)]
        + ["SMOKE_ONLY=False", "Finish fold 5", "save results in archive.npz"]
    )


def _runner_inputs(job, gpu="0"):
    values = {
        "python": Path("/env/bin/python"),
        "repository": Path("/repo"),
        "data_root": Path("/data/IEMOCAP"),
        "mask_bank_root": Path("/tmp/banks"),
    }
    return values, _job_payload(job, gpu=gpu, **values)


def _write_complete_job(job, payload):
    job.output_directory.mkdir(parents=True)
    (job.output_directory / "command.json").write_text(json.dumps(payload))
    (job.output_directory / "status.json").write_text(
        json.dumps({"status": "success", "return_code": 0})
    )
    (job.output_directory / "train.log").write_text(_complete_log())
    saved = job.output_directory / "saved"
    saved.mkdir()
    (saved / "run.npz").write_bytes(b"npz")


class LockedRunnerTests(unittest.TestCase):
    def test_second_aggregation_arms_keep_original_graph_and_addition_fusion(self):
        self.assertEqual(ARM_TO_GRAPH_VARIANT["genagg"], "original")
        self.assertEqual(ARM_TO_GRAPH_VARIANT["soft_medoid"], "original")
        self.assertEqual(ARM_TO_GRAPH_VARIANT["ssma"], "original")
        self.assertEqual(ARM_TO_BRANCH_FUSION["genagg"], "addition")
        self.assertEqual(ARM_TO_BRANCH_FUSION["soft_medoid"], "addition")
        self.assertEqual(ARM_TO_BRANCH_FUSION["ssma"], "addition")
        self.assertEqual(
            ARM_TO_SECOND_GRAPH_AGGREGATION,
            {
                "genagg": "genagg",
                "soft_medoid": "soft_medoid",
                "ssma": "ssma",
            },
        )

    def test_second_aggregation_flag_is_appended_only_for_candidate_arms(self):
        jobs = build_jobs(
            "formal",
            Path("/tmp/results"),
            arms=("genagg", "soft_medoid", "ssma"),
            rates=(0.0,),
            seeds=(66,),
        )

        for job, expected in zip(jobs, ("genagg", "soft_medoid", "ssma")):
            command = build_command(
                job,
                python=Path("/env/bin/python"),
                repository=Path("/repo"),
                data_root=Path("/data/IEMOCAP"),
                mask_bank_root=Path("/tmp/banks"),
            )
            self.assertEqual(command[-2:], ["--second-graph-aggregation", expected])

        legacy_arms = tuple(
            arm
            for arm in ARM_TO_GRAPH_VARIANT
            if arm not in ARM_TO_SECOND_GRAPH_AGGREGATION
        )
        legacy_jobs = build_jobs(
            "formal",
            Path("/tmp/results"),
            arms=legacy_arms,
            rates=(0.0,),
            seeds=(66,),
        )
        for job in legacy_jobs:
            command = build_command(
                job,
                python=Path("/env/bin/python"),
                repository=Path("/repo"),
                data_root=Path("/data/IEMOCAP"),
                mask_bank_root=Path("/tmp/banks"),
            )
            self.assertNotIn("--second-graph-aggregation", command)

    def test_legacy_command_is_byte_for_byte_unchanged(self):
        job = build_jobs(
            "formal",
            Path("/tmp/results"),
            arms=("original",),
            rates=(0.7,),
            seeds=(68,),
        )[0]
        command = build_command(
            job,
            python=Path("/env/bin/python"),
            repository=Path("/repo"),
            data_root=Path("/data/IEMOCAP"),
            mask_bank_root=Path("/tmp/banks"),
        )

        self.assertEqual(
            command,
            [
                "/env/bin/python",
                "-u",
                "/repo/gcnet/train_gcnet.py",
                "--audio-feature",
                "wav2vec-large-c-UTT",
                "--text-feature",
                "deberta-large-4-UTT",
                "--video-feature",
                "manet_UTT",
                "--dataset",
                "IEMOCAPSix",
                "--data-root",
                "/data/IEMOCAP",
                "--base-model",
                "LSTM",
                "--windowp",
                "2",
                "--windowf",
                "2",
                "--hidden",
                "200",
                "--lr",
                "0.001",
                "--dropout",
                "0.5",
                "--batch-size",
                "32",
                "--num-threads",
                "6",
                "--epochs",
                "100",
                "--seed",
                "68",
                "--mask-seed",
                "68",
                "--mask-type",
                "constant-0.7",
                "--fold-index",
                "5",
                "--graph-conv-variant",
                "original",
                "--branch-fusion",
                "addition",
                "--mask-bank-root",
                "/tmp/banks",
                "--output-dir",
                "/tmp/results/formal/original/miss_0p7/seed_68/fold_5/saved",
                "--loss-recon",
            ],
        )
        self.assertNotIn("--second-graph-aggregation", command)

    def test_formal_first_wave_has_twelve_unique_candidate_jobs(self):
        jobs = build_jobs(
            "formal",
            Path("/tmp/results"),
            arms=("genagg", "soft_medoid"),
            rates=(0.0, 0.7),
            seeds=(66, 67, 68),
        )
        keys = {(job.arm, job.missing_rate, job.seed) for job in jobs}

        self.assertEqual(len(jobs), 12)
        self.assertEqual(len(keys), 12)
        self.assertNotIn("original", {job.arm for job in jobs})

    def test_ssma_discrimination_wave_has_six_jobs_and_no_original(self):
        jobs = build_jobs(
            "formal",
            Path("/tmp/results"),
            arms=("ssma",),
            rates=(0.0, 0.7),
            seeds=(66, 67, 68),
        )

        self.assertEqual(len(jobs), 6)
        self.assertEqual({job.arm for job in jobs}, {"ssma"})
        self.assertEqual({job.missing_rate for job in jobs}, {0.0, 0.7})
        self.assertEqual({job.seed for job in jobs}, {66, 67, 68})
        for job in jobs:
            command = build_command(
                job,
                python=Path("/env/bin/python"),
                repository=Path("/repo"),
                data_root=Path("/data/IEMOCAP"),
                mask_bank_root=Path("/tmp/banks"),
            )
            self.assertIn("--second-graph-aggregation", command)
            self.assertNotIn("mask_sequence_aff", command)

    def test_gate_contains_two_rates_two_seeds_and_two_arms(self):
        jobs = build_jobs("gate", Path("/tmp/results"))

        self.assertEqual(len(jobs), 8)
        self.assertEqual({job.missing_rate for job in jobs}, {0.0, 0.7})
        self.assertEqual({job.seed for job in jobs}, {66, 67})
        self.assertEqual({job.arm for job in jobs}, {"original", "full"})

    def test_formal_contains_eighty_unique_paired_jobs(self):
        jobs = build_jobs("formal", Path("/tmp/results"))
        keys = {(job.arm, job.missing_rate, job.seed) for job in jobs}

        self.assertEqual(len(jobs), 80)
        self.assertEqual(len(keys), 80)

    def test_custom_ablation_grid_preserves_requested_labels(self):
        jobs = build_jobs(
            "formal",
            Path("/tmp/results"),
            arms=("pattern_only", "parameter_matched"),
            rates=(0.3, 0.5, 0.7),
            seeds=(66, 67, 68, 69, 70),
        )

        self.assertEqual(len(jobs), 30)
        self.assertEqual(
            {job.arm for job in jobs},
            {"pattern_only", "parameter_matched"},
        )

    def test_parameter_matched_label_maps_to_content_control(self):
        job = build_jobs(
            "formal",
            Path("/tmp/results"),
            arms=("parameter_matched",),
            rates=(0.5,),
            seeds=(66,),
        )[0]
        command = build_command(
            job,
            python=Path("/env/bin/python"),
            repository=Path("/repo"),
            data_root=Path("/data/IEMOCAP"),
            mask_bank_root=Path("/tmp/banks"),
        )

        variant_index = command.index("--graph-conv-variant") + 1
        self.assertEqual(command[variant_index], "content_film_control")

    def test_cp_lecc_label_maps_to_cp_lecc_graph_variant(self):
        job = build_jobs(
            "formal",
            Path("/tmp/results"),
            arms=("cp_lecc",),
            rates=(0.5,),
            seeds=(66,),
        )[0]
        command = build_command(
            job,
            python=Path("/env/bin/python"),
            repository=Path("/repo"),
            data_root=Path("/data/IEMOCAP"),
            mask_bank_root=Path("/tmp/banks"),
        )

        self.assertEqual(
            command[command.index("--graph-conv-variant") + 1], "cp_lecc"
        )

    def test_cp_lecc_gate_grid_has_complete_and_ten_paired_jobs(self):
        complete = build_jobs(
            "formal",
            Path("/tmp/results"),
            arms=("cp_lecc",),
            rates=(0.0,),
            seeds=(66,),
        )
        paired = build_jobs(
            "formal",
            Path("/tmp/results"),
            arms=("cp_lecc",),
            rates=(0.5, 0.7),
            seeds=(66, 67, 68, 69, 70),
        )

        jobs = complete + paired
        self.assertEqual(len(jobs), 11)
        self.assertEqual(
            [(job.missing_rate, job.seed) for job in jobs],
            [(0.0, 66)]
            + [(rate, seed) for rate in (0.5, 0.7) for seed in range(66, 71)],
        )

    def test_film_ab_labels_map_to_the_only_changed_operation(self):
        jobs = build_jobs(
            "formal",
            Path("/tmp/results"),
            arms=("linearized_film", "faithful_edgewise_film"),
            rates=(0.5,),
            seeds=(66,),
        )
        variants = []
        for job in jobs:
            command = build_command(
                job,
                python=Path("/env/bin/python"),
                repository=Path("/repo"),
                data_root=Path("/data/IEMOCAP"),
                mask_bank_root=Path("/tmp/banks"),
            )
            variants.append(command[command.index("--graph-conv-variant") + 1])

        self.assertEqual(variants, ["full", "faithful_edgewise"])

    def test_child_environment_requires_deterministic_cublas(self):
        source = Path(
            "experiments/mpfilm_iemocap6/run_locked_ab.py"
        ).read_text(encoding="utf-8")

        self.assertIn('CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"', source)
        self.assertIn('PYTHONHASHSEED"] = "0"', source)

    def test_command_locks_fold_and_omits_smoke_flag(self):
        job = build_jobs("gate", Path("/tmp/results"))[0]
        command = build_command(
            job,
            python=Path("/env/bin/python"),
            repository=Path("/repo"),
            data_root=Path("/data/IEMOCAP"),
            mask_bank_root=Path("/tmp/banks"),
        )

        joined = " ".join(command)
        self.assertIn("--fold-index 5", joined)
        self.assertIn("--epochs 100", joined)
        self.assertIn("--hidden 200", joined)
        self.assertIn("--num-threads 6", joined)
        self.assertNotIn("--allow-short-run", command)


class RunnerResumeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.job = build_jobs(
            "formal",
            Path(self.temporary.name),
            arms=("cp_lecc",),
            rates=(0.5,),
            seeds=(66,),
        )[0]
        self.inputs, self.payload = _runner_inputs(self.job)

    def test_valid_complete_job_is_skipped(self):
        _write_complete_job(self.job, self.payload)
        self.assertTrue(_completed(self.job, ("0",), **self.inputs))

    def test_invalid_complete_artifacts_fail_loudly(self):
        mutations = {
            "return_code": lambda: (self.job.output_directory / "status.json").write_text(
                json.dumps({"status": "success", "return_code": 1})
            ),
            "exactly one": lambda: (self.job.output_directory / "saved" / "extra.npz").write_bytes(b"x"),
            "command.json": lambda: (self.job.output_directory / "command.json").write_text("{}"),
            "100 epoch": lambda: (self.job.output_directory / "train.log").write_text("epoch:1\n"),
        }
        for message, mutate in mutations.items():
            with self.subTest(message=message):
                if self.job.output_directory.exists():
                    import shutil
                    shutil.rmtree(self.job.output_directory)
                _write_complete_job(self.job, self.payload)
                mutate()
                with self.assertRaisesRegex(RuntimeError, message):
                    _completed(self.job, ("0",), **self.inputs)

    def test_zero_archive_and_partial_directory_fail_loudly(self):
        _write_complete_job(self.job, self.payload)
        (self.job.output_directory / "saved" / "run.npz").unlink()
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            _completed(self.job, ("0",), **self.inputs)

        import shutil
        shutil.rmtree(self.job.output_directory)
        self.job.output_directory.mkdir(parents=True)
        (self.job.output_directory / "partial.txt").write_text("partial")
        with self.assertRaisesRegex(RuntimeError, "partial"):
            _completed(self.job, ("0",), **self.inputs)

    def test_existing_lock_is_rejected_and_claim_release_is_atomic(self):
        self.job.output_directory.mkdir(parents=True)
        lock = _claim_job(self.job)
        self.assertTrue(lock.exists())
        with self.assertRaisesRegex(RuntimeError, "lock"):
            _claim_job(self.job)
        _release_job(self.job)
        self.assertFalse(lock.exists())

    def test_launch_never_truncates_an_existing_log(self):
        self.job.output_directory.mkdir(parents=True)
        log = self.job.output_directory / "train.log"
        log.write_text("keep-me")
        with self.assertRaises(FileExistsError):
            _launch(self.job, "0", **self.inputs)
        self.assertEqual(log.read_text(), "keep-me")
        self.assertFalse((self.job.output_directory / ".active.lock").exists())

    def test_first_wave_paths_are_a_subset_of_expanded_formal_and_completed_skip(self):
        root = Path(self.temporary.name)
        first_wave = build_jobs(
            "formal",
            root,
            arms=("genagg", "soft_medoid"),
            rates=(0.0, 0.7),
            seeds=(66, 67, 68),
        )
        expanded = build_jobs(
            "formal",
            root,
            arms=("genagg", "soft_medoid"),
            rates=(0.0, 0.7),
            seeds=(66, 67, 68, 69, 70),
        )
        self.assertTrue(
            {job.output_directory for job in first_wave}
            < {job.output_directory for job in expanded}
        )

        values, payload = _runner_inputs(first_wave[0])
        _write_complete_job(first_wave[0], payload)
        launched = []

        class Process:
            def poll(self):
                return 0

        class Handle:
            def close(self):
                pass

        def launch(job, gpu, *args):
            launched.append(job)
            return Process(), Handle()

        with mock.patch(
            "experiments.mpfilm_iemocap6.run_locked_ab._launch", side_effect=launch
        ), mock.patch("experiments.mpfilm_iemocap6.run_locked_ab.time.sleep"):
            run_jobs(first_wave, ("0",), 12, **values)

        self.assertEqual(len(launched), 11)
        self.assertNotIn(first_wave[0], launched)

    def test_gate_artifacts_cannot_complete_a_formal_job(self):
        root = Path(self.temporary.name)
        gate_job = build_jobs(
            "gate", root, arms=("genagg",), rates=(0.7,), seeds=(66,)
        )[0]
        formal_job = build_jobs(
            "formal", root, arms=("genagg",), rates=(0.7,), seeds=(66,)
        )[0]
        values, gate_payload = _runner_inputs(gate_job)
        _write_complete_job(gate_job, gate_payload)

        self.assertNotEqual(gate_job.output_directory, formal_job.output_directory)
        self.assertEqual(gate_payload["stage"], "gate")
        self.assertFalse(_completed(formal_job, ("0",), **values))


class ParallelArmTests(unittest.TestCase):
    def _run_main(self, parallel):
        arguments = [
            "run_locked_ab.py",
            "--stage",
            "formal",
            "--output-root",
            "/out",
            "--data-root",
            "/data",
            "--mask-bank-root",
            "/masks",
            "--arms",
            "genagg",
            "soft_medoid",
            "--rates",
            "0.0",
            "0.7",
            "--seeds",
            "66",
            "67",
            "68",
        ]
        if parallel:
            arguments.append("--parallel-arms")
        with mock.patch.object(sys, "argv", arguments), mock.patch(
            "experiments.mpfilm_iemocap6.run_locked_ab._ensure_run_manifest"
        ) as manifest, mock.patch(
            "experiments.mpfilm_iemocap6.run_locked_ab.run_jobs"
        ) as runner:
            main()
        return manifest, runner

    def test_parallel_arms_submits_all_twelve_jobs_once(self):
        manifest, runner = self._run_main(parallel=True)

        self.assertEqual(runner.call_count, 1)
        self.assertEqual(len(list(runner.call_args.args[0])), 12)
        self.assertTrue(manifest.call_args.kwargs["parallel_arms"])

    def test_default_schedules_each_arm_sequentially(self):
        manifest, runner = self._run_main(parallel=False)

        self.assertEqual(runner.call_count, 2)
        self.assertEqual(
            [len(list(call.args[0])) for call in runner.call_args_list], [6, 6]
        )
        self.assertFalse(manifest.call_args.kwargs["parallel_arms"])

    def test_four_gpus_with_three_workers_launch_twelve_at_once(self):
        jobs = build_jobs(
            "formal",
            Path("/tmp/results"),
            arms=("genagg", "soft_medoid"),
            rates=(0.0, 0.7),
            seeds=(66, 67, 68),
        )
        assignments = []

        class Process:
            def poll(self):
                return 0

        class Handle:
            def close(self):
                pass

        def launch(job, gpu, *args):
            assignments.append(gpu)
            return Process(), Handle()

        with mock.patch(
            "experiments.mpfilm_iemocap6.run_locked_ab._completed", return_value=False
        ), mock.patch(
            "experiments.mpfilm_iemocap6.run_locked_ab._launch", side_effect=launch
        ), mock.patch("experiments.mpfilm_iemocap6.run_locked_ab.time.sleep"):
            run_jobs(
                jobs,
                ("0", "1", "2", "3"),
                3,
                Path("/python"),
                Path("/repo"),
                Path("/data"),
                Path("/masks"),
            )

        self.assertEqual(assignments, ["0", "1", "2", "3"] * 3)


class RunManifestTests(unittest.TestCase):
    def _manifest(
        self,
        root,
        head="abc",
        dirty=False,
        rates=(0.5, 0.7),
        seeds=(66, 67),
        python=Path(sys.executable),
        parallel_arms=False,
    ):
        git_status = " M dirty.py\n" if dirty else ""

        def command(args, **kwargs):
            if args[:2] == ["git", "rev-parse"]:
                return head + "\n"
            if args[:2] == ["git", "status"]:
                return git_status
            if args[0] == "nvidia-smi":
                return "GPU A\n"
            if args[0] == str(python):
                python_kwargs = dict(kwargs)
                python_kwargs.pop("cwd", None)
                return subprocess.check_output(args, **python_kwargs)
            raise AssertionError(args)

        return _ensure_run_manifest(
            output_root=root,
            stage="formal",
            repository=Path("/repo"),
            data_root=Path("/data"),
            mask_bank_root=Path("/masks"),
            arms=("cp_lecc",),
            rates=rates,
            seeds=seeds,
            gpus=("0",),
            workers_per_gpu=1,
            python=python,
            command_output=command,
            parallel_arms=parallel_arms,
        )

    def test_manifest_schema_and_identical_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = self._manifest(root)
            manifest_path = root / "formal" / "run_manifest.json"
            manifest_path.write_text(json.dumps(first))
            second = self._manifest(root)
            self.assertEqual(first, second)
            self.assertEqual(first["git"], {"head": "abc", "clean": True})
            self.assertEqual(first["environment"]["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
            self.assertEqual(first["environment"]["PYTHONHASHSEED"], "0")
            self.assertEqual(first["locked_training"]["hidden"], 200)
            self.assertEqual(first["gpu_names"], ["GPU A"])

    def test_manifest_mismatch_and_dirty_worktree_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._manifest(root)
            with self.assertRaisesRegex(RuntimeError, "manifest mismatch"):
                self._manifest(root, head="different")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "clean git worktree"):
                self._manifest(Path(tmp), dirty=True)

    def test_complete_then_missing_invocations_share_one_immutable_run_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shared = self._manifest(root, rates=(0.0,), seeds=(66,))
            repeated = self._manifest(
                root, rates=(0.5, 0.7), seeds=(66, 67, 68, 69, 70)
            )

            self.assertEqual(shared, repeated)
            invocations = sorted((root / "formal" / "invocations").glob("*.json"))
            self.assertEqual(len(invocations), 2)
            grids = [json.loads(path.read_text()) for path in invocations]
            self.assertEqual(
                {tuple(grid["rates"]) for grid in grids},
                {(0.0,), (0.5, 0.7)},
            )

    def test_parallel_arms_is_recorded_only_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy_manifest = self._manifest(root)
            legacy_invocation = next(
                (root / "formal" / "invocations").glob("*.json")
            )
            self.assertNotIn("parallel_arms", json.loads(legacy_invocation.read_text()))

            parallel_manifest = self._manifest(root, parallel_arms=True)
            self.assertEqual(legacy_manifest, parallel_manifest)
            invocations = [
                json.loads(path.read_text())
                for path in (root / "formal" / "invocations").glob("*.json")
            ]
            self.assertEqual(len(invocations), 2)
            self.assertEqual(
                sum(item.get("parallel_arms") is True for item in invocations), 1
            )

    def test_python_provenance_comes_from_requested_existing_interpreter(self):
        alternate = Path("/usr/bin/python3")
        if not alternate.exists():
            self.skipTest("/usr/bin/python3 is unavailable")
        expected = json.loads(
            subprocess.check_output(
                [
                    str(alternate),
                    "-c",
                    "import json,platform,sys; print(json.dumps({'executable':sys.executable,'version':platform.python_version()}))",
                ],
                text=True,
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self._manifest(Path(tmp), python=alternate)

        self.assertEqual(manifest["python"]["executable"], expected["executable"])
        self.assertEqual(manifest["python"]["version"], expected["version"])
        self.assertEqual(manifest["python"]["requested"], str(alternate))


if __name__ == "__main__":
    unittest.main()
