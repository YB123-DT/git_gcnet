import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from versions.mpfilm.run_locked_ab import (
    _claim_job,
    _completed,
    _ensure_run_manifest,
    _job_payload,
    _launch,
    _release_job,
    build_command,
    build_jobs,
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
            "versions/mpfilm/run_locked_ab.py"
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


class RunManifestTests(unittest.TestCase):
    def _manifest(
        self,
        root,
        head="abc",
        dirty=False,
        rates=(0.5, 0.7),
        seeds=(66, 67),
        python=Path(sys.executable),
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
