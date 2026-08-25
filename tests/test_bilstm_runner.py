import ast
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class BiLSTMRunnerContractTests(unittest.TestCase):
    def setUp(self):
        from experiments.bilstm_ablation import run_factorial

        self.runner = run_factorial

    def test_registry_and_factorial_arm_contract(self):
        self.assertEqual(
            self.runner.DATASETS,
            {
                "IEMOCAPFour": {
                    "directory": "IEMOCAP",
                    "fold": 5,
                    "split_tag": "fold5_screening",
                    "metric": "multiclass_weighted_f1",
                },
                "IEMOCAPSix": {
                    "directory": "IEMOCAP",
                    "fold": 5,
                    "split_tag": "fold5_screening",
                    "metric": "multiclass_weighted_f1",
                },
                "CMUMOSI": {
                    "directory": "CMUMOSI",
                    "fold": None,
                    "split_tag": "official_split",
                    "metric": "nonzero_binary_weighted_f1",
                },
                "CMUMOSEI": {
                    "directory": "CMUMOSEI",
                    "fold": None,
                    "split_tag": "official_split",
                    "metric": "nonzero_binary_weighted_f1",
                },
            },
        )
        self.assertEqual(
            self.runner.ARMS,
            {
                "original": ("bilstm", "bilstm"),
                "no_pre_bilstm": ("linear", "bilstm"),
                "no_post_bilstm": ("bilstm", "linear"),
                "no_all_bilstm": ("linear", "linear"),
            },
        )

    def test_stage_grids_have_exact_unique_identities(self):
        expected = {"smoke": 16, "pilot": 96, "formal": 640}
        for stage, count in expected.items():
            with self.subTest(stage=stage):
                jobs = self.runner.build_jobs(stage, Path("/results"))
                identities = {
                    (j.dataset, j.arm, j.missing_rate, j.seed, j.split_tag)
                    for j in jobs
                }
                self.assertEqual(len(jobs), count)
                self.assertEqual(len(identities), count)

    def test_custom_grid_keeps_requested_identity(self):
        jobs = self.runner.build_jobs(
            "formal",
            Path("/results"),
            datasets=("CMUMOSI",),
            arms=("no_pre_bilstm",),
            rates=(0.3,),
            seeds=(70,),
        )
        self.assertEqual(len(jobs), 1)
        job = jobs[0]
        self.assertEqual(
            (job.stage, job.dataset, job.arm, job.missing_rate, job.seed),
            ("formal", "CMUMOSI", "no_pre_bilstm", 0.3, 70),
        )
        self.assertEqual(
            job.output_directory,
            Path("/results/formal/CMUMOSI/no_pre_bilstm/miss_0p3/seed_70/official_split"),
        )

    def test_commands_lock_hparams_contexts_roots_and_fold_behavior(self):
        inputs = dict(
            python=Path("/env/python"),
            repository=Path("/repo"),
            data_root_root=Path("/data"),
            mask_root=Path("/masks"),
        )
        iemocap, mosi = self.runner.build_jobs(
            "formal",
            Path("/results"),
            datasets=("IEMOCAPSix", "CMUMOSI"),
            arms=("no_pre_bilstm",),
            rates=(0.7,),
            seeds=(66,),
        )
        iemocap_command = self.runner.build_command(iemocap, **inputs)
        mosi_command = self.runner.build_command(mosi, **inputs)
        joined = " ".join(iemocap_command)
        for fragment in (
            "--data-root /data/IEMOCAP",
            "--mask-bank-root /masks/IEMOCAPSix/fold5_screening",
            "--audio-feature wav2vec-large-c-UTT",
            "--text-feature deberta-large-4-UTT",
            "--video-feature manet_UTT",
            "--graph-conv-variant original",
            "--pre-graph-context linear",
            "--post-graph-context bilstm",
            "--windowp 2",
            "--windowf 2",
            "--hidden 200",
            "--lr 0.001",
            "--dropout 0.5",
            "--batch-size 32",
            "--num-threads 6",
            "--epochs 100",
            "--fold-index 5",
            "--loss-recon",
        ):
            self.assertIn(fragment, joined)
        self.assertNotIn("--fold-index", mosi_command)
        self.assertIn("/masks/CMUMOSI/official_split", mosi_command)
        self.assertNotEqual(
            iemocap_command[iemocap_command.index("--mask-bank-root") + 1],
            mosi_command[mosi_command.index("--mask-bank-root") + 1],
        )

    def test_smoke_short_run_requires_both_explicit_options(self):
        job = self.runner.build_jobs("smoke", Path("/results"))[0]
        standard = self.runner.build_command(
            job, Path("/python"), Path("/repo"), Path("/data"), Path("/masks")
        )
        short = self.runner.build_command(
            job,
            Path("/python"),
            Path("/repo"),
            Path("/data"),
            Path("/masks"),
            epochs=2,
            allow_short_run=True,
        )
        self.assertEqual(standard[standard.index("--epochs") + 1], "100")
        self.assertNotIn("--allow-short-run", standard)
        self.assertEqual(short[short.index("--epochs") + 1], "2")
        self.assertIn("--allow-short-run", short)
        with self.assertRaisesRegex(ValueError, "allow-short-run"):
            self.runner.build_command(
                job,
                Path("/python"),
                Path("/repo"),
                Path("/data"),
                Path("/masks"),
                epochs=2,
            )

    def test_source_parses_as_python38_and_avoids_union_operator(self):
        source = Path("experiments/bilstm_ablation/run_factorial.py").read_text()
        ast.parse(source, feature_version=(3, 8))
        self.assertNotIn(" | None", source)


def _complete_log(epochs=100, smoke=False):
    return "\n".join(
        ["epoch:{}; train_fscore:0.0".format(i) for i in range(1, epochs + 1)]
        + [
            "SMOKE_ONLY={}".format(smoke),
            "Finish",
            "save results in archive.npz",
        ]
    )


class BiLSTMRunnerResumeTests(unittest.TestCase):
    def setUp(self):
        from experiments.bilstm_ablation import run_factorial

        self.runner = run_factorial
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.job = self.runner.build_jobs(
            "formal",
            Path(self.temporary.name),
            datasets=("CMUMOSI",),
            arms=("original",),
            rates=(0.7,),
            seeds=(66,),
        )[0]
        self.inputs = dict(
            python=Path("/python"),
            repository=Path("/repo"),
            data_root_root=Path("/data"),
            mask_root=Path("/masks"),
        )
        self.payload = self.runner._job_payload(self.job, "0", **self.inputs)

    def _write_complete(self):
        directory = self.job.output_directory
        directory.mkdir(parents=True)
        (directory / "command.json").write_text(json.dumps(self.payload))
        (directory / "status.json").write_text(
            json.dumps({"status": "success", "return_code": 0})
        )
        (directory / "train.log").write_text(_complete_log())
        (directory / "saved").mkdir()
        (directory / "saved/result.npz").write_bytes(b"npz")

    def test_valid_resume_and_command_drift_rejection(self):
        self._write_complete()
        self.assertTrue(self.runner._completed(self.job, ("0",), **self.inputs))
        self.payload["command"].append("--drift")
        (self.job.output_directory / "command.json").write_text(
            json.dumps(self.payload)
        )
        with self.assertRaisesRegex(RuntimeError, "command.json mismatch"):
            self.runner._completed(self.job, ("0",), **self.inputs)

    def test_partial_and_stale_lock_are_rejected(self):
        self.job.output_directory.mkdir(parents=True)
        (self.job.output_directory / "partial").write_text("x")
        with self.assertRaisesRegex(RuntimeError, "partial"):
            self.runner._completed(self.job, ("0",), **self.inputs)
        shutil.rmtree(self.job.output_directory)
        self.job.output_directory.mkdir(parents=True)
        (self.job.output_directory / ".active.lock").write_text("{}")
        with self.assertRaisesRegex(RuntimeError, "lock"):
            self.runner._completed(self.job, ("0",), **self.inputs)

    def test_claim_is_atomic_and_launch_never_truncates_log(self):
        self.job.output_directory.mkdir(parents=True)
        lock = self.runner._claim_job(self.job)
        with self.assertRaisesRegex(RuntimeError, "lock"):
            self.runner._claim_job(self.job)
        self.runner._release_job(self.job)
        self.assertFalse(lock.exists())
        log = self.job.output_directory / "train.log"
        log.write_text("keep")
        with self.assertRaises(FileExistsError):
            self.runner._launch(self.job, "0", **self.inputs)
        self.assertEqual(log.read_text(), "keep")


class BiLSTMManifestTests(unittest.TestCase):
    def setUp(self):
        from experiments.bilstm_ablation import run_factorial

        self.runner = run_factorial
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.data = root / "data"
        for dataset, entry in self.runner.DATASETS.items():
            dataset_root = self.data / entry["directory"]
            (dataset_root / "features").mkdir(parents=True, exist_ok=True)
            label = self.runner.LABEL_FILENAMES[dataset]
            (dataset_root / label).write_bytes((dataset + " labels").encode())
            for feature in self.runner.FEATURE_NAMES.values():
                feature_root = dataset_root / "features" / feature
                feature_root.mkdir(parents=True, exist_ok=True)
                (feature_root / "part.bin").write_bytes(
                    (dataset + feature).encode()
                )
        self.output = root / "output"

    def _command(self, args, **kwargs):
        if args[:2] == ["git", "status"]:
            return ""
        if args[:2] == ["git", "rev-parse"]:
            return "abc\n"
        if args[0] == "nvidia-smi":
            return "V100\n"
        if args[0] == sys.executable:
            python_kwargs = dict(kwargs)
            python_kwargs.pop("cwd", None)
            return subprocess.check_output(args, **python_kwargs)
        raise AssertionError(args)

    def _manifest(self, datasets=("IEMOCAPFour", "CMUMOSI")):
        return self.runner._ensure_run_manifest(
            output_root=self.output,
            stage="formal",
            repository=Path("/repo"),
            data_root_root=self.data,
            mask_root=Path("/masks"),
            datasets=datasets,
            arms=("original",),
            rates=(0.0,),
            seeds=(66,),
            gpus=("0",),
            workers_per_gpu=1,
            python=Path(sys.executable),
            command_output=self._command,
        )

    def test_manifest_records_dataset_split_metric_and_four_fingerprints(self):
        manifest = self._manifest()
        self.assertEqual(manifest["git"], {"head": "abc", "clean": True})
        for dataset in ("IEMOCAPFour", "CMUMOSI"):
            record = manifest["datasets"][dataset]
            self.assertEqual(record["split_tag"], self.runner.DATASETS[dataset]["split_tag"])
            self.assertEqual(record["metric"], self.runner.DATASETS[dataset]["metric"])
            self.assertEqual(
                set(record["fingerprints"]),
                {"label", "audio", "text", "video"},
            )
            for fingerprint in record["fingerprints"].values():
                self.assertRegex(fingerprint["sha256"], r"^[0-9a-f]{64}$")
                self.assertGreaterEqual(fingerprint["file_count"], 1)

    def test_manifest_detects_dataset_drift_and_dirty_git(self):
        self._manifest()
        changed = self.data / "CMUMOSI/features/manet_UTT/part.bin"
        changed.write_bytes(b"changed")
        with self.assertRaisesRegex(RuntimeError, "manifest mismatch"):
            self._manifest()

        def dirty(args, **kwargs):
            if args[:2] == ["git", "status"]:
                return " M dirty.py\n"
            return self._command(args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "clean git worktree"):
                self.runner._ensure_run_manifest(
                    Path(tmp), "formal", Path("/repo"), self.data,
                    Path("/masks"), ("CMUMOSI",), ("original",), (0.0,),
                    (66,), ("0",), 1, Path(sys.executable), dirty,
                )

    def test_invocations_are_distinct_but_share_immutable_run_manifest(self):
        first = self._manifest()
        second = self.runner._ensure_run_manifest(
            self.output, "formal", Path("/repo"), self.data, Path("/masks"),
            ("IEMOCAPFour", "CMUMOSI"), ("original",), (0.7,), (67, 68),
            ("0",), 1, Path(sys.executable), self._command,
        )
        self.assertEqual(first, second)
        self.assertEqual(len(list((self.output / "formal/invocations").glob("*.json"))), 2)

    def test_gpu_worker_cap_is_enforced(self):
        job = self.runner.build_jobs(
            "formal", self.output, datasets=("CMUMOSI",), arms=("original",),
            rates=(0.0,), seeds=(66,),
        )[0]
        with self.assertRaisesRegex(ValueError, "at most 3"):
            self.runner.run_jobs(
                (job,), ("0",), 4, Path("/python"), Path("/repo"),
                self.data, Path("/masks"),
            )


if __name__ == "__main__":
    unittest.main()
