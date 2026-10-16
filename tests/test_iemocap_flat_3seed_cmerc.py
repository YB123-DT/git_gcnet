"""Finite Flat67/68 queue and evidence-only reuse of completed Flat66."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import tempfile
import unittest

from tests.test_iemocap_flat_seed66_restore import sources as original_sources, complete

MODULE = "experiments.osram_iemocap_flat_3seed_cmerc_20261016.run"
PYTHON = "/data2/yb/conda_envs/cmerc-repro/bin/python"
TORCH = "1.13.1+cu117"


def sources(seeds=(66,)):
    items = {}
    for seed in seeds:
        for (classes, _, fold), old in original_sources().items():
            item = deepcopy(old)
            item["config"]["seed"] = seed
            items[classes, seed, fold] = item
    return items


def result(job, sha, acc=70.):
    record = complete(job, acc)
    record["summary"]["seed"] = job["config"]["seed"]
    record["summary"]["identity"] = dict(source_git_sha=sha)
    record["summary"]["provenance"]["environment"] = dict(torch=TORCH, numpy="1.26.4")
    if job["config"]["seed"] != 66:
        record["summary"]["provenance"]["replication_environment"] = dict(
            executable=PYTHON, torch=TORCH, cuda="11.7", numpy="1.26.4", sklearn="1.3.2")
    return record


class ThreeSeedTests(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module(MODULE)
        except ModuleNotFoundError:
            self.fail("cmerc Flat three-seed completion launcher is missing")

    def test_original_builder_supports_seed_subset_without_changing_default(self):
        old = importlib.import_module("experiments.osram_iemocap_flat_seed66_restore_20261016.run")
        self.assertEqual(len(old.make_jobs(sources(), "/output")), 10)
        try:
            jobs = old.make_jobs(sources((67, 68)), "/output", seeds=(67, 68))
        except TypeError:
            self.fail("original builder has no seed-subset support")
        self.assertEqual(len(jobs), 20)
        self.assertEqual({j["config"]["seed"] for j in jobs}, {67, 68})
        self.assertTrue(all(j["config"]["osram_meaningful_block"] == "none" for j in jobs))
        self.assertEqual(len({j["output"] for j in jobs}), 20)
        for seeds in ((), (67, 67), (69,)):
            with self.assertRaises(ValueError):
                old.make_jobs({}, "/output", seeds=seeds)

    def test_original_loader_accepts_exact_requested_seed_subset(self):
        old = importlib.import_module("experiments.osram_iemocap_flat_seed66_restore_20261016.run")
        with tempfile.TemporaryDirectory() as root:
            for (classes, seed, fold), row in sources((67, 68)).items():
                path = Path(root) / f"iemocap{classes}/seed_{seed}/fold_{fold}/config.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(row["config"]))
            try:
                loaded = old.load_sources(root, seeds=(67, 68))
            except TypeError:
                self.fail("original source loader has no seed-subset support")
            self.assertEqual(set(loaded), set(sources((67, 68))))

    def test_runtime_guard_requires_resolved_cmerc_python_and_exact_torch(self):
        module = self.module()
        self.assertEqual(module.require_environment(PYTHON, TORCH)["torch"], TORCH)
        for executable, version in (("/wrong/python", TORCH), (PYTHON, "2.0.0")):
            with self.assertRaises(ValueError):
                module.require_environment(executable, version)

    def reference(self, module, root):
        jobs = module.flat.make_jobs(sources(), root)
        manifest = dict(provenance=dict(training_source_git_sha=module.TRAINING_SHA,
            environment=dict(executable=PYTHON)))
        Path(root).mkdir(parents=True, exist_ok=True)
        (Path(root) / "MANIFEST.json").write_text(json.dumps(manifest))
        for job in jobs:
            record = result(job, module.TRAINING_SHA)
            (Path(job["output"]) / "SUMMARY.json").write_text(json.dumps(record["summary"]))
        return jobs

    def test_reused_seed66_requires_all_ten_exact_results_source_and_environment(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as root:
            jobs = self.reference(module, root)
            records, evidence = module.load_seed66(jobs, root)
            self.assertEqual(len(records), 10)
            self.assertTrue(all(r["reused"] for r in records))
            self.assertEqual(len(evidence["summaries"]), 10)
            path = Path(jobs[0]["output"]) / "SUMMARY.json"
            original = path.read_text()
            for change in ("source", "torch", "method", "config"):
                bad = json.loads(original)
                if change == "source":
                    bad["identity"]["source_git_sha"] = "wrong"
                elif change == "torch":
                    bad["provenance"]["environment"]["torch"] = "wrong"
                elif change == "method":
                    bad["method"] = "B6_tuned_pca_only"
                else:
                    bad["config"]["seed"] = 67
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    module.load_seed66(jobs, root)
            path.write_text(original)
            manifest = Path(root) / "MANIFEST.json"
            data = json.loads(manifest.read_text())
            data["provenance"]["environment"]["executable"] = "/wrong/python"
            manifest.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                module.load_seed66(jobs, root)

    def test_only_twenty_new_jobs_and_replication_flag_with_unchanged_gpu_limits(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as root:
            frozen = importlib.import_module("experiments.osram_meld_block2_auto_20261014.scheduler")
            scheduler = module.make_scheduler(frozen, root, "/frozen", "/deployed/worker.py", 3590.)
            jobs = module.flat.make_jobs(sources((67, 68)), root, seeds=(67, 68))
            command = scheduler.command(jobs[0], "/config", "GPU-test-1")
            self.assertIn("--flat-only", command)
            self.assertIn("--flat-replication", command)
            self.assertEqual(scheduler.capacity, {1: 4, 3: 2})
            self.assertEqual(scheduler.gpus, (1, 3))
            self.assertEqual({j["config"]["seed"] for j in jobs}, {67, 68})

    def test_worker_replication_environment_schema_is_accepted_and_wrong_runtime_rejected(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as root:
            frozen = importlib.import_module("experiments.osram_meld_block2_auto_20261014.scheduler")
            scheduler = module.make_scheduler(frozen, root, "/frozen", "/deployed/worker.py", 3590.)
            job = module.flat.make_jobs(sources((67, 68)), root, seeds=(67, 68))[0]
            summary = result(job, module.TRAINING_SHA)["summary"]
            path = Path(job["output"]) / "SUMMARY.json"
            path.write_text(json.dumps(summary))
            self.assertIsNotNone(scheduler.summary(job))
            summary["provenance"]["replication_environment"]["executable"] = "/wrong/python"
            path.write_text(json.dumps(summary))
            self.assertIsNone(scheduler.summary(job))

    def test_joins_reused_ten_and_new_twenty_without_partial_three_seed_averages(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs66 = self.reference(module, root / "old")
            reused, _ = module.load_seed66(jobs66, root / "old")
            jobs = module.flat.make_jobs(sources((67, 68)), root / "new", seeds=(67, 68))
            records = [result(j, module.TRAINING_SHA, acc=j["config"]["seed"] + j["config"]["fold"]) for j in jobs]
            records[-1]["status"] = "failed"
            report = module.summarize(records, jobs, reused, root)
            self.assertEqual(report["completed_fold_runs"], 29)
            self.assertIsNone(report["datasets"]["IEMOCAPFour"]["three_seed"])
            self.assertIsNone(report["datasets"]["IEMOCAPSix"]["three_seed"])
            records[-1]["status"] = "complete"
            report = module.summarize(records, jobs, reused, root)
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["completed_fold_runs"], 30)
            self.assertEqual(report["completed_new_fold_runs"], 20)
            average = report["datasets"]["IEMOCAPFour"]["three_seed"]["mean"]["ACC"]["mean"]
            self.assertAlmostEqual(average, (70 + 70 + 71) / 3)
            self.assertEqual(report["reused_fold_runs"], 10)


if __name__ == "__main__":
    unittest.main()
