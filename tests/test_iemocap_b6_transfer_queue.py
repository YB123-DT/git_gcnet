"""CPU-only contracts for the finite original-protocol IEMOCAP B6 transfer."""
import importlib
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch


MODULE = "experiments.osram_iemocap_b6_transfer_20261015.run"


def baseline(dataset="IEMOCAPFour", seed=66, fold=5):
    return dict(dataset=dataset, seed=seed, fold=fold, epochs=100, batch_size=32,
                optimizer="adam", learning_rate=.001, weight_decay=1e-5, dropout=.5,
                train_rate_mode="cyclic", train_missing_rates=[i / 10 for i in range(8)],
                missing_mask_protocol="gcnet-random-mask-v1", training_objective="emotion-only",
                checkpoint_selection="test-oracle-per-rate", evaluate_test=True,
                osram_meaningful_block="none", osram_disable_local_skip=False,
                osram_readout_fusion="flat", osram_output_dim=1600, backbone_type="osram",
                device="cuda", preserve_selection_artifacts=False,
                feature_path="/original/IEMOCAP/features", arbitrary_preserved_setting=123)


def sources():
    return {(classes, seed, fold): dict(config=baseline(
                "IEMOCAPFour" if classes == 4 else "IEMOCAPSix", seed, fold),
                config_path=f"/original/iemocap{classes}/seed_{seed}/fold_{fold}/config.json",
                config_sha256=f"source-{classes}-{seed}-{fold}")
            for classes in (4, 6) for seed in (66, 67, 68) for fold in (1, 2, 3, 4, 5)}


def completed(job, acc=70.):
    metrics = dict(ACC=acc, UA=acc - 2, WF1=acc - 1, CE=.5)
    summary = dict(status="complete", dataset=job["config"]["dataset"],
                   seed=job["config"]["seed"], fold=job["config"]["fold"],
                   config=job["config"], provenance={"source_git_sha": "frozen"},
                   testbest=dict(rates={f"{i / 10:.1f}": dict(metrics) for i in range(8)},
                                 mean=dict(metrics), high=dict(metrics)))
    return dict(job, status="complete", summary=summary)


class TransferQueueTests(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module(MODULE)
        except ModuleNotFoundError:
            self.fail("bounded IEMOCAP B6 transfer launcher is missing")

    def test_exact_thirty_jobs_fold5_first_and_only_allowed_configuration_changes(self):
        module = self.module()
        original = sources()
        jobs = module.make_jobs(original, "/new/output")
        self.assertEqual(len(jobs), 30)
        self.assertEqual([j["config"]["fold"] for j in jobs[:6]], [5] * 6)
        self.assertEqual([j["config"]["fold"] for j in jobs[6:12]], [1] * 6)
        self.assertEqual(len({j["output"] for j in jobs}), 30)
        self.assertEqual(len({j["id"] for j in jobs}), 30)
        for job in jobs:
            config = job["config"]
            key = (4 if config["dataset"] == "IEMOCAPFour" else 6,
                   config["seed"], config["fold"])
            old = original[key]["config"]
            changed = {k for k in set(old) | set(config) if old.get(k) != config.get(k)}
            self.assertEqual(changed, {"osram_meaningful_block", "osram_rpca_rank",
                "osram_rpca_steps", "osram_rpca_threshold_init", "osram_rpca_lr_multiplier",
                "preserve_selection_artifacts", "checkpoint_selection_metric", "device"})
            self.assertEqual(config["learning_rate"], .001)
            self.assertEqual(config["dropout"], .5)
            self.assertEqual(config["weight_decay"], 1e-5)
            self.assertEqual(config["osram_rpca_rank"], 2)
            self.assertEqual(config["osram_rpca_steps"], 14)
            self.assertEqual(config["osram_rpca_threshold_init"], .33599569865951234)
            self.assertEqual(config["osram_rpca_lr_multiplier"], 2.8200380294401013)
            self.assertEqual(job["baseline"]["config"], old)
            self.assertEqual(set(job["configuration_diff"]), changed)
        self.assertEqual(original, sources())

    def test_rejects_unapproved_source_protocol_or_incomplete_scope(self):
        module = self.module()
        changes = dict(learning_rate=.0003433234536467373, dropout=.22489,
                       weight_decay=4.485e-6, epochs=50, batch_size=16,
                       osram_disable_local_skip=True, train_rate_mode="fixed",
                       checkpoint_selection_metric="weighted_f1", training_objective="joint",
                       osram_meaningful_block="other", seed=99,
                       osram_readout_fusion="other", osram_output_dim=800)
        for name, value in changes.items():
            with self.subTest(field=name):
                configs = sources()
                configs[4, 66, 5]["config"][name] = value
                with self.assertRaises(ValueError):
                    module.make_jobs(configs, "/output")
        configs = sources()
        configs.pop((4, 66, 5))
        with self.assertRaises(ValueError):
            module.make_jobs(configs, "/output")

    def test_loads_every_original_config_with_hash_and_absolute_path(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for (classes, seed, fold), source in sources().items():
                path = root / f"iemocap{classes}/seed_{seed}/fold_{fold}/config.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(source["config"]))
            loaded = module.load_sources(root)
            self.assertEqual(set(loaded), set(sources()))
            for source in loaded.values():
                self.assertEqual(len(source["config_sha256"]), 64)
                self.assertTrue(Path(source["config_path"]).is_absolute())

    def test_smoke_requires_real_complete_compatible_finite_peak(self):
        module = self.module()
        jobs = module.make_jobs(sources(), "/output")
        smoke = dict(status="complete", smoke_only=True, peak_memory_mib=1500.,
                     config=jobs[0]["config"])
        self.assertEqual(module.smoke_peak(smoke, jobs), 1500.)
        for change in (dict(status="failed"), dict(smoke_only=False),
                       dict(peak_memory_mib=float("nan")), dict(peak_memory_mib=0),
                       dict(config=baseline())):
            with self.assertRaises(ValueError):
                module.smoke_peak(dict(smoke, **change), jobs)

    def scheduler(self, root, peak=100.):
        module = self.module()
        frozen = importlib.import_module("experiments.osram_meld_block2_auto_20261014.scheduler")
        scheduler = module.make_scheduler(frozen, root, "/frozen", "/deployed/worker.py", peak)
        scheduler.gpu_snapshot = lambda: {
            g: dict(uuid=f"GPU-healthy-{g}", free_mib=100000., compute_pids=[])
            for g in (1, 3, 4, 5)}
        scheduler.disk_margin_bytes = 0
        scheduler.smoke_costs["B6"]["disk_bytes"] = 0
        return scheduler

    def test_new_absolute_worker_and_only_physical_gpu1_and3(self):
        with tempfile.TemporaryDirectory() as root:
            scheduler = self.scheduler(root)
            job = self.module().make_jobs(sources(), root)[0]
            command = scheduler.command(job, Path(root) / "config.json", "GPU-healthy-1")
            self.assertEqual(command[2], "/deployed/worker.py")
            self.assertIn("--source-root", command)
            self.assertEqual(command[command.index("--source-root") + 1], "/frozen")
            self.assertEqual(scheduler.gpus, (1, 3))
            self.assertEqual(scheduler.capacity, {1: 8, 3: 2})

    def test_gpu1_total_process_limit_gpu3_owned_limit_and_peak_reservations(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as root:
            scheduler = self.scheduler(root)
            job = module.make_jobs(sources(), root)[0]
            snapshot = scheduler.gpu_snapshot()
            snapshot[1]["compute_pids"] = list(range(8))
            scheduler.gpu_snapshot = lambda: snapshot
            self.assertEqual(scheduler.admit(job), (3, "GPU-healthy-3"))
            scheduler.state["jobs"] = {str(i): dict(job, status="running", gpu=3, pid=100 + i)
                                       for i in range(2)}
            self.assertIsNone(scheduler.admit(job))
            snapshot[1]["compute_pids"] = []
            snapshot[1]["free_mib"] = 1123.
            self.assertIsNone(scheduler.admit(job))
            snapshot[1]["free_mib"] = 1124.
            self.assertEqual(scheduler.admit(job), (1, "GPU-healthy-1"))
            scheduler.state["gpu_uuids"]["1"] = "GPU-changed"
            with self.assertRaisesRegex(RuntimeError, "UUID"):
                scheduler.admit(job)

    def test_summary_keeps_fold5_distinct_and_failed_jobs_out_of_fivefold_mean(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as root:
            jobs = module.make_jobs(sources(), root)
            records = [completed(j, acc=70 + j["config"]["fold"]) for j in jobs[:6]]
            # Real worker rows contain selection provenance and class breakdowns.
            for record in records:
                for row in record['summary']['testbest']['rates'].values():
                    row.update(epoch=17, recall=[1, 2, 3, 4], support=[3, 2, 1, 1],
                               checkpoint_sha256='checked', mask_sha256='checked')
            report = module.summarize(records, jobs, root)
            self.assertEqual(report["status"], "incomplete_or_failed")
            self.assertEqual(report["completed_fold_runs"], 6)
            self.assertEqual(report["datasets"]["IEMOCAPFour"]["fold5"]["three_seed"]["mean"]["ACC"]["mean"], 75.)
            self.assertIsNone(report["datasets"]["IEMOCAPFour"]["fivefold"])
            self.assertTrue((Path(root) / "FOLD5.json").exists())
            records = [completed(j, acc=70 + j["config"]["fold"]) for j in jobs]
            records[-1]["status"] = "failed"
            report = module.summarize(records, jobs, root)
            self.assertEqual(report["completed_fold_runs"], 29)
            self.assertIsNone(report["datasets"]["IEMOCAPSix"]["fivefold"])
            records[-1]["status"] = "complete"
            report = module.summarize(records, jobs, root)
            self.assertEqual(report["status"], "complete")
            aggregate = report["datasets"]["IEMOCAPSix"]["fivefold"]
            self.assertEqual(aggregate["three_seed"]["mean"]["ACC"]["mean"], 73.)
            self.assertEqual(report["baseline_exact_testbest_UA"], "unavailable; old exact selection-epoch artifacts not established")

    def test_summary_rejects_wrong_config_identity_or_nonfinite_metrics(self):
        module = self.module()
        job = module.make_jobs(sources(), "/output")[0]
        self.assertTrue(module.valid_summary(completed(job)["summary"], job))
        for change in (dict(fold=1), dict(config=baseline()), dict(status="failed")):
            self.assertFalse(module.valid_summary(dict(completed(job)["summary"], **change), job))
        summary = completed(job)["summary"]
        summary["testbest"]["rates"]["0.0"]["UA"] = float("nan")
        self.assertFalse(module.valid_summary(summary, job))

    def test_adopts_existing_complete_exact_output_without_duplicate_launch(self):
        with tempfile.TemporaryDirectory() as root:
            scheduler = self.scheduler(root)
            job = self.module().make_jobs(sources(), root)[0]
            output = Path(job["output"])
            output.mkdir(parents=True)
            (output / "SUMMARY.json").write_text(json.dumps(completed(job)["summary"]))
            with patch.object(scheduler, "command", side_effect=AssertionError("duplicate launch")):
                result = scheduler.run_jobs([job])[0]
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["attempts"], 0)

    def test_real_cpu_workers_complete_finite_queue_and_resume_reuses_all_jobs(self):
        with tempfile.TemporaryDirectory() as root:
            scheduler = self.scheduler(root)
            scheduler.poll_seconds = .01
            scheduler.code_root = Path(root)
            jobs = self.module().make_jobs(sources(), root)
            def command(job, config, uuid):
                script = ("import json,pathlib; p=pathlib.Path(" + repr(job["output"]) + "); "
                          "p.mkdir(parents=True,exist_ok=True); "
                          "(p/'SUMMARY.json').write_text(" + repr(json.dumps(completed(job)["summary"])) + ")")
                return [sys.executable, "-c", script, str(config), uuid]
            scheduler.command = command
            records = scheduler.run_jobs(jobs)
            self.assertEqual(len(records), 30)
            self.assertTrue(all(r["status"] == "complete" for r in records))
            self.assertTrue(all(r["gpu"] in (1, 3) for r in records))
            self.assertEqual(set(json.loads((Path(root) / "QUEUE.json").read_text())["jobs"]),
                             {job["id"] for job in jobs})
            with patch.object(scheduler, "command", side_effect=AssertionError("duplicate launch")):
                reused = scheduler.run_jobs(jobs)
            self.assertEqual([r["attempts"] for r in reused], [1] * 30)


if __name__ == "__main__":
    unittest.main()
