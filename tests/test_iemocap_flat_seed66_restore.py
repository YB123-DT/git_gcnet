"""CPU-only contracts for the ten-job original Flat seed66 restoration."""
import importlib
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


MODULE = "experiments.osram_iemocap_flat_seed66_restore_20261016.run"


def sources():
    return {(classes, 66, fold): dict(config_path=f"/original/iemocap{classes}/seed_66/fold_{fold}/config.json",
        config_sha256=f"original-{classes}-{fold}", config=dict(
            dataset="IEMOCAPFour" if classes == 4 else "IEMOCAPSix", seed=66, fold=fold,
            epochs=100, batch_size=32, optimizer="adam", learning_rate=.001, weight_decay=1e-5,
            dropout=.5, training_objective="emotion-only", missing_mask_protocol="gcnet-random-mask-v1",
            train_rate_mode="cyclic", train_missing_rates=[i / 10 for i in range(8)],
            checkpoint_selection="test-oracle-per-rate", osram_meaningful_block="none",
            osram_readout_fusion="flat", osram_output_dim=1600, backbone_type="osram",
            osram_disable_local_skip=False, initial_backbone_checkpoint=None, evaluate_test=True,
            preserve_selection_artifacts=False, device="cuda", original_untouched_field="kept"))
        for classes in (4, 6) for fold in (1, 2, 3, 4, 5)}


def complete(job, acc=70.):
    metrics = dict(ACC=acc, UA=acc - 5, WF1=acc - 1, CE=.6)
    summary = dict(status="complete", method="Original_Flat", config=job["config"],
        dataset=job["config"]["dataset"], seed=66, fold=job["config"]["fold"], provenance={"source": "frozen"},
        testbest=dict(rates={str(i / 10): dict(metrics, support=[1, 2], epoch=13) for i in range(8)},
                      mean=dict(metrics), high=dict(metrics)))
    output = Path(job["output"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "FIRST_EPOCH_RETENTION_CHECK.json").write_text(json.dumps(dict(
        status="complete", best_checkpoints=8, selected_artifacts=8, cuda_rng_states=1)))
    return dict(job, status="complete", summary=summary)


class FlatRestoreTests(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module(MODULE)
        except ModuleNotFoundError:
            self.fail("Flat seed66 restoration launcher is missing")

    def test_exact_ten_original_flat_jobs_with_only_explicit_bookkeeping_changes(self):
        module = self.module()
        jobs = module.make_jobs(sources(), "/new")
        self.assertEqual(len(jobs), 10)
        self.assertEqual(len({j["output"] for j in jobs}), 10)
        self.assertEqual(len({j["id"] for j in jobs}), 10)
        self.assertEqual({j["config"]["seed"] for j in jobs}, {66})
        self.assertEqual({j["config"]["fold"] for j in jobs}, {1, 2, 3, 4, 5})
        for job in jobs:
            self.assertEqual(job["method"], "Flat")
            config = job["config"]
            self.assertEqual(config["osram_meaningful_block"], "none")
            self.assertEqual(config["osram_rpca_lr_multiplier"], 1.)
            self.assertEqual(set(job["configuration_diff"]), {"preserve_selection_artifacts",
                "checkpoint_selection_metric", "device", "osram_rpca_lr_multiplier"})
            self.assertEqual(job["config"]["original_untouched_field"], "kept")
            self.assertFalse(job["baseline"]["config"]["preserve_selection_artifacts"])

    def test_rejects_different_seeds_modules_protocols_or_old_weight_initialization(self):
        module = self.module()
        for key, value in dict(seed=67, epochs=50, learning_rate=.0003433,
            osram_meaningful_block="learned_robust_pca_evidence_decomposition",
            osram_readout_fusion="nested", training_objective="joint",
            osram_disable_local_skip=True, missing_mask_protocol="conversation-mask-v2",
            initial_backbone_checkpoint="/old/best.pt").items():
            with self.subTest(key=key):
                configs = sources()
                configs[4, 66, 5]["config"][key] = value
                with self.assertRaises(ValueError):
                    module.make_jobs(configs, "/new")
        configs = sources()
        configs[4, 67, 5] = configs.pop((4, 66, 5))
        with self.assertRaises(ValueError):
            module.make_jobs(configs, "/new")

    def test_load_sources_reads_only_original_seed66_ten_configs(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as root:
            for (classes, seed, fold), item in sources().items():
                path = Path(root) / f"iemocap{classes}/seed_{seed}/fold_{fold}/config.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(item["config"]))
            loaded = module.load_sources(root)
            self.assertEqual(set(loaded), set(sources()))
            self.assertTrue(all(len(row["config_sha256"]) == 64 for row in loaded.values()))

    def test_b6_smoke_bound_is_reused_without_requiring_flat_config(self):
        module = self.module()
        smoke = dict(status="complete", smoke_only=True, peak_memory_mib=3333.87,
                     peak_reserved_mib=3590., config=dict(dataset="IEMOCAPSix",
                     osram_meaningful_block="learned_robust_pca_evidence_decomposition"))
        self.assertEqual(module.smoke_peak(smoke), 3590.)
        for changes in (dict(status="failed"), dict(smoke_only=False), dict(peak_memory_mib=0),
                        dict(peak_memory_mib=float("nan"))):
            with self.assertRaises(ValueError):
                module.smoke_peak(dict(smoke, **changes))

    def scheduler(self, root):
        module = self.module()
        frozen = importlib.import_module("experiments.osram_meld_block2_auto_20261014.scheduler")
        scheduler = module.make_scheduler(frozen, root, "/frozen", "/deployment/worker.py", 3590.)
        scheduler.smoke_costs["Flat"]["disk_bytes"] = 0
        scheduler.disk_margin_bytes = 0
        scheduler.gpu_snapshot = lambda: {
            g: dict(uuid=f"GPU-healthy-{g}", free_mib=100000., compute_pids=[]) for g in (1, 3, 4, 5)}
        return scheduler

    def test_scheduler_caps_and_command_require_explicit_flat_flag(self):
        with tempfile.TemporaryDirectory() as root:
            scheduler = self.scheduler(root)
            job = self.module().make_jobs(sources(), root)[0]
            command = scheduler.command(job, Path(root) / "config.json", "GPU-healthy-1")
            self.assertEqual(command[2], "/deployment/worker.py")
            self.assertIn("--flat-only", command)
            self.assertEqual(scheduler.gpus, (1, 3))
            self.assertEqual(scheduler.capacity, {1: 4, 3: 2})
            for g, count in ((1, 4), (3, 2)):
                for i in range(count):
                    scheduler.state["jobs"][f"{g}-{i}"] = dict(job, status="running", gpu=g, pid=100 * g + i)
            self.assertIsNone(scheduler.admit(job))

    def test_acceptance_requires_flat_method_and_first_epoch_retention_file(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as root:
            job = module.make_jobs(sources(), root)[0]
            record = complete(job)
            self.assertTrue(module.valid_summary(record["summary"], job))
            record["summary"]["method"] = "B6_tuned_pca_only"
            self.assertFalse(module.valid_summary(record["summary"], job))
            record["summary"]["method"] = "Original_Flat"
            path = Path(job["output"]) / "FIRST_EPOCH_RETENTION_CHECK.json"
            path.write_text(json.dumps(dict(status="failed")))
            self.assertFalse(module.valid_summary(record["summary"], job))
            path.write_text(json.dumps(dict(status="complete", best_checkpoints=0,
                                            selected_artifacts=8, cuda_rng_states=1)))
            self.assertFalse(module.valid_summary(record["summary"], job))
            path.unlink()
            self.assertFalse(module.valid_summary(record["summary"], job))

    def test_fivefold_aggregate_only_after_five_valid_complete_folds(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as root:
            jobs = module.make_jobs(sources(), root)
            records = [complete(j, 70 + j["config"]["fold"]) for j in jobs]
            records[-1]["status"] = "failed"
            report = module.summarize(records, jobs, root)
            self.assertEqual(report["completed_fold_runs"], 9)
            self.assertIsNone(report["datasets"]["IEMOCAPSix"]["fivefold"])
            records[-1]["status"] = "complete"
            report = module.summarize(records, jobs, root)
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["datasets"]["IEMOCAPFour"]["fivefold"]["mean"]["ACC"], 73.)
            self.assertEqual(report["datasets"]["IEMOCAPFour"]["fivefold"]["rates"]["0.7"]["UA"], 68.)
            self.assertTrue((Path(root) / "SUMMARY.csv").exists())

    def test_real_cpu_queue_reuses_complete_outputs_and_never_restarts(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as root:
            scheduler = self.scheduler(root)
            scheduler.code_root = Path(root)
            scheduler.poll_seconds = .01
            jobs = module.make_jobs(sources(), root)
            def command(job, config, uuid):
                summary = complete(job)["summary"]
                script = ("import pathlib; p=pathlib.Path(" + repr(job["output"]) + "); "
                          "(p/'SUMMARY.json').write_text(" + repr(json.dumps(summary)) + ")")
                return [sys.executable, "-c", script, str(config), uuid]
            scheduler.command = command
            records = scheduler.run_jobs(jobs)
            self.assertEqual(len(records), 10)
            self.assertTrue(all(r["status"] == "complete" for r in records))
            with patch.object(scheduler, "command", side_effect=AssertionError("duplicate")):
                self.assertTrue(all(r["status"] == "complete" for r in scheduler.run_jobs(jobs)))

    def test_check_only_prepares_ten_configs_and_manifest_without_scheduler_or_gpu(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline_root, source = root / "baseline", root / "numerical"
            source.mkdir()
            for (classes, seed, fold), item in sources().items():
                path = baseline_root / f"iemocap{classes}/seed_{seed}/fold_{fold}/config.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(item["config"]))
            worker, smoke, data, output = (root / name for name in ("worker.py", "smoke.json", "data.json", "output"))
            worker.write_text("# deployed worker\n")
            smoke.write_text(json.dumps(dict(status="complete", smoke_only=True,
                peak_memory_mib=3333.87, peak_reserved_mib=3590., config=dict(dataset="IEMOCAPSix",
                osram_meaningful_block="learned_robust_pca_evidence_decomposition"))))
            data.write_text(json.dumps(dict(feature_roots=["/original/features"])))
            def git(command, cwd=None, **kwargs):
                if command[0] != "git":
                    return "cpu-test\n"
                return "" if "diff" in command else module.TRAINING_SHA + "\n"
            with patch.object(module.subprocess, "check_output", side_effect=git), \
                 patch.object(module, "make_scheduler", side_effect=AssertionError("GPU queue created")), \
                 redirect_stdout(io.StringIO()):
                module.main(["--source-root", str(source), "--baseline-root", str(baseline_root),
                    "--worker", str(worker), "--smoke-summary", str(smoke), "--data-provenance", str(data),
                    "--output", str(output), "--check-only"])
            manifest = json.loads((output / "MANIFEST.json").read_text())
            self.assertEqual(manifest["planned_fold_runs"], 10)
            self.assertEqual(manifest["seeds"], [66])
            self.assertTrue(manifest["keep_all_best_weights"])
            self.assertEqual(len(list((output / "configs").glob("*.json"))), 10)
            self.assertFalse((output / "RUNNER.json").exists())
            self.assertFalse((output / "queue").exists())


if __name__ == "__main__":
    unittest.main()
