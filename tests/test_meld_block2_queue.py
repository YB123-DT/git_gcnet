"""CPU-only contracts for the finite MELD scheduler and paired HPO score."""
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


MODULE = "experiments.osram_meld_block2_auto_20261014"


class QueueTests(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module(MODULE + ".scheduler")
        except ModuleNotFoundError:
            self.fail("persistent scheduler implementation is missing")

    def scheduler(self, root, **kwargs):
        module = self.module()
        scheduler = module.Scheduler(root, gpus=[0, 2, 3], per_gpu=4,
                                     smoke_costs={"m": {"peak_memory_mib": 100, "disk_bytes": 0}}, **kwargs)
        scheduler.poll_seconds = 0.01
        scheduler.disk_margin_bytes = 0
        scheduler.gpu_snapshot = lambda: {
            index: {"uuid": f"GPU-test-{index}", "free_mib": 24000}
            for index in (0, 2, 3, 4)
        }
        return scheduler

    def job(self, root, name="job"):
        return {"id": name, "method": "m", "config": {"epochs": 50},
                "output": str(Path(root) / "runs" / name)}

    def command(self, job, config, gpu):
        script = ("import json,pathlib,time; time.sleep(.03); "
                  "p=pathlib.Path(" + repr(job["output"]) + "); "
                  "p.mkdir(parents=True,exist_ok=True); "
                  "(p/'SUMMARY.json').write_text(json.dumps({'status':'complete',"
                  "'paired':{'acc':62.,'ua':38.,'mask_digest':'fixed'}}))")
        return [sys.executable, "-c", script, str(config)]

    def test_missing_implementation_is_detected(self):
        self.assertTrue(hasattr(self.module(), "Scheduler"))

    def test_real_cpu_children_persist_and_completed_jobs_are_reused(self):
        with tempfile.TemporaryDirectory() as root:
            scheduler = self.scheduler(root)
            scheduler.command = self.command
            records = scheduler.run_jobs([self.job(root, f"j{i}") for i in range(7)])
            self.assertTrue(all(row["status"] == "complete" for row in records))
            self.assertEqual(len({row["gpu"] for row in records}), 3)
            self.assertTrue(all(row["gpu_uuid"] != "GPU-test-4" for row in records))
            state = json.loads((Path(root) / "QUEUE.json").read_text())
            self.assertEqual(state["status"], "complete")
            self.assertTrue(all(row["pid"] > 0 for row in state["jobs"].values()))
            with patch.object(scheduler, "command", side_effect=AssertionError("relaunch")):
                self.assertEqual(scheduler.run_jobs([self.job(root, "j0")])[0]["status"],
                                 "complete")

    def test_bad_gpu_and_excess_capacity_are_rejected(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as root:
            for gpus, per_gpu in [([4], 1), ([0, 1, 2, 3], 1), ([0], 5)]:
                with self.assertRaises(ValueError):
                    module.Scheduler(root, gpus=gpus, per_gpu=per_gpu)

    def test_admission_reserves_peak_plus_one_gib_and_checks_disk(self):
        with tempfile.TemporaryDirectory() as root:
            scheduler = self.scheduler(root)
            scheduler.gpu_snapshot = lambda: {
                index: {"uuid": f"GPU-test-{index}", "free_mib": 1123 if index == 0 else 0}
                for index in (0, 2, 3)}
            self.assertIsNone(scheduler.admit(self.job(root)))
            scheduler.gpu_snapshot = lambda: {
                index: {"uuid": f"GPU-test-{index}", "free_mib": 1124 if index == 0 else 0}
                for index in (0, 2, 3)}
            self.assertEqual(scheduler.admit(self.job(root)), (0, "GPU-test-0"))
            scheduler.disk_margin_bytes = 10 ** 30
            self.assertIsNone(scheduler.admit(self.job(root)))

    def test_oom_retries_once_without_changing_configuration(self):
        with tempfile.TemporaryDirectory() as root:
            scheduler = self.scheduler(root)
            def command(job, config, gpu):
                script = ("import pathlib,json,sys; p=pathlib.Path(" + repr(job["output"]) +
                          "); p.mkdir(parents=True,exist_ok=True); marker=p/'attempt'; "
                          "first=not marker.exists(); marker.write_text('1'); "
                          "print('CUDA out of memory' if first else 'OK',flush=True); "
                          "(p/'SUMMARY.json').write_text(json.dumps({'status':'complete'})) "
                          "if not first else None; sys.exit(1 if first else 0)")
                return [sys.executable, "-c", script, str(config)]
            scheduler.command = command
            row = scheduler.run_jobs([self.job(root)])[0]
            self.assertEqual(row["status"], "complete")
            self.assertEqual(row["attempts"], 2)
            self.assertEqual(len(row["failures"]), 1)
            self.assertEqual(row["config"], {"epochs": 50})
            self.assertLess(scheduler.capacity[row["failures"][0]["gpu"]], 4)

    def test_recovery_does_not_adopt_unrelated_pid(self):
        with tempfile.TemporaryDirectory() as root:
            scheduler = self.scheduler(root)
            external = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"])
            try:
                job = self.job(root)
                scheduler.state["jobs"][job["id"]] = dict(
                    job, status="running", pid=external.pid, process_start="wrong",
                    command=["unrelated"], attempts=1, failures=[], gpu=0,
                    gpu_uuid="GPU-test-0", log=str(Path(root) / "missing.log"))
                scheduler.save()
                recovered = self.scheduler(root)
                recovered.command = self.command
                row = recovered.run_jobs([job])[0]
                self.assertEqual(row["status"], "complete")
                self.assertIsNone(external.poll())
                self.assertTrue(any("identity" in f["reason"] for f in row["failures"]))
            finally:
                external.terminate()
                external.wait()

    def test_recovery_waits_for_owned_worker_instead_of_relaunching(self):
        with tempfile.TemporaryDirectory() as root:
            scheduler = self.scheduler(root)
            job = self.job(root)
            row = dict(job, status="pending", attempts=0, failures=[])
            scheduler.state["jobs"][job["id"]] = row
            scheduler.command = self.command
            scheduler.launch(row, (0, "GPU-test-0"))
            recovered = self.scheduler(root)
            with patch.object(recovered, "command", side_effect=AssertionError("duplicate")):
                result = recovered.run_jobs([job])[0]
            scheduler.children[job["id"]].wait()
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["attempts"], 1)

    def test_zero_exit_without_summary_is_audited_failure(self):
        with tempfile.TemporaryDirectory() as root:
            scheduler = self.scheduler(root)
            scheduler.command = lambda job, config, gpu: [sys.executable, "-c", "pass", str(config)]
            row = scheduler.run_jobs([self.job(root)])[0]
            self.assertEqual(row["status"], "failed")
            self.assertEqual(row["attempts"], 1)
            self.assertEqual(len(row["failures"]), 1)

    def test_gpu_uuid_change_refuses_admission(self):
        with tempfile.TemporaryDirectory() as root:
            scheduler = self.scheduler(root)
            scheduler.admit(self.job(root))
            scheduler.state["gpu_uuids"]["0"] = "GPU-unexpected"
            with self.assertRaisesRegex(RuntimeError, "UUID changed"):
                scheduler.admit(self.job(root))

    def test_two_jobs_cannot_share_outputs_or_change_saved_configuration(self):
        with tempfile.TemporaryDirectory() as root:
            scheduler = self.scheduler(root)
            scheduler.command = self.command
            first = self.job(root, "first")
            second = dict(self.job(root, "second"), output=first["output"])
            with self.assertRaisesRegex(ValueError, "output"):
                scheduler.run_jobs([first, second])
            scheduler.run_jobs([first])
            changed = dict(first, config={"epochs": 20})
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                scheduler.run_jobs([changed])


class SearchScoreTests(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module(MODULE + ".optuna_search")
        except ModuleNotFoundError:
            self.fail("bounded search implementation is missing")

    def test_objective_uses_paired_common_metrics_only(self):
        module = self.module()
        base = {"acc": 60., "ua": 35., "mask_digest": "fixed"}
        summary = {"status": "complete", "paired": {"acc": 61., "ua": 37.,
                   "mask_digest": "fixed"}, "testbest": {"acc": 100., "ua": 100.}}
        self.assertAlmostEqual(module.paired_objective(summary, base), 1.4)
        summary["paired"]["mask_digest"] = "different"
        with self.assertRaisesRegex(ValueError, "mask"):
            module.paired_objective(summary, base)
        summary["status"] = "running"
        with self.assertRaises(ValueError):
            module.paired_objective(summary, base)

    def test_search_space_and_explicit_initial_five_are_bounded(self):
        module = self.module()
        self.assertEqual(len(module.INITIAL_CONFIGS), 5)
        self.assertEqual(len({tuple(sorted(p.items())) for p in module.INITIAL_CONFIGS}), 5)
        for config in module.INITIAL_CONFIGS:
            self.assertEqual(set(config), {"lr", "weight_decay", "dropout"})
            self.assertTrue(3e-4 <= config["lr"] <= 2e-3)
            self.assertTrue(1e-6 <= config["weight_decay"] <= 1e-3)
            self.assertTrue(.2 <= config["dropout"] <= .6)

    def test_worker_paired_schema_is_accepted(self):
        module = self.module()
        base = {"on": {"mean": {"ACC": 60., "UA": 35.}}, "mask_digest": "fixed"}
        summary = {"status": "complete", "paired": {
            "on": {"mean": {"ACC": 61., "UA": 37.}}, "mask_digest": "fixed"}}
        self.assertAlmostEqual(module.paired_objective(summary, base), 1.4)

    @unittest.skipUnless(importlib.util.find_spec("optuna"), "requires training environment Optuna")
    def test_sqlite_search_bounded_batches_and_resume_without_new_trials(self):
        module = self.module()
        from experiments.osram_meld_block2_auto_20261014.methods import _FIXED
        class CPUResults:
            def __init__(self):
                self.batches = []
            def run_jobs(self, jobs):
                self.batches.append(jobs)
                return [dict(job, status="complete", summary={"status": "complete",
                    "paired": {"on": {"mean": {"ACC": 61., "UA": 37.}},
                               "mask_digest": "fixed"}}) for job in jobs]
        with tempfile.TemporaryDirectory() as root:
            base = dict(_FIXED, train_missing_rates=[i / 10 for i in range(8)])
            flat = {"on": {"mean": {"ACC": 60., "UA": 35.}}, "mask_digest": "fixed"}
            results = CPUResults()
            method = "B1"
            rows = module.run_search(root, [method], base, flat, results, trials_per_model=16)
            self.assertEqual(len(rows), 16)
            self.assertTrue(all(row["state"] == "COMPLETE" for row in rows))
            self.assertEqual([len(batch) for batch in results.batches[:5]], [1] * 5)
            self.assertTrue(all(len(batch) <= 2 for batch in results.batches[5:]))
            self.assertTrue(all(job["config"]["epochs"] == 50 for batch in results.batches for job in batch))
            count = len(results.batches)
            module.run_search(root, [method], base, flat, results, trials_per_model=16)
            self.assertEqual(len(results.batches), count)

    @unittest.skipUnless(importlib.util.find_spec("optuna"), "requires training environment Optuna")
    def test_sqlite_running_trial_reconciliation_and_failure_audit(self):
        module = self.module()
        from experiments.osram_meld_block2_auto_20261014.methods import _FIXED
        class InterruptedResults:
            interrupted = False
            def run_jobs(self, jobs):
                if not self.interrupted:
                    self.interrupted = True
                    raise KeyboardInterrupt("controller interrupted; worker identity remains in queue")
                return [dict(job, status="failed", summary={"status": "failed"})
                        if job["id"].endswith("000") else
                        dict(job, status="complete", summary={"status": "complete",
                             "paired": {"acc": 61., "ua": 37., "mask_digest": "fixed"}})
                        for job in jobs]
        with tempfile.TemporaryDirectory() as root:
            base = dict(_FIXED, train_missing_rates=[i / 10 for i in range(8)])
            flat = {"acc": 60., "ua": 35., "mask_digest": "fixed"}
            method = "B1"
            results = InterruptedResults()
            with self.assertRaises(KeyboardInterrupt):
                module.run_search(root, [method], base, flat, results, trials_per_model=16)
            rows = module.run_search(root, [method], base, flat, results, trials_per_model=16)
            self.assertEqual(len(rows), 16)
            self.assertEqual(rows[0]["state"], "FAIL")
            self.assertIsNone(rows[0]["value"])
            self.assertIn("failure", rows[0]["user_attrs"])
            self.assertEqual(len(rows[0]["params"]), 3)
            self.assertEqual(sum(row["state"] == "COMPLETE" for row in rows), 15)
            best = json.loads((Path(root) / f"{method}_TRIALS.json").read_text())["best"]
            self.assertNotEqual(best["number"], 0)


if __name__ == "__main__":
    unittest.main()
