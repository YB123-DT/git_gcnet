"""Complete Flat seeds67/68 in cmerc-repro; reuse ten completed seed66 folds read-only."""
from __future__ import annotations

import argparse
import csv
import fcntl
import importlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.osram_iemocap_flat_seed66_restore_20261016 import run as flat

common = flat.common
TRAINING_SHA = flat.TRAINING_SHA
EXPECTED_PYTHON = Path("/data2/yb/conda_envs/cmerc-repro/bin/python")
EXPECTED_TORCH = "1.13.1+cu117"
DEFAULT_OUTPUT = Path("/home/qinjinghui/remote_experiments/osram_iemocap_flat_3seed_cmerc_20261016")
DEFAULT_B6 = common.DEFAULT_OUTPUT
NEW_SEEDS = (67, 68)
GPU5_UUID = "GPU-fa1e8bfd-85d8-9599-f804-7c88b9c71b62"
LABEL = "INTERNAL PER-RATE TEST-ORACLE ACC; original Flat seeds66/67/68; cmerc-repro"


def require_environment(executable=None, torch_version=None):
    executable = sys.executable if executable is None else executable
    if Path(executable).resolve() != EXPECTED_PYTHON.resolve():
        raise ValueError("Replication requires /data2/yb/conda_envs/cmerc-repro/bin/python")
    if torch_version is None:
        import torch
        torch_version = str(torch.__version__)
    if torch_version != EXPECTED_TORCH:
        raise ValueError("Replication requires torch1.13.1+cu117")
    return dict(executable=str(executable), resolved_executable=str(Path(executable).resolve()),
                python=sys.version, torch=torch_version)


def valid_summary(summary, job):
    valid = (flat.valid_summary(summary, job)
            and summary.get("identity", {}).get("source_git_sha") == TRAINING_SHA
            and summary.get("provenance", {}).get("environment", {}).get("torch") == EXPECTED_TORCH)
    if not valid or job["config"]["seed"] == 66:
        return valid
    environment = summary.get("provenance", {}).get("replication_environment", {})
    return (Path(environment.get("executable", "")).resolve() == EXPECTED_PYTHON.resolve()
            and all(environment.get(k) == v for k, v in dict(torch=EXPECTED_TORCH,
                cuda="11.7", numpy="1.26.4", sklearn="1.3.2").items()))


def load_seed66(jobs, root):
    manifest_path = (Path(root) / "MANIFEST.json").resolve(strict=True)
    manifest = json.loads(manifest_path.read_text())
    provenance = manifest.get("provenance", {})
    environment = provenance.get("environment", {})
    if (len(jobs) != 10 or any(j["config"]["seed"] != 66 for j in jobs)
            or provenance.get("training_source_git_sha") != TRAINING_SHA
            or Path(environment.get("executable", "")).resolve() != EXPECTED_PYTHON.resolve()):
        raise ValueError("Completed seed66 manifest must prove source785a012 and cmerc-repro executable")
    records, evidence = [], dict(root=str(Path(root).resolve()), manifest_path=str(manifest_path),
        manifest_sha256=common.sha256(manifest_path), environment=environment, summaries=[])
    for job in jobs:
        path = Path(job["output"]) / "SUMMARY.json"
        summary = json.loads(path.read_text())
        if not valid_summary(summary, job):
            raise ValueError("Completed seed66 result/config/source/environment/retention mismatch: " + job["id"])
        retention = path.with_name("FIRST_EPOCH_RETENTION_CHECK.json")
        evidence["summaries"].append(dict(id=job["id"], path=str(path), sha256=common.sha256(path),
            retention_path=str(retention), retention_sha256=common.sha256(retention),
            environment=summary["provenance"]["environment"], identity=summary["identity"]))
        records.append(dict(job, status="complete", summary=summary, reused=True))
    return records, evidence


def make_scheduler(frozen, root, source, worker, peak):
    inherited = flat.make_scheduler(frozen, root, source, worker, peak)

    class ReplicationScheduler(type(inherited)):
        def admit(self, job):
            # Latest authorization replaces this replication queue's old1/3
            # placement only. Reuse the original health/memory/disk admission.
            device = frozen.Scheduler.admit(self, job)
            if device is not None and device != (5, GPU5_UUID):
                raise RuntimeError("Flat replication requires the verified physical GPU5 UUID")
            return device

        def command(self, job, config_path, uuid):
            return super().command(job, config_path, uuid) + ["--flat-replication"]

        def summary(self, row):
            result = super().summary(row)
            return result if result is not None and valid_summary(result, row) else None

    scheduler = ReplicationScheduler(root, gpus=(5,), per_gpu=2, per_gpu_limit=2, code_root=source,
        smoke_costs={"Flat": dict(peak_memory_mib=peak, disk_bytes=2 * 2**30)})
    saved = scheduler.state.get("capacity", {})
    scheduler.capacity = {5: min(2, int(saved.get("5", 2)))}
    return scheduler


def summarize(records, jobs, reused, output):
    planned = {j["id"]: j for j in [*jobs, *reused]}
    if len(jobs) != 20 or len(reused) != 10 or len(planned) != 30:
        raise ValueError("Report requires exactly20 new Flat67/68 jobs plus10 reused Flat66 jobs")
    rows = {r["id"]: r for r in [*reused, *records]}
    complete = {key: rows[key]["summary"] for key, job in planned.items()
        if rows.get(key, {}).get("status") == "complete" and valid_summary(rows[key].get("summary", {}), job)}
    all_complete = len(complete) == 30
    report = dict(label=LABEL, status="complete" if all_complete else "incomplete_or_failed",
        planned_fold_runs=30, planned_new_fold_runs=20, reused_fold_runs=10, completed_fold_runs=len(complete),
        completed_new_fold_runs=sum(j["id"] in complete for j in jobs), datasets={},
        B6_reference_root=str(DEFAULT_B6), B6_reference_note="Completed reference only; this queue never submits B6 or seed66",
        jobs=[dict(id=key, output=job["output"], reused=key in {r["id"] for r in reused},
            status=rows.get(key, {}).get("status", "pending"), summary_valid=key in complete,
            failures=rows.get(key, {}).get("failures", [])) for key, job in planned.items()])
    table = []
    for dataset in flat.DATASETS.values():
        seeds, folds = {}, {}
        for seed in (66, 67, 68):
            selected = [s for s in complete.values() if s["dataset"] == dataset and s["seed"] == seed]
            folds[str(seed)] = {str(s["fold"]): s["testbest"] for s in selected}
            if {s["fold"] for s in selected} == set(flat.FOLDS):
                seeds[str(seed)] = common._average(selected)
            for s in selected:
                for section, metrics in list(s["testbest"]["rates"].items()) + [(name, s["testbest"][name]) for name in ("mean", "high")]:
                    table.append(dict(dataset=dataset, seed=seed, fold=s["fold"], rate=section,
                                      **{metric: metrics[metric] for metric in flat.METRICS}))
        aggregate = None
        if all_complete:
            aggregate = dict(common._three_seed(seeds), rates={rate: {metric: dict(
                mean=statistics.mean(s["rates"][rate][metric] for s in seeds.values()),
                std_ddof1=statistics.stdev(s["rates"][rate][metric] for s in seeds.values()))
                for metric in flat.METRICS} for rate in flat.RATES})
        report["datasets"][dataset] = dict(folds=folds, seed_fivefold=seeds, three_seed=aggregate)
    output = Path(output)
    common.write_json(output / "SUMMARY.json", report)
    with (output / "SUMMARY.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=("dataset", "seed", "fold", "rate", *flat.METRICS))
        writer.writeheader()
        writer.writerows(table)
    lines = ["# Original Flat three-seed completion", "", LABEL, "",
        f"Complete fold runs: {len(complete)}/30 (10 reused; {report['completed_new_fold_runs']}/20 new complete).", "",
        "| Dataset | Fivefold three-seed scope | ACC | UA | W-F1 |", "|---|---|---:|---:|---:|"]
    for dataset, result in report["datasets"].items():
        if result["three_seed"]:
            for section in ("mean", "high"):
                values = result["three_seed"][section]
                cells = [f"{values[m]['mean']:.3f} ± {values[m]['std_ddof1']:.3f}" for m in ("ACC", "UA", "WF1")]
                lines.append(f"| {dataset} | {section} | " + " | ".join(cells) + " |")
    lines += ["", "Fivefold means require all five folds; three-seed mean ± sample SD appears only after all30 folds complete.",
        "Seed66 outputs are reused read-only. New training is limited to Flat67/68; all checkpoints/artifacts/full recovery states are retained.",
        f"Completed B6 reference: {DEFAULT_B6}. No B6 run is submitted. Test-best results are internal per-rate test-oracle ACC diagnostics.", ""]
    (output / "RESULT.md").write_text("\n".join(lines))
    return report


def prepare(args, environment):
    source, worker = args.source_root.resolve(strict=True), args.worker.resolve(strict=True)
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip() != TRAINING_SHA:
        raise ValueError("Require immutable numerical source785a012")
    if subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=source, text=True).strip():
        raise ValueError("Numerical source must have no tracked edits")
    jobs = flat.make_jobs(flat.load_sources(args.baseline_root, seeds=NEW_SEEDS), args.output, seeds=NEW_SEEDS)
    # Start4/6classes together for each seed/fold; this changes placement/order,
    # not any job's model, optimizer, masks or training RNG.
    jobs.sort(key=lambda job: (job['config']['seed'], flat.FOLDS.index(job['config']['fold']),
                              job['config']['dataset']))
    seed66_jobs = flat.make_jobs(flat.load_sources(args.baseline_root), args.seed66_root)
    reused, evidence = load_seed66(seed66_jobs, args.seed66_root)
    smoke_path, data_path = args.smoke_summary.resolve(strict=True), args.data_provenance.resolve(strict=True)
    smoke, data = json.loads(smoke_path.read_text()), json.loads(data_path.read_text())
    peak = flat.smoke_peak(smoke)
    provenance = dict(training_source_git_sha=TRAINING_SHA, source_root=str(source),
        implementation_git_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        launcher_sha256=common.sha256(__file__), flat_helper_sha256=common.sha256(flat.__file__),
        transfer_helper_sha256=common.sha256(common.__file__), worker_path=str(worker), worker_sha256=common.sha256(worker),
        environment=environment, seed66_reuse=evidence, smoke_summary_path=str(smoke_path),
        smoke_summary_sha256=common.sha256(smoke_path), smoke_bound_mib=peak, smoke_source_identity=smoke.get("identity"),
        data_provenance_path=str(data_path), data_provenance_sha256=common.sha256(data_path), data_provenance=data,
        B6_reference_root=str(DEFAULT_B6), B6_rerun=False)
    for job in jobs:
        job["provenance"] = provenance
    manifest = dict(label=LABEL, planned_new_fold_runs=20, reused_fold_runs=10, planned_fold_runs=30,
        new_seeds=list(NEW_SEEDS), reused_seeds=[66], capacities={"5": 2},
        authorized_gpu_uuid=GPU5_UUID, excluded_gpus=[0, 1, 2, 3, 4, 6, 7],
        preserve_all_best_weights=True, preserve_selection_artifacts=True, preserve_last_training=True,
        provenance=provenance, jobs=jobs)
    path = args.output / "MANIFEST.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Do not mix source/environment/config/worker/reused-result versions in this output")
    common.write_json(path, manifest)
    for job in jobs:
        common.write_json(args.output / "configs" / f"{job['id']}.json", job["config"])
    return jobs, reused, peak, source, worker, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=flat.DEFAULT_SOURCE)
    parser.add_argument("--baseline-root", type=Path, default=flat.DEFAULT_BASELINE)
    parser.add_argument("--seed66-root", type=Path, default=flat.DEFAULT_OUTPUT)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--smoke-summary", type=Path, required=True)
    parser.add_argument("--data-provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    environment = require_environment()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".controller.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        jobs, reused, peak, source, worker, manifest = prepare(args, environment)
        if args.check_only:
            print(json.dumps(dict(preflight="prepared", new_jobs=len(jobs), reused_jobs=len(reused), environment=environment)))
            return
        sys.path.insert(0, str(source))
        frozen = importlib.import_module("experiments.osram_meld_block2_auto_20261014.scheduler")
        if Path(frozen.__file__).resolve().parents[2] != source:
            raise ValueError("Scheduler must import from immutable numerical source")
        scheduler = make_scheduler(frozen, args.output / "queue", source, worker, peak)
        runner = dict(status="running", pid=os.getpid(), started_unix=time.time(), provenance=manifest["provenance"])
        common.write_json(args.output / "RUNNER.json", runner)
        try:
            while True:
                records = scheduler.run_jobs(jobs, return_on_first_completion=True)
                report = summarize(records, jobs, reused, args.output)
                if all(r["status"] in ("complete", "failed") for r in records):
                    break
            runner.update(status=report["status"], finished_unix=time.time())
        except BaseException as error:
            runner.update(status="failed", error=repr(error), finished_unix=time.time())
            raise
        finally:
            common.write_json(args.output / "RUNNER.json", runner)


if __name__ == "__main__":
    main()
