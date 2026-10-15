"""Finite original-Flat-protocol B6 transfer: two IEMOCAP tasks, three seeds, five folds."""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import fcntl
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time


TRAINING_SHA = "785a01242ef1a82b97d5337acbb33ba963df5537"
DEFAULT_SOURCE = Path("/home/qinjinghui/remote_experiments/osram_iemocap_b6_source_785a012")
DEFAULT_BASELINE = Path("/data2/yb/paper/remote_runs_20261012/results/osram_gcnet_random_mask_iemocap_meld_20261013_r1")
DEFAULT_OUTPUT = Path("/home/qinjinghui/remote_experiments/osram_iemocap_b6_transfer_20261015")
DATASETS = {4: "IEMOCAPFour", 6: "IEMOCAPSix"}
SEEDS = (66, 67, 68)
FOLDS = (5, 1, 2, 3, 4)
RATES = tuple(f"{i / 10:.1f}" for i in range(8))
METRICS = ("ACC", "UA", "WF1", "CE")
B6 = dict(osram_meaningful_block="learned_robust_pca_evidence_decomposition",
          osram_rpca_rank=2, osram_rpca_steps=14,
          osram_rpca_threshold_init=.33599569865951234,
          osram_rpca_lr_multiplier=2.8200380294401013)
BOOKKEEPING = dict(preserve_selection_artifacts=True,
                   checkpoint_selection_metric="accuracy", device="cuda:0")
LABEL = "INTERNAL PER-RATE TEST-ORACLE ACC; selected B6 settings transferred from MELD"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_sources(root):
    sources = {}
    for classes in DATASETS:
        for seed in SEEDS:
            for fold in FOLDS:
                path = (Path(root) / f"iemocap{classes}/seed_{seed}/fold_{fold}/config.json").resolve(strict=True)
                sources[classes, seed, fold] = dict(config=json.loads(path.read_text()),
                    config_path=str(path), config_sha256=sha256(path))
    return sources


def make_jobs(sources, output):
    expected_keys = {(c, s, f) for c in DATASETS for s in SEEDS for f in FOLDS}
    if set(sources) != expected_keys:
        raise ValueError("Exactly the original 30 IEMOCAP baseline configurations are required")
    jobs = []
    for fold in FOLDS:  # First six jobs are fold5, never label this as five-fold CV.
        for classes, dataset in DATASETS.items():
            for seed in SEEDS:
                source = deepcopy(sources[classes, seed, fold])
                old = source["config"]
                expected = dict(dataset=dataset, seed=seed, fold=fold, epochs=100,
                    batch_size=32, optimizer="adam", learning_rate=.001, weight_decay=1e-5,
                    dropout=.5, train_rate_mode="cyclic", missing_mask_protocol="gcnet-random-mask-v1",
                    training_objective="emotion-only", checkpoint_selection="test-oracle-per-rate",
                    evaluate_test=True, osram_meaningful_block="none",
                    osram_readout_fusion="flat", osram_output_dim=1600, backbone_type="osram")
                if any(old.get(k) != value for k, value in expected.items()):
                    raise ValueError(f"Original Flat baseline protocol mismatch: {classes}/{seed}/{fold}")
                if (old.get("osram_disable_local_skip", False)
                        or old.get("checkpoint_selection_metric", "dataset-default") not in ("dataset-default", "accuracy")
                        or tuple(old.get("train_missing_rates", ())) != tuple(i / 10 for i in range(8))):
                    raise ValueError("Require original Local Skip, cyclic eight rates, and IEMOCAP ACC selection")
                config = dict(old, **B6, **BOOKKEEPING)
                diff = {k: dict(original=old.get(k), effective=config[k], added=k not in old)
                        for k in config if k not in old or old[k] != config[k]}
                name = f"iemocap{classes}_B6_tuned_seed{seed}_fold{fold}"
                jobs.append(dict(id=name, method="B6", config=config,
                    output=str((Path(output) / "runs" / f"iemocap{classes}" / f"seed_{seed}" / f"fold_{fold}").resolve()),
                    baseline=source, configuration_diff=diff))
    return jobs


def smoke_peak(smoke, jobs):
    peak = smoke.get("peak_memory_mib")
    if (smoke.get("status") != "complete" or smoke.get("smoke_only") is not True
            or smoke.get("config") not in [j["config"] for j in jobs]
            or isinstance(peak, bool) or not isinstance(peak, (int, float))
            or not math.isfinite(peak) or peak <= 0):
        raise ValueError("Require an existing complete compatible real-batch smoke with finite measured peak")
    return float(peak)


def valid_summary(summary, job):
    try:
        config = job["config"]
        if (summary["status"] != "complete" or summary.get("smoke_only", False)
                or summary["config"] != config or not summary["provenance"]
                or any(summary[k] != config[k] for k in ("dataset", "seed", "fold"))
                or set(summary["testbest"]["rates"]) != set(RATES)):
            return False
        rows = list(summary["testbest"]["rates"].values()) + [
            summary["testbest"][section] for section in ("mean", "high")]
        return all(isinstance(row[metric], (int, float)) and not isinstance(row[metric], bool)
                   and math.isfinite(row[metric]) for row in rows for metric in METRICS)
    except (KeyError, TypeError, ValueError):
        return False


def make_scheduler(frozen, root, source, worker, peak):
    class TransferScheduler(frozen.Scheduler):
        def gpu_snapshot(self):
            snapshot = super().gpu_snapshot()
            raw = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                           "--format=csv,noheader,nounits"], text=True)
            processes = {}
            for line in raw.splitlines():
                if line.strip():
                    uuid, pid = (part.strip() for part in line.split(","))
                    processes.setdefault(uuid, set()).add(int(pid))
            for row in snapshot.values():
                row["compute_pids"] = sorted(processes.get(row["uuid"], set()))
            return snapshot

        def command(self, job, config_path, uuid):
            return [self.python, "-u", str(Path(worker).resolve()), "--source-root", str(self.code_root),
                    "--config", str(config_path), "--output", job["output"], "--gpu-uuid", uuid]

        def admit(self, job):
            peak_mib, disk = self.costs(job)
            active = [r for r in self.state["jobs"].values() if r["status"] in ("running", "launching")]
            if shutil.disk_usage(self.root).free < disk + sum(self.costs(r)[1] for r in active) + self.disk_margin_bytes:
                return None
            snapshot = self.gpu_snapshot()  # Recheck devices, actual PIDs, memory before every launch.
            counts = {g: sum(r.get("gpu") == g for r in active) for g in self.gpus}
            for gpu in self.gpus:
                device = snapshot.get(gpu)
                if (not device or not device["uuid"].startswith("GPU-")
                        or device["uuid"] in {snapshot.get(g, {}).get("uuid") for g in (4, 5)}):
                    raise RuntimeError("Healthy GPU1/GPU3 UUID validation failed; GPU4/5 excluded")
                old = self.state["gpu_uuids"].get(str(gpu), device["uuid"])
                if old != device["uuid"]:
                    raise RuntimeError(f"GPU {gpu} UUID changed since queue creation")
                self.state["gpu_uuids"][str(gpu)] = device["uuid"]
            for gpu in sorted(self.gpus, key=lambda g: (counts[g], -snapshot[g]["free_mib"])):
                ours = [r for r in active if r.get("gpu") == gpu]
                visible = set(snapshot[gpu]["compute_pids"])
                total = len(visible | {r["pid"] for r in ours if r.get("pid")}) + sum(not r.get("pid") for r in ours)
                reserved = sum(self.costs(r)[0] for r in ours)
                if (counts[gpu] < self.capacity[gpu] and (gpu != 1 or total < 8)
                        and snapshot[gpu]["free_mib"] - reserved >= peak_mib + 1024):
                    return gpu, snapshot[gpu]["uuid"]
            return None

        def summary(self, row):
            result = super().summary(row)
            return result if result is not None and valid_summary(result, row) else None

        def recover(self):
            super().recover()
            for row in self.state["jobs"].values():
                if row["status"] == "pending" and self.summary(row) is not None:
                    self.finish(row, None)

        def launch(self, row, device):
            write_json(self.root / "provenance" / f"{row['id']}.json",
                       {k: row[k] for k in ("baseline", "configuration_diff", "config", "provenance") if k in row})
            super().launch(row, device)

    scheduler = TransferScheduler(root, gpus=(1, 3), per_gpu=6, per_gpu_limit=6,
        code_root=source, smoke_costs={"B6": dict(peak_memory_mib=peak, disk_bytes=2 * 2**30)})
    saved = scheduler.state.get("capacity", {})
    scheduler.capacity = {g: min(limit, int(saved.get(str(g), limit))) for g, limit in ((1, 8), (3, 2))}
    return scheduler


def _average(summaries):
    return dict(rates={rate: {metric: statistics.mean(s["testbest"]["rates"][rate][metric] for s in summaries)
                             for metric in METRICS} for rate in RATES},
                **{section: {metric: statistics.mean(s["testbest"][section][metric] for s in summaries)
                             for metric in METRICS} for section in ("mean", "high")})


def _three_seed(seeds):
    return {section: {metric: dict(mean=statistics.mean(s[section][metric] for s in seeds.values()),
                                  std_ddof1=statistics.stdev(s[section][metric] for s in seeds.values()))
                      for metric in METRICS} for section in ("mean", "high")}


def summarize(records, jobs, output):
    expected = {j["id"]: j for j in jobs}
    rows = {r["id"]: r for r in records if r["id"] in expected}
    complete = {key: r["summary"] for key, r in rows.items()
                if r["status"] == "complete" and valid_summary(r.get("summary", {}), expected[key])}
    report = dict(status="complete" if len(complete) == 30 else "incomplete_or_failed", label=LABEL,
                  planned_fold_runs=30, completed_fold_runs=len(complete), datasets={},
                  baseline_exact_testbest_UA="unavailable; old exact selection-epoch artifacts not established",
                  baseline_note="B6 results only; old ACC or fresh final reevaluation must not be relabelled as exact-epoch baseline UA",
                  jobs=[dict(id=j["id"], status=rows.get(j["id"], {}).get("status", "pending"),
                             summary_valid=j["id"] in complete, output=j["output"],
                             failures=rows.get(j["id"], {}).get("failures", [])) for j in jobs])
    table = []
    for dataset in DATASETS.values():
        found = [s for s in complete.values() if s["dataset"] == dataset]
        fold5 = {str(s["seed"]): s["testbest"] for s in found if s["fold"] == 5}
        seed_fivefold = {}
        for seed in SEEDS:
            seed_rows = [s for s in found if s["seed"] == seed]
            if {s["fold"] for s in seed_rows} == set(FOLDS):
                seed_fivefold[str(seed)] = _average(seed_rows)
        report["datasets"][dataset] = dict(fold5=dict(seeds=fold5, three_seed=_three_seed(fold5) if len(fold5) == 3 else None),
            fivefold=dict(seeds=seed_fivefold, three_seed=_three_seed(seed_fivefold)) if len(seed_fivefold) == 3 else None,
            completed_folds_by_seed={str(seed): sorted(s["fold"] for s in found if s["seed"] == seed) for seed in SEEDS})
        for s in sorted(found, key=lambda s: (s["fold"], s["seed"])):
            for section, metrics in list(s["testbest"]["rates"].items()) + [
                    (section, s["testbest"][section]) for section in ("mean", "high")]:
                table.append(dict(dataset=dataset, seed=s["seed"], fold=s["fold"], rate=section,
                                  **{metric: metrics[metric] for metric in METRICS}))
    output = Path(output)
    write_json(output / "SUMMARY.json", report)
    write_json(output / "FOLD5.json", dict(label=LABEL, scope="fold5 only; not a fivefold average",
        datasets={d: result["fold5"] for d, result in report["datasets"].items()}))
    with (output / "SUMMARY.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=("dataset", "seed", "fold", "rate", *METRICS))
        writer.writeheader()
        writer.writerows(table)
    lines = ["# Original-protocol IEMOCAP B6 transfer", "", LABEL, "",
             f"Completed fold runs: {len(complete)}/30. Status: {report['status']}.", "",
             "Fold5 results are stored separately in FOLD5.json. Fivefold averages appear only after all five folds and three seeds finish.", "",
             "| Dataset | Scope | ACC | UA | W-F1 |", "|---|---|---:|---:|---:|"]
    for dataset, result in report["datasets"].items():
        for section in ("fold5", "fivefold"):
            aggregate = (result[section] or {}).get("three_seed")
            if aggregate:
                values = [f"{aggregate['mean'][metric]['mean']:.3f} ± {aggregate['mean'][metric]['std_ddof1']:.3f}" for metric in ("ACC", "UA", "WF1")]
                lines.append(f"| {dataset} | {section} | " + " | ".join(values) + " |")
    lines += ["", "Values are equal-fold, then three-seed mean ± sample SD. Test-best uses separate ACC-selected epochs at eight missing rates.",
              "Old exact-selection-epoch baseline UA is unavailable. Fresh final reevaluation and old baseline ACC are separate evidence.", ""]
    (output / "RESULT.md").write_text("\n".join(lines))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--smoke-summary", type=Path, required=True)
    parser.add_argument("--data-provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    source = args.source_root.resolve(strict=True)
    worker = args.worker.resolve(strict=True)
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip() != TRAINING_SHA:
        raise RuntimeError("Require immutable IEMOCAP artifact-capable training checkout785a012")
    if subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=source, text=True).strip():
        raise RuntimeError("Frozen tracked training source has uncommitted changes")
    sys.path.insert(0, str(source))
    frozen = importlib.import_module("experiments.osram_meld_block2_auto_20261014.scheduler")
    if Path(frozen.__file__).resolve().parents[2] != source:
        raise RuntimeError("Scheduler did not import from the frozen source")
    jobs = make_jobs(load_sources(args.baseline_root), args.output)
    smoke_path = args.smoke_summary.resolve(strict=True)
    smoke = json.loads(smoke_path.read_text())
    peak = smoke_peak(smoke, jobs)
    data_path = args.data_provenance.resolve(strict=True)
    data = json.loads(data_path.read_text())
    packages = {}
    for package in ("torch", "numpy", "scikit-learn"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "unavailable"
    provenance = dict(training_source_git_sha=TRAINING_SHA, source_root=str(source), worker=str(worker),
        worker_sha256=sha256(worker), launcher_sha256=sha256(__file__),
        data_provenance_path=str(data_path), data_provenance_sha256=sha256(data_path),
        data_provenance=data, smoke_summary_path=str(smoke_path), smoke_summary_sha256=sha256(smoke_path),
        peak_memory_mib=peak, environment=dict(python=sys.version, executable=sys.executable,
            platform=platform.platform(), packages=packages,
            variables={k: os.environ[k] for k in ("GCNET_DATASET_ROOT", "IEMOCAP_DATASET_ROOT", "GCNET_CPU_THREADS") if k in os.environ}))
    for job in jobs:
        job["provenance"] = provenance
    manifest = dict(label=LABEL, planned_fold_runs=30, fold_order=list(FOLDS), seeds=list(SEEDS),
        capacities={"1": 8, "3": 2}, gpu1_limit_scope="all actual compute PIDs plus launch reservations",
        gpu3_limit_scope="this queue's jobs", excluded_gpus=[4, 5], memory_reserve_mib=1024,
        provenance=provenance, jobs=jobs)
    if args.check_only:
        print(json.dumps(dict(preflight="passed", jobs=[j["id"] for j in jobs], peak_memory_mib=peak)))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".controller.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = args.output / "MANIFEST.json"
        if old.exists() and json.loads(old.read_text()) != manifest:
            raise ValueError("Do not mix worker/source/data/config versions in this output")
        write_json(old, manifest)
        scheduler = make_scheduler(frozen, args.output / "queue", source, worker, peak)
        runner = dict(status="running", pid=os.getpid(), started_unix=time.time(), provenance=provenance)
        write_json(args.output / "RUNNER.json", runner)
        try:
            while True:
                records = scheduler.run_jobs(jobs, return_on_first_completion=True)
                result = summarize(records, jobs, args.output)
                if all(r["status"] in ("complete", "failed") for r in records):
                    break
            runner.update(status=result["status"], finished_unix=time.time())
        except BaseException as error:
            runner.update(status="failed", error=repr(error), finished_unix=time.time())
            raise
        finally:
            write_json(args.output / "RUNNER.json", runner)


if __name__ == "__main__":
    main()
