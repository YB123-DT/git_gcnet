"""Restore ONLY original Flat seed66, IEMOCAP4+6 fivefold, with retained exact-epoch artifacts."""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import fcntl
import importlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.osram_iemocap_b6_transfer_20261015 import run as common

TRAINING_SHA = common.TRAINING_SHA
DEFAULT_SOURCE = common.DEFAULT_SOURCE
DEFAULT_BASELINE = common.DEFAULT_BASELINE
DEFAULT_OUTPUT = Path("/home/qinjinghui/remote_experiments/osram_iemocap_flat_seed66_restore_20261016")
DATASETS, FOLDS, RATES, METRICS = common.DATASETS, common.FOLDS, common.RATES, common.METRICS
BOOKKEEPING = dict(common.BOOKKEEPING, osram_rpca_lr_multiplier=1.)
LABEL = "INTERNAL PER-RATE TEST-ORACLE ACC; original Flat, seed66, 100 epochs"


def load_sources(root):
    sources = {}
    for classes in DATASETS:
        for fold in FOLDS:
            path = (Path(root) / f"iemocap{classes}/seed_66/fold_{fold}/config.json").resolve(strict=True)
            sources[classes, 66, fold] = dict(config=json.loads(path.read_text()),
                config_path=str(path), config_sha256=common.sha256(path))
    return sources


def make_jobs(sources, output):
    if set(sources) != {(c, 66, f) for c in DATASETS for f in FOLDS}:
        raise ValueError("Exactly ten original Flat seed66 configurations are required")
    jobs = []
    for fold in FOLDS:
        for classes, dataset in DATASETS.items():
            baseline = deepcopy(sources[classes, 66, fold])
            original = baseline["config"]
            expected = dict(dataset=dataset, seed=66, fold=fold, epochs=100, batch_size=32,
                optimizer="adam", learning_rate=.001, weight_decay=1e-5, dropout=.5,
                train_rate_mode="cyclic", missing_mask_protocol="gcnet-random-mask-v1",
                training_objective="emotion-only", checkpoint_selection="test-oracle-per-rate",
                evaluate_test=True, osram_meaningful_block="none", osram_readout_fusion="flat",
                osram_output_dim=1600, backbone_type="osram")
            if (any(original.get(k) != v for k, v in expected.items())
                    or original.get("osram_disable_local_skip", False)
                    or original.get("initial_backbone_checkpoint")
                    or original.get("checkpoint_selection_metric", "dataset-default") not in ("dataset-default", "accuracy")
                    or original.get("train_missing_rates") != [i / 10 for i in range(8)]):
                raise ValueError(f"Original Flat/no-JEPA/corrected-mask protocol mismatch: {classes}/66/{fold}")
            config = dict(original, **BOOKKEEPING)
            name = f"iemocap{classes}_Original_Flat_seed66_fold{fold}"
            jobs.append(dict(id=name, method="Flat", config=config, baseline=baseline,
                configuration_diff={k: dict(original=original.get(k), effective=v, added=k not in original)
                                    for k, v in BOOKKEEPING.items() if k not in original or original[k] != v},
                output=str((Path(output) / f"runs/iemocap{classes}/seed_66/fold_{fold}").resolve())))
    return jobs


def smoke_peak(smoke):
    peak, reserved = smoke.get("peak_memory_mib"), smoke.get("peak_reserved_mib", 0.)
    if (smoke.get("status") != "complete" or smoke.get("smoke_only") is not True
            or smoke.get("config", {}).get("dataset") != "IEMOCAPSix"
            or smoke.get("config", {}).get("osram_meaningful_block") != common.B6["osram_meaningful_block"]
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                   for v in (peak, reserved)) or peak <= 0 or reserved < 0):
        raise ValueError("Require the existing complete IEMOCAPSix B6 smoke as conservative Flat memory bound")
    return float(max(peak, reserved))


def valid_summary(summary, job):
    if summary.get("method") != "Original_Flat" or not common.valid_summary(summary, job):
        return False
    try:
        check = json.loads((Path(job["output"]) / "FIRST_EPOCH_RETENTION_CHECK.json").read_text())
        return (check.get("status") == "complete" and check.get("best_checkpoints") == 8
                and check.get("selected_artifacts") == 8 and check.get("cuda_rng_states") == 1)
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def make_scheduler(frozen, root, source, worker, peak):
    # Reuse the audited UUID/process/memory admission and completed-run recovery.
    inherited = common.make_scheduler(frozen, root, source, worker, peak)

    class FlatScheduler(type(inherited)):
        def command(self, job, config_path, uuid):
            return super().command(job, config_path, uuid) + ["--flat-only"]

        def summary(self, row):
            result = super().summary(row)
            return result if result is not None and valid_summary(result, row) else None

    scheduler = FlatScheduler(root, gpus=(1, 3), per_gpu=4, per_gpu_limit=6, code_root=source,
        smoke_costs={"Flat": dict(peak_memory_mib=peak, disk_bytes=2 * 2**30)})
    saved = scheduler.state.get("capacity", {})
    scheduler.capacity = {g: min(limit, int(saved.get(str(g), limit))) for g, limit in ((1, 4), (3, 2))}
    return scheduler


def summarize(records, jobs, output):
    rows = {r["id"]: r for r in records}
    complete = {j["id"]: rows[j["id"]]["summary"] for j in jobs
        if rows.get(j["id"], {}).get("status") == "complete"
        and valid_summary(rows[j["id"]].get("summary", {}), j)}
    report = dict(label=LABEL, status="complete" if len(complete) == 10 else "incomplete_or_failed",
        seed=66, planned_fold_runs=10, completed_fold_runs=len(complete), datasets={},
        retention="Keep all 80 per-rate BEST weights, exact-epoch selection artifacts and ten complete last_training states",
        jobs=[dict(id=j["id"], output=j["output"], status=rows.get(j["id"], {}).get("status", "pending"),
                   summary_valid=j["id"] in complete, failures=rows.get(j["id"], {}).get("failures", [])) for j in jobs])
    table = []
    for dataset in DATASETS.values():
        selected = [s for s in complete.values() if s["dataset"] == dataset]
        fivefold = common._average(selected) if {s["fold"] for s in selected} == set(FOLDS) else None
        report["datasets"][dataset] = dict(folds={str(s["fold"]): s["testbest"] for s in selected}, fivefold=fivefold)
        for s in sorted(selected, key=lambda s: s["fold"]):
            for section, metrics in list(s["testbest"]["rates"].items()) + [
                    (name, s["testbest"][name]) for name in ("mean", "high")]:
                table.append(dict(dataset=dataset, seed=66, fold=s["fold"], rate=section,
                                  **{metric: metrics[metric] for metric in METRICS}))
        if fivefold:
            for section, metrics in list(fivefold["rates"].items()) + [(name, fivefold[name]) for name in ("mean", "high")]:
                table.append(dict(dataset=dataset, seed=66, fold="fivefold", rate=section, **metrics))
    output = Path(output)
    common.write_json(output / "SUMMARY.json", report)
    with (output / "SUMMARY.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=("dataset", "seed", "fold", "rate", *METRICS))
        writer.writeheader()
        writer.writerows(table)
    lines = ["# Original Flat seed66 restoration", "", LABEL, "",
        f"Completed fold runs: {len(complete)}/10. Status: {report['status']}.", "",
        "| Dataset | Scope | ACC | UA | W-F1 |", "|---|---|---:|---:|---:|"]
    for dataset, result in report["datasets"].items():
        if result["fivefold"]:
            for section in ("mean", "high"):
                m = result["fivefold"][section]
                lines.append(f"| {dataset} | fivefold {section} | {m['ACC']:.3f} | {m['UA']:.3f} | {m['WF1']:.3f} |")
    lines += ["", "Fivefold averages require all five valid folds and weight folds equally. Seed66 only; no multi-seed uncertainty estimate.",
        "Eight independently ACC-selected epochs are internal test-oracle diagnostics. Exact UA uses the same selected logits and masks.",
        "All 80 BEST checkpoints and selected artifacts plus ten last_training states are retained; old discarded weights are not loaded.", ""]
    (output / "RESULT.md").write_text("\n".join(lines))
    return report


def prepare(args):
    source, worker = args.source_root.resolve(strict=True), args.worker.resolve(strict=True)
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip() != TRAINING_SHA:
        raise ValueError("Require immutable numerical source785a012")
    if subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=source, text=True).strip():
        raise ValueError("Numerical source must have no tracked changes")
    jobs = make_jobs(load_sources(args.baseline_root), args.output)
    smoke_path, data_path = args.smoke_summary.resolve(strict=True), args.data_provenance.resolve(strict=True)
    smoke, data = json.loads(smoke_path.read_text()), json.loads(data_path.read_text())
    peak = smoke_peak(smoke)
    provenance = dict(training_source_git_sha=TRAINING_SHA, source_root=str(source),
        implementation_git_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        launcher_sha256=common.sha256(__file__), helper_sha256=common.sha256(common.__file__),
        worker_path=str(worker), worker_sha256=common.sha256(worker),
        data_provenance_path=str(data_path), data_provenance_sha256=common.sha256(data_path), data_provenance=data,
        smoke_summary_path=str(smoke_path), smoke_summary_sha256=common.sha256(smoke_path),
        smoke_source_identity=smoke.get("identity"), smoke_bound_mib=peak,
        smoke_note="Reuse authorized longest-batch B6 bound; no new GPU smoke; max(allocated,reserved)",
        code_hashes={str(p.relative_to(source)): common.sha256(p)
                     for folder in ("gcnet_missing_m3", "gcnet_modality_jepa") for p in sorted((source / folder).glob("*.py"))},
        environment=dict(python=sys.version, executable=sys.executable, platform=platform.platform()))
    for job in jobs:
        job["provenance"] = provenance
    manifest = dict(label=LABEL, planned_fold_runs=10, seeds=[66], fold_order=list(FOLDS),
        capacities={"1": 4, "3": 2}, global_ours_limit=6, excluded_gpus=[4, 5],
        keep_all_best_weights=True, keep_selection_artifacts=True, keep_last_training=True,
        fresh_initialization=True, provenance=provenance, jobs=jobs)
    path = args.output / "MANIFEST.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Do not mix source/data/config/worker versions in this output")
    common.write_json(path, manifest)
    for job in jobs:
        common.write_json(args.output / "configs" / f"{job['id']}.json", job["config"])
    return jobs, peak, source, worker, manifest


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
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".controller.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        jobs, peak, source, worker, manifest = prepare(args)
        if args.check_only:
            print(json.dumps(dict(preflight="prepared", jobs=[j["id"] for j in jobs], peak_bound_mib=peak)))
            return
        sys.path.insert(0, str(source))
        frozen = importlib.import_module("experiments.osram_meld_block2_auto_20261014.scheduler")
        if Path(frozen.__file__).resolve().parents[2] != source:
            raise ValueError("Scheduler must import from the immutable numerical source")
        scheduler = make_scheduler(frozen, args.output / "queue", source, worker, peak)
        runner = dict(status="running", pid=os.getpid(), started_unix=time.time(), provenance=manifest["provenance"])
        common.write_json(args.output / "RUNNER.json", runner)
        try:
            while True:
                records = scheduler.run_jobs(jobs, return_on_first_completion=True)
                result = summarize(records, jobs, args.output)
                if all(row["status"] in ("complete", "failed") for row in records):
                    break
            runner.update(status=result["status"], finished_unix=time.time())
        except BaseException as error:
            runner.update(status="failed", error=repr(error), finished_unix=time.time())
            raise
        finally:
            common.write_json(args.output / "RUNNER.json", runner)


if __name__ == "__main__":
    main()
