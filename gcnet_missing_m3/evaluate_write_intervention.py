"""Evaluate a frozen checkpoint with fixed-step or address-based interventions.

Run as ``python -m gcnet_missing_m3.evaluate_write_intervention --help``.
No optimizer, checkpoint selection, or training is performed.
"""

import argparse
import gzip
import hashlib
import inspect
import json
import math
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path


MODES = ("reference", "fixed0.9", "global", "protected")
ALLOWED_MODES = (*MODES, "fixed0.95", "fixed0.8", "fixed0.6", "fixed0.4", "fixed0.2", "fixed0.0")
IDENTITY_MODES = ("reference", "global", "protected")
WRITE_FIT_METRICS = ("err_before", "err_after", "err_original_after", "fit_gain")
RETENTION_METRICS = ("err_decay", "err_post", "decay_damage", "write_damage")
AUDIT_METRICS = (
    "original_norm", "protected_norm", "global_norm", "global_scale",
    "norm_mismatch", "applied_norm",
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--feature-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rates", nargs="+", type=float, default=[0., .1, .3, .5, .7])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--modes", nargs="+", choices=ALLOWED_MODES, default=list(MODES))
    parser.add_argument("--evaluation-write-step", type=float, default=None,
                        help="Explicit native write-step override for reference-only cross evaluation.")
    return parser


def resolve_evaluation_step(saved_step, override, modes):
    step = float(saved_step if override is None else override)
    if not math.isfinite(step) or not 0. <= step <= 1.:
        raise ValueError("evaluation write step must be finite and in [0,1]")
    if (float(saved_step) != 1. or override is not None) and list(modes) != ["reference"]:
        raise ValueError("Native nonunit step/override requires reference-only evaluation; do not double-scale")
    return step


def _accumulate(stats, key, value):
    value = float(value)
    if not math.isfinite(value):
        raise RuntimeError(f"Non-finite diagnostic {key}: {value}")
    current = stats.setdefault(key, {"count": 0, "sum": 0., "min": value, "max": value})
    current["count"] += 1
    current["sum"] += value
    current["min"] = min(current["min"], value)
    current["max"] = max(current["max"], value)


def _summarize(stats):
    return {
        key: {"count": value["count"], "mean": value["sum"] / value["count"],
              "min": value["min"], "max": value["max"]}
        for key, value in stats.items()
    }


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if len(set(args.modes)) != len(args.modes):
        parser.error("--modes must be distinct")
    if "reference" in args.modes and args.modes[0] != "reference":
        parser.error("reference must be first when requested")
    if len(set(args.rates)) != len(args.rates) or any(
        rate not in (0., .1, .3, .5, .7) for rate in args.rates
    ):
        parser.error("--rates must be distinct official rates: 0.0 0.1 0.3 0.5 0.7")

    import numpy as np
    import torch
    from . import train_gcnet as tr
    from .memory_retention import MemoryRetentionDiagnostics
    from .model import MissingM3GraphModel
    from .write_intervention import WriteIntervention

    torch.set_num_threads(2)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = tr.TrainConfig(**checkpoint["config"])
    saved_write_step = float(getattr(cfg, "osram_write_step", 1.))
    evaluation_write_step = resolve_evaluation_step(
        saved_write_step, args.evaluation_write_step, args.modes)
    if cfg.backbone_type != "osram" or cfg.osram_bidirectional or cfg.osram_forward_slot_reuse:
        raise ValueError("Requires unidirectional OSRAM without forward slot reuse")
    tr.set_random_seed(cfg.seed)
    shape = tr._resolve_task_contract(cfg.dataset, cfg.mosi_task_mode)
    paths = [str(args.feature_root / name) for name in
             ("wav2vec-large-c-UTT", "deberta-large-4-UTT", "manet_UTT")]
    _, _, test, adim, tdim, vdim = tr.get_loaders(
        audio_root=paths[0], text_root=paths[1], video_root=paths[2],
        num_folder=int(shape["num_folds"]), dataset=cfg.dataset,
        batch_size=cfg.batch_size, num_workers=0, seed=cfg.seed,
        validation_fraction=cfg.validation_fraction, evaluation_protocol=cfg.evaluation_protocol,
    )
    settings = asdict(cfg)
    names = inspect.signature(MissingM3GraphModel.__init__).parameters
    kwargs = {key: value for key, value in settings.items() if key in names}
    kwargs.update(adim=adim, tdim=tdim, vdim=vdim, D_e=cfg.hidden,
                  graph_hidden_size=cfg.hidden // 2, n_speakers=int(shape["num_speakers"]),
                  n_classes=int(shape["num_classes"]), time_attn=cfg.time_attention,
                  no_cuda=args.device == "cpu")
    device = torch.device(args.device)
    model = MissingM3GraphModel(**kwargs).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    # Plain runtime attribute only: keep saved config and all learned weights intact.
    model.osram.write_step = evaluation_write_step
    initial_state = {key: value.detach().cpu().clone()
                     for key, value in model.state_dict().items()}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for rate in args.rates:
        reference_logits = []
        reference_digest = None
        for mode in args.modes:
            # Fresh, frozen official schedule makes the paired mask contract explicit.
            schedule = tr._build_schedule(cfg, "test", rate)
            mask_digest = hashlib.sha256()
            predictions, labels = [], []
            retention_stats = defaultdict(dict)
            retention_counts = defaultdict(lambda: defaultdict(int))
            observed_stats = defaultdict(dict)
            observed_counts = defaultdict(int)
            audit_stats = {}
            audit_count = protected_count = batch_count = 0
            stem = f"rate{rate:g}_{mode}"
            with gzip.open(args.output_dir / f"{stem}_retention.jsonl.gz", "wt", encoding="utf-8") as stream, \
                 gzip.open(args.output_dir / f"{stem}_writes.jsonl.gz", "wt", encoding="utf-8") as writes, \
                 gzip.open(args.output_dir / f"{stem}_observed_fit.jsonl.gz", "wt", encoding="utf-8") as fits, \
                 torch.no_grad(), WriteIntervention(model.osram, mode) as audit:
                def sink(row):
                    stream.write(json.dumps(dict(row, mode=mode)) + "\n")
                    target = row["target_modality"]
                    retention_counts[target][row["status"]] += 1
                    if row["status"] == "retention":
                        for metric in RETENTION_METRICS:
                            _accumulate(retention_stats[target], metric, row[metric])

                for batch_index, raw in enumerate(test[cfg.fold - 1]):
                    data = tr._move_batch(raw, device)
                    view = tr._prepare_view(data, schedule, epoch=0, dimensions=(adim, tdim, vdim))
                    mask_digest.update(view["availability"].detach().cpu().numpy().tobytes())
                    sample_ids = [f"seed{cfg.seed}:{item}" for item in data[-1]]
                    collector = MemoryRetentionDiagnostics(
                        sink, dataset=cfg.dataset, missing_rate=rate, sample_ids=sample_ids,
                    )

                    def inject(module, inputs, hook_kwargs):
                        return inputs, dict(hook_kwargs, collect_memory_retention_diagnostics=True,
                                            memory_retention_diagnostics=collector)

                    audit.records.clear()
                    audit.observed_records.clear()
                    handle = model.osram.register_forward_pre_hook(inject, with_kwargs=True)
                    try:
                        output = model([view["incomplete"]], view["availability"], view["qmask"],
                                       view["umask"], view["lengths"], predict_missing=False)
                    finally:
                        handle.remove()
                    logits = output[0]
                    if output[3] is not None:
                        raise RuntimeError("Inference returned missing predictions")
                    if rate == 0. and "reference" in args.modes and mode in IDENTITY_MODES:
                        if mode == "reference":
                            reference_logits.append(logits.detach().cpu().clone())
                        elif batch_index >= len(reference_logits) or not torch.equal(
                            reference_logits[batch_index], logits.detach().cpu()
                        ):
                            raise RuntimeError(f"Complete-input logits differ for {mode}, batch {batch_index}")
                    predicted, expected, _ = tr._collect_predictions(
                        cfg.dataset, logits, view["labels"], view["umask"], cfg.mosi_task_mode,
                    )
                    predictions.append(predicted)
                    labels.append(expected)
                    for record in audit.records:
                        row = dict(record)
                        row["norm_mismatch"] = abs(row["protected_norm"] - row["global_norm"])
                        writes.write(json.dumps(dict(row, mode=mode, missing_rate=rate,
                                                     loader_batch_index=batch_index,
                                                     sample_id=sample_ids[row["batch_index"]])) + "\n")
                        audit_count += 1
                        protected_count += int(row["has_protection"])
                        for metric in AUDIT_METRICS:
                            _accumulate(audit_stats, metric, row[metric])
                    audit.records.clear()
                    for row in audit.observed_records:
                        fits.write(json.dumps(dict(row, mode=mode, missing_rate=rate,
                                                   loader_batch_index=batch_index,
                                                   sample_id=sample_ids[row["batch_index"]])) + "\n")
                        target = row["target_modality"]
                        observed_counts[target] += 1
                        for metric in WRITE_FIT_METRICS:
                            _accumulate(observed_stats[target], metric, row[metric])
                    audit.observed_records.clear()
                    batch_count += 1
                    print(f"rate={rate:g} mode={mode} batch={batch_index} writes={audit_count}", flush=True)
            if not predictions:
                raise RuntimeError("Empty test loader")
            digest = mask_digest.hexdigest()
            if mode == args.modes[0]:
                reference_digest = digest
            elif digest != reference_digest:
                raise RuntimeError(f"Mask digest differs for rate={rate}, mode={mode}")
            if rate == 0. and "reference" in args.modes and batch_count != len(reference_logits):
                raise RuntimeError("Complete-input batch counts differ")
            result = dict(
                mode=mode, rate=rate, batches=batch_count, mask_sha256=digest,
                task_metrics=tr._metrics(cfg.dataset, np.concatenate(labels),
                                         np.concatenate(predictions), cfg.mosi_task_mode),
                retention={target: dict(counts=dict(retention_counts[target]),
                                        metrics=_summarize(retention_stats[target]))
                           for target in sorted(retention_counts)},
                write_audit=dict(records=audit_count, with_protection=protected_count,
                                 metrics=_summarize(audit_stats)),
                current_observed_write_fit={
                    target: dict(counts=observed_counts[target], metrics=_summarize(observed_stats[target]))
                    for target in sorted(observed_counts)
                },
                complete_input_logits_exact=True if rate == 0. and "reference" in args.modes
                                            and mode in IDENTITY_MODES else None,
            )
            results.append(result)
            (args.output_dir / f"{stem}_summary.json").write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8")

    final_state = model.state_dict()
    if final_state.keys() != initial_state.keys() or any(
        not torch.equal(initial_state[key], value.detach().cpu())
        for key, value in final_state.items()
    ):
        raise RuntimeError("Model state changed during evaluation")
    metadata = dict(
        checkpoint=str(args.checkpoint), checkpoint_sha256=tr._sha256_file(args.checkpoint),
        config=checkpoint["config"], epoch=checkpoint["epoch"],
        selection_protocol=checkpoint.get("selection_protocol"),
        dataset=cfg.dataset, seed=cfg.seed, fold=cfg.fold, rates=args.rates,
        evaluation_only=True, new_checkpoint_selection=False, weights_unchanged=True,
        training_write_step=saved_write_step,
        evaluation_write_step=evaluation_write_step,
        evaluation_write_step_override=args.evaluation_write_step,
        all_mode_masks_equal=True, complete_input_logits_exact=True if 0. in args.rates
                                                             and "reference" in args.modes else None,
        complete_input_identity_modes=[m for m in args.modes if m in IDENTITY_MODES]
                                      if "reference" in args.modes else [],
        modes=args.modes,
        fixed_global_strength=0.9 if "fixed0.9" in args.modes else None,
        fixed_global_strengths={m: v for m, v in (("fixed0.95", .95), ("fixed0.9", .9),
                                                ("fixed0.8", .8), ("fixed0.6", .6),
                                                ("fixed0.4", .4), ("fixed0.2", .2),
                                                ("fixed0.0", 0.)) if m in args.modes},
        ridge=.001, device=str(device), results=results,
        retention_note="Head-level means; err_post probes future retention, not current prediction.",
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
