"""P1: frozen matched content replacement using the exact P0 source protocol."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

from experiments.osram_history_effect_stratification_20261013 import evaluate as p0
from experiments.osram_history_content_intervention_20261014.intervention import (
    ROW_FIELDS, evaluate_view, summarize_rows,
)

ROOT = Path(__file__).resolve().parents[2]


def rows_to_arrays(rows, classes):
    strings = {"conversation_id", "modality", "arm"}
    booleans = {"source_same_speaker", "donor_same_speaker", "correction", "harm", "current_local_exact",
                "current_local_within_tolerance", "current_availability_exact", "availability_trajectory_exact", "current_input_exact"}
    integers = {"position", "source_position", "donor_position", "source_lag", "donor_lag", "lag_mismatch",
                "target_speaker", "source_speaker", "donor_speaker", "label"}
    result = {}
    for field in ROW_FIELDS:
        dtype = str if field in strings else bool if field in booleans else np.int64 if field in integers else np.float64
        result[field] = np.asarray([row[field] for row in rows], dtype=dtype)
        if field.startswith("logits_"):
            result[field] = result[field].reshape(len(rows), classes)
    return result


def evaluate_rate(model, loader, config, dimensions, schedule, source_prediction=None, *,
                  source_missing_mask_protocol=None, chunk_size=16):
    from gcnet_missing_m3.train_gcnet import _move_batch, _prepare_view, _prepare_view_from_primary_masks

    # P0 proves frozen original/source parity and its exact missing-mask replay.
    original_summary, original_artifacts = p0.evaluate_rate(
        model, loader, config, dimensions, schedule, source_prediction,
        source_missing_mask_protocol=source_missing_mask_protocol,
    )
    rows, skipped, original_parts, batch_checks = [], [], [], []
    for raw in loader:
        data = _move_batch(raw, next(model.parameters()).device)
        if config.dataset == "MELD" and source_missing_mask_protocol == "gcnet-legacy-fixed-v1":
            host, guest = p0.materialize_legacy_masks(schedule, data[-1], data[7], epoch=0)
            view = _prepare_view_from_primary_masks(data, host, guest, dimensions)
        else:
            view = _prepare_view(data, schedule, epoch=0, dimensions=dimensions)
        batch_rows, batch_skipped, checks, logits = evaluate_view(model, view, dimensions, chunk_size=chunk_size)
        rows.extend(batch_rows)
        skipped.extend(batch_skipped)
        batch_checks.append(checks)
        original_parts.append(logits)
    np.testing.assert_array_equal(np.concatenate(original_parts), original_artifacts["logits_LBG"])
    arrays = rows_to_arrays(rows, original_artifacts["logits_LBG"].shape[1])
    checks = {name: sum(batch[name] for batch in batch_checks) for name in batch_checks[0]}
    checks.update(p0_original_logits_exact=True, current_local_rtol=2e-5, current_local_atol=2e-5,
                  current_local_exact_count=int(arrays["current_local_exact"].sum()),
                  current_local_tolerance_all=bool(arrays["current_local_within_tolerance"].all()),
                  current_local_max_abs_error=float(arrays["current_local_max_abs_error"].max()) if rows else 0.,
                  current_availability_exact_all=bool(arrays["current_availability_exact"].all()),
                  availability_trajectory_exact_all=bool(arrays["availability_trajectory_exact"].all()),
                  current_input_exact_all=bool(arrays["current_input_exact"].all()))
    return dict(checks=checks, modalities=summarize_rows(rows, skipped), original_p0=original_summary), arrays, skipped, original_artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--meld-pickle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--rates", type=float, nargs="+", default=[i/10 for i in range(8)])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.chunk_size < 1 or not args.rates or len(set(args.rates)) != len(args.rates) or any(rate not in {i/10 for i in range(8)} for rate in args.rates):
        raise ValueError("positive chunk size and unique rates on 0.0--0.7 grid required")
    source = json.loads((args.source / "config.json").read_text())
    p0.validate_source(source)
    protocol = source.get("missing_mask_protocol", "conversation-mask-v2")
    if source["dataset"] == "MELD" and (args.meld_pickle is None or not args.meld_pickle.is_file()):
        raise ValueError("MELD requires source --meld-pickle")
    for rate in args.rates:
        suffix = f"{rate:.1f}".replace(".", "p")
        for name in ([f"best_miss_{suffix}.pt", f"predictions_miss_{suffix}.npz"] if source["dataset"] != "MELD" else [f"best_miss_{suffix}.pt"]):
            if not (args.source / name).is_file():
                raise FileNotFoundError(args.source / name)
    os.environ["GCNET_DATASET_ROOT"] = str(args.dataset_root)
    if args.meld_pickle:
        os.environ["GCNET_MELD_PICKLE"] = str(args.meld_pickle)
    from gcnet_missing_m3.train_gcnet import TrainConfig, _dataset_shape, _schedules
    from gcnet_modality_jepa.protocol import SeedBundle
    from gcnet_modality_jepa.train_gcnet import get_loaders
    from experiments.osram_meld_flat_source_combinations_20261013.legacy_schedule import LegacyFixedMaskSchedule

    config = replace(TrainConfig(**p0._runtime_config(p0.runtime_config_payload(source))), device=args.device)
    torch.set_num_threads(2)
    feature_root = args.dataset_root / ("MELD" if config.dataset == "MELD" else "IEMOCAP") / "features"
    roots = [str(feature_root/name) for name in ("wav2vec-large-c-UTT", "deberta-large-4-UTT", "manet_UTT")]
    _, _, test, ad, td, vd = get_loaders(audio_root=roots[0], text_root=roots[1], video_root=roots[2],
        num_folder=int(_dataset_shape(config.dataset)["num_folds"]), dataset=config.dataset,
        batch_size=config.batch_size, num_workers=0, seed=config.seed,
        validation_fraction=config.validation_fraction, evaluation_protocol=config.evaluation_protocol)
    model = p0._build_model(config, (ad, td, vd), torch.device(args.device))
    if config.dataset == "MELD" and protocol == "gcnet-legacy-fixed-v1":
        schedules = {rate: LegacyFixedMaskSchedule(config.dataset, "test", config.fold, rate,
            SeedBundle(config.seed).derive("missing_mask"), freeze_evaluation=True) for rate in args.rates}
    else:
        schedules = _schedules(config, "test")
    args.output.mkdir(parents=True)
    p0._write(args.output/"runtime_config.json", asdict(config))
    provenance = p0.code_provenance()
    provenance["p1_sha256"] = {str(path.relative_to(ROOT)): p0._sha256(path) for path in Path(__file__).parent.glob("*.py")}
    p0._write(args.output/"code_provenance.json", provenance)
    summary = dict(status="running", training=False, diagnostic="P1_history_content", dataset=config.dataset,
                   seed=config.seed, fold=config.fold, source=str(args.source),
                   source_config_sha256=p0._sha256(args.source/"config.json"), source_missing_mask_protocol=protocol,
                   protocol="INTERNAL DIAGNOSTIC ONLY; frozen per-rate source Test-oracle checkpoints; no training",
                   source_rule="latest strictly-prior same-target-speaker observed source per modality",
                   donor_rule="strict-past observed modality; nearest source lag, ties latest; same/other target speaker; labels/predictions never used",
                   delta_definition="delta_ce=CE(replaced)-CE(original), positive means harmful; true_probability_delta=P(replaced)-P(original)",
                   chunk_size=args.chunk_size, rates={}, command=sys.argv,
                   environment=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES")))
    if args.meld_pickle:
        summary.update(dataset_pickle=str(args.meld_pickle), dataset_pickle_sha256=p0._sha256(args.meld_pickle))
    p0._write(args.output/"SUMMARY.json", summary)
    try:
        for rate in args.rates:
            rate_name = f"{rate:.1f}"
            suffix = rate_name.replace(".", "p")
            checkpoint = args.source/f"best_miss_{suffix}.pt"
            prediction = args.source/f"predictions_miss_{suffix}.npz"
            checkpoint_sha256 = p0._sha256(checkpoint)
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(state["model"], strict=True)
            result, artifacts, skipped, originals = evaluate_rate(model, test[config.fold-1], config, (ad, td, vd), schedules[rate],
                prediction if prediction.is_file() else None, source_missing_mask_protocol=protocol, chunk_size=args.chunk_size)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value.detach().cpu(), state["model"][name], rtol=0, atol=0)
            if p0._sha256(checkpoint) != checkpoint_sha256:
                raise AssertionError("source checkpoint changed during P1 evaluation")
            result["checks"].update(model_state_exact=True, checkpoint_sha256_unchanged=True)
            filename = f"interventions_miss_{suffix}.npz"
            original_name = f"p0_utterances_miss_{suffix}.npz"
            skipped_name = f"skipped_miss_{suffix}.json"
            np.savez_compressed(args.output/filename, **artifacts)
            np.savez_compressed(args.output/original_name, **originals)
            p0._write(args.output/skipped_name, skipped)
            summary["rates"][rate_name] = dict(checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha256, epoch=state.get("epoch"),
                intervention_file=filename, original_file=original_name, skipped_file=skipped_name, **result)
            p0._write(args.output/"SUMMARY.json", summary)
            print(f"{config.dataset} seed={config.seed} fold={config.fold} rate={rate_name} pairs={len(artifacts['label'])}", flush=True)
        summary["status"] = "complete"
    except BaseException as error:
        summary.update(status="failed", error=repr(error))
        raise
    finally:
        p0._write(args.output/"SUMMARY.json", summary)


if __name__ == "__main__":
    main()
