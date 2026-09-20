"""Frozen causal OSRAM diagnostic for swapping simultaneously active Gap queries."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from dataclasses import asdict
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--feature-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rates", nargs="+", type=float, default=[0.1, 0.3, 0.5, 0.7])
    parser.add_argument("--device", default="cpu")
    return parser


def _trajectory_equal(left, right, torch):
    if len(left) != len(right):
        return False
    for left_item, right_item in zip(left, right):
        if left_item[0] != right_item[0]:
            return False
        if not torch.equal(left_item[1], right_item[1]):
            return False
        if not torch.equal(left_item[2], right_item[2]):
            return False
    return True


def _metric_or_none(train_module, dataset, labels, predictions, task_mode):
    if labels.size == 0:
        return None
    return train_module._metrics(dataset, labels, predictions, task_mode)


def _metric_delta(reference, swapped):
    if reference is None or swapped is None:
        return None
    keys = sorted(set(reference).intersection(swapped))
    return {
        key: float(swapped[key] - reference[key])
        for key in keys
        if isinstance(reference[key], (int, float))
        and isinstance(swapped[key], (int, float))
    }


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    allowed_rates = (0.0, 0.1, 0.3, 0.5, 0.7)
    if len(set(args.rates)) != len(args.rates) or any(
        rate not in allowed_rates for rate in args.rates
    ):
        parser.error("--rates must be distinct official rates: 0.0 0.1 0.3 0.5 0.7")

    import numpy as np
    import torch

    from . import train_gcnet as tr
    from .gap_query_swap import GapQuerySwap, active_gap_swap_mask
    from .model import MissingM3GraphModel

    torch.set_num_threads(2)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = tr.TrainConfig(**checkpoint["config"])
    if cfg.backbone_type != "osram" or cfg.osram_bidirectional or cfg.osram_forward_slot_reuse:
        raise ValueError("requires causal OSRAM without forward slot reuse")
    tr.set_random_seed(cfg.seed)
    shape = tr._resolve_task_contract(cfg.dataset, cfg.mosi_task_mode)
    paths = [
        str(args.feature_root / name)
        for name in ("wav2vec-large-c-UTT", "deberta-large-4-UTT", "manet_UTT")
    ]
    _, _, test, adim, tdim, vdim = tr.get_loaders(
        audio_root=paths[0],
        text_root=paths[1],
        video_root=paths[2],
        num_folder=int(shape["num_folds"]),
        dataset=cfg.dataset,
        batch_size=cfg.batch_size,
        num_workers=0,
        seed=cfg.seed,
        validation_fraction=cfg.validation_fraction,
        evaluation_protocol=cfg.evaluation_protocol,
    )
    settings = asdict(cfg)
    names = inspect.signature(MissingM3GraphModel.__init__).parameters
    kwargs = {key: value for key, value in settings.items() if key in names}
    kwargs.update(
        adim=adim,
        tdim=tdim,
        vdim=vdim,
        D_e=cfg.hidden,
        graph_hidden_size=cfg.hidden // 2,
        n_speakers=int(shape["num_speakers"]),
        n_classes=int(shape["num_classes"]),
        time_attn=cfg.time_attention,
        no_cuda=args.device == "cpu",
    )
    device = torch.device(args.device)
    model = MissingM3GraphModel(**kwargs).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    initial_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)

    results = []
    for rate in args.rates:
        schedule = tr._build_schedule(cfg, "test", rate)
        mask_digest = hashlib.sha256()
        records_path = args.output_dir / f"rate{rate:g}_affected.jsonl"
        reference_predictions = []
        swapped_predictions = []
        reference_labels = []
        swapped_labels = []
        reference_affected_predictions = []
        swapped_affected_predictions = []
        reference_affected_labels = []
        swapped_affected_labels = []
        reference_unaffected_predictions = []
        swapped_unaffected_predictions = []
        reference_unaffected_labels = []
        swapped_unaffected_labels = []
        valid_count = 0
        affected_count = 0
        changed_count = 0
        affected_changed_count = 0
        abs_logit_sum = 0.0
        affected_abs_logit_sum = 0.0
        max_abs_logit_delta = 0.0
        snapshot_equal = True
        with records_path.open("w", encoding="utf-8") as records:
            for batch_index, raw in enumerate(test[cfg.fold - 1]):
                data = tr._move_batch(raw, device)
                view = tr._prepare_view(
                    data, schedule, epoch=0, dimensions=(adim, tdim, vdim)
                )
                availability = view["availability"]
                mask_digest.update(availability.detach().cpu().numpy().tobytes())
                sample_ids = [str(item) for item in data[-1]]
                with torch.no_grad(), GapQuerySwap(
                    model.osram, swap_queries=False, capture_memory=True
                ) as reference_probe:
                    reference_output = model(
                        [view["incomplete"]],
                        view["availability"],
                        view["qmask"],
                        view["umask"],
                        view["lengths"],
                        predict_missing=False,
                    )
                with torch.no_grad(), GapQuerySwap(
                    model.osram, swap_queries=True, capture_memory=True
                ) as swapped_probe:
                    swapped_output = model(
                        [view["incomplete"]],
                        view["availability"],
                        view["qmask"],
                        view["umask"],
                        view["lengths"],
                        predict_missing=False,
                    )
                if not _trajectory_equal(
                    reference_probe.memory_snapshots,
                    swapped_probe.memory_snapshots,
                    torch,
                ):
                    snapshot_equal = False
                    raise RuntimeError(
                        f"Gap query swap changed persistent memory at rate={rate:g}, "
                        f"batch={batch_index}"
                    )
                reference_logits = reference_output[0].detach().cpu()
                swapped_logits = swapped_output[0].detach().cpu()
                if reference_logits.shape != swapped_logits.shape:
                    raise RuntimeError("reference and swapped logits have different shapes")
                diff = (swapped_logits - reference_logits).squeeze(-1).T
                valid = view["umask"].bool().detach().cpu()
                affected = active_gap_swap_mask(availability).T.detach().cpu() & valid
                flat_affected = affected[valid].numpy()
                valid_count += int(valid.sum())
                affected_count += int(affected.sum())
                changed = diff.abs().gt(1e-8) & valid
                affected_changed = changed & affected
                changed_count += int(changed.sum())
                affected_changed_count += int(affected_changed.sum())
                abs_logit_sum += float(diff.abs()[valid].sum())
                affected_abs_logit_sum += float(diff.abs()[affected].sum())
                max_abs_logit_delta = max(max_abs_logit_delta, float(diff.abs().max()))

                ref_pred, ref_expected, ref_raw = tr._collect_predictions(
                    cfg.dataset,
                    reference_output[0],
                    view["labels"],
                    view["umask"],
                    cfg.mosi_task_mode,
                )
                swap_pred, swap_expected, swap_raw = tr._collect_predictions(
                    cfg.dataset,
                    swapped_output[0],
                    view["labels"],
                    view["umask"],
                    cfg.mosi_task_mode,
                )
                if not np.array_equal(ref_expected, swap_expected) or not np.array_equal(
                    ref_raw, swap_raw
                ):
                    raise RuntimeError("Reference and swapped labels differ")
                reference_predictions.append(ref_pred)
                swapped_predictions.append(swap_pred)
                reference_labels.append(ref_expected)
                swapped_labels.append(swap_expected)
                reference_affected_predictions.append(ref_pred[flat_affected])
                swapped_affected_predictions.append(swap_pred[flat_affected])
                reference_affected_labels.append(ref_expected[flat_affected])
                swapped_affected_labels.append(swap_expected[flat_affected])
                reference_unaffected_predictions.append(ref_pred[~flat_affected])
                swapped_unaffected_predictions.append(swap_pred[~flat_affected])
                reference_unaffected_labels.append(ref_expected[~flat_affected])
                swapped_unaffected_labels.append(swap_expected[~flat_affected])

                ref_matrix = reference_logits.squeeze(-1).T
                swap_matrix = swapped_logits.squeeze(-1).T
                for batch_pos, time_pos in zip(*valid.nonzero(as_tuple=True)):
                    if not bool(affected[batch_pos, time_pos]):
                        continue
                    row = {
                        "rate": rate,
                        "loader_batch_index": batch_index,
                        "sample_id": sample_ids[int(batch_pos)],
                        "time_index": int(time_pos),
                        "availability": availability[
                            time_pos, batch_pos
                        ].detach().cpu().int().tolist(),
                        "label": float(view["labels"][batch_pos, time_pos].detach().cpu()),
                        "reference_prediction": float(ref_matrix[batch_pos, time_pos]),
                        "swapped_prediction": float(swap_matrix[batch_pos, time_pos]),
                        "prediction_delta": float(
                            swap_matrix[batch_pos, time_pos]
                            - ref_matrix[batch_pos, time_pos]
                        ),
                    }
                    records.write(json.dumps(row) + "\n")
                print(
                    f"rate={rate:g} batch={batch_index} affected={int(affected.sum())}",
                    flush=True,
                )

        ref_predictions = np.concatenate(reference_predictions)
        swap_predictions = np.concatenate(swapped_predictions)
        ref_labels = np.concatenate(reference_labels)
        swap_labels = np.concatenate(swapped_labels)
        ref_affected_predictions = np.concatenate(reference_affected_predictions)
        swap_affected_predictions = np.concatenate(swapped_affected_predictions)
        ref_affected_labels = np.concatenate(reference_affected_labels)
        swap_affected_labels = np.concatenate(swapped_affected_labels)
        ref_unaffected_predictions = np.concatenate(reference_unaffected_predictions)
        swap_unaffected_predictions = np.concatenate(swapped_unaffected_predictions)
        ref_unaffected_labels = np.concatenate(reference_unaffected_labels)
        swap_unaffected_labels = np.concatenate(swapped_unaffected_labels)

        reference = {
            "overall": _metric_or_none(
                tr, cfg.dataset, ref_labels, ref_predictions, cfg.mosi_task_mode
            ),
            "two_missing": _metric_or_none(
                tr,
                cfg.dataset,
                ref_affected_labels,
                ref_affected_predictions,
                cfg.mosi_task_mode,
            ),
            "other": _metric_or_none(
                tr,
                cfg.dataset,
                ref_unaffected_labels,
                ref_unaffected_predictions,
                cfg.mosi_task_mode,
            ),
        }
        swapped = {
            "overall": _metric_or_none(
                tr, cfg.dataset, swap_labels, swap_predictions, cfg.mosi_task_mode
            ),
            "two_missing": _metric_or_none(
                tr,
                cfg.dataset,
                swap_affected_labels,
                swap_affected_predictions,
                cfg.mosi_task_mode,
            ),
            "other": _metric_or_none(
                tr,
                cfg.dataset,
                swap_unaffected_labels,
                swap_unaffected_predictions,
                cfg.mosi_task_mode,
            ),
        }
        result = {
            "rate": rate,
            "batches": len(test[cfg.fold - 1]),
            "mask_sha256": mask_digest.hexdigest(),
            "valid_utterances": valid_count,
            "two_missing_utterances": affected_count,
            "two_missing_fraction": affected_count / max(valid_count, 1),
            "query_swap_rows": affected_count,
            "memory_trajectory_equal": snapshot_equal,
            "reference": reference,
            "swapped": swapped,
            "delta_swapped_minus_reference": {
                key: _metric_delta(reference[key], swapped[key])
                for key in reference
            },
            "logit_change": {
                "valid_changed_count": changed_count,
                "valid_changed_fraction": changed_count / max(valid_count, 1),
                "two_missing_changed_count": affected_changed_count,
                "two_missing_changed_fraction": affected_changed_count / max(affected_count, 1),
                "mean_abs_delta_valid": abs_logit_sum / max(valid_count, 1),
                "mean_abs_delta_two_missing": affected_abs_logit_sum / max(affected_count, 1),
                "max_abs_delta": max_abs_logit_delta,
            },
        }
        results.append(result)
        (args.output_dir / f"rate{rate:g}_summary.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )

    final_state = model.state_dict()
    if final_state.keys() != initial_state.keys() or any(
        not torch.equal(initial_state[key], value.detach().cpu())
        for key, value in final_state.items()
    ):
        raise RuntimeError("Model state changed during evaluation")
    metadata = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": tr._sha256_file(args.checkpoint),
        "config": checkpoint["config"],
        "epoch": checkpoint["epoch"],
        "selection_protocol": checkpoint.get("selection_protocol"),
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "fold": cfg.fold,
        "rates": args.rates,
        "evaluation_only": True,
        "new_checkpoint_selection": False,
        "weights_unchanged": True,
        "all_mode_masks_equal": True,
        "memory_trajectory_equal": all(item["memory_trajectory_equal"] for item in results),
        "query_swap": "exchange the two missing-modality queries only when exactly one modality is observed",
        "modes": ["reference", "swapped-gap-query"],
        "device": str(device),
        "results": results,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
