"""Evaluation-only fixed-modality ablation for the original MOSI cfg84 model."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

REMOTE = Path("/data2/yb/remote_experiments/osram_cfg84_fixed_modality_ablation_20260923")
CHECKPOINT_ROOT = Path("/data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full")
FEATURE_ROOT = Path("/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset/CMUMOSI/features")
SEEDS = (66, 67, 68, 69, 70)
GPUS = (4, 5, 7)
PATTERNS = {
    "A": (1, 0, 0),
    "L": (0, 1, 0),
    "V": (0, 0, 1),
    "AL": (1, 1, 0),
    "AV": (1, 0, 1),
    "LV": (0, 1, 1),
    "ALV": (1, 1, 1),
}
# Use the already selected checkpoint whose official rate is closest to the
# exact fixed missing fraction. This does not select a checkpoint on the new
# fixed-pattern scores.
CHECKPOINT_RATE = {
    "A": "0.7", "L": "0.7", "V": "0.7",
    "AL": "0.3", "AV": "0.3", "LV": "0.3",
    "ALV": "0.0",
}


def _roots() -> tuple[str, str, str]:
    return tuple(str(FEATURE_ROOT / name) for name in (
        "wav2vec-large-c-UTT", "deberta-large-4-UTT", "manet_UTT"
    ))


def _build_model(config, dimensions):
    """Construct the exact MOSI cfg84 model stored in the source checkpoint."""
    from gcnet_modality_jepa.protocol import SeedBundle
    from gcnet_modality_jepa.train_gcnet import set_random_seed
    from gcnet_missing_m3.model import MissingM3GraphModel

    adim, tdim, vdim = dimensions
    set_random_seed(SeedBundle(config.seed).derive("missing_m3_model_init:fold:5"))
    return MissingM3GraphModel(
        config.base_model, adim, tdim, vdim, config.hidden, config.hidden // 2,
        n_speakers=1, window_past=config.window_past,
        window_future=config.window_future, n_classes=1, dropout=config.dropout,
        time_attn=config.time_attention, no_cuda=config.device != "cuda",
        latent_dim=config.latent_dim, num_experts=config.num_experts,
        top_k=config.top_k, projector_dropout=config.projector_dropout,
        predictor_dropout=config.predictor_dropout, fusion_type=config.fusion_type,
        local_context_residual=config.local_context_residual,
        local_fusion_hidden_dim=config.local_fusion_hidden_dim,
        local_fusion_dropout=config.local_fusion_dropout,
        graph_branch_mode=config.graph_branch_mode,
        mmoe_variant=config.mmoe_variant,
        target_private_rank=config.target_private_rank,
        classification_completion=config.classification_completion,
        representation_type=config.representation_type,
        node_interaction_residual=config.node_interaction_residual,
        readout_type=config.readout_type, readout_rank=config.readout_rank,
        recurrent_padding_mode=config.recurrent_padding_mode,
        postgraph_sequence_mode=config.postgraph_sequence_mode,
        graph_message_calibration=config.graph_message_calibration,
        graph_second_layer=config.graph_second_layer,
        postgraph_bilstm_ablation=config.postgraph_bilstm_ablation,
        backbone_type=config.backbone_type,
        osram_output_dim=config.osram_output_dim,
        osram_num_heads=config.osram_num_heads,
        osram_key_dim=config.osram_key_dim,
        osram_value_dim=config.osram_value_dim,
        osram_read_ridge=config.osram_read_ridge,
        osram_write_ridge=config.osram_write_ridge,
        osram_predictor_mode=config.osram_predictor_mode,
        osram_ablation=config.osram_ablation,
        osram_gap_read=config.osram_gap_read,
        gap_residual_strength=config.gap_residual_strength,
        beta_mode=config.beta_mode,
        osram_emotion_ablation=config.osram_emotion_ablation,
        osram_query_availability=config.osram_query_availability,
        osram_bidirectional=config.osram_bidirectional,
        osram_forward_slot_reuse=config.osram_forward_slot_reuse,
        osram_write_step=config.osram_write_step,
        completion_path=config.completion_path,
        osram_readout_fusion=config.osram_readout_fusion,
        teacher_mode=config.teacher_mode,
        teacher_checkpoint=config.teacher_checkpoint,
        target_space=config.target_space,
        text_subspace_checkpoint=config.text_subspace_checkpoint,
        training_objective=config.training_objective,
        text_core=config.text_core,
        disable_unused_aux_modules=config.disable_unused_aux_modules,
        simple_regression_predictor=config.simple_regression_predictor,
    )


def _evaluate_pattern(model, loader, pattern, dimensions, device):
    import torch
    from gcnet_missing_m3.train_gcnet import (
        _collect_predictions,
        _metrics,
        _move_batch,
        _prepare_view_from_primary_masks,
    )

    predictions = []
    labels = []
    model.eval()
    with torch.no_grad():
        for raw in loader:
            data = _move_batch(raw, device)
            umask = data[7]
            batch, sequence = umask.shape
            availability = torch.zeros(
                sequence, batch, 3, dtype=torch.uint8, device=device
            )
            fixed = torch.tensor(pattern, dtype=torch.uint8, device=device)
            availability[umask.T.bool()] = fixed
            view = _prepare_view_from_primary_masks(
                data, availability, availability, dimensions
            )
            logits, _, _, missing = model(
                [view["incomplete"]], view["availability"], view["qmask"],
                view["umask"], view["lengths"], predict_missing=False,
            )
            if missing is not None:
                raise RuntimeError("no-JEPA inference unexpectedly returned predictions")
            predicted, expected, _ = _collect_predictions(
                "CMUMOSI", logits, view["labels"], view["umask"], "regression"
            )
            predictions.append(predicted)
            labels.append(expected)
    predicted = np.concatenate(predictions)
    expected = np.concatenate(labels)
    result = _metrics("CMUMOSI", expected, predicted, "regression")
    result["sample_count"] = int(expected.size)
    result["nonzero_sample_count"] = int(np.count_nonzero(expected))
    return result


def evaluate_seed(seed: int) -> None:
    import torch
    from gcnet_modality_jepa.train_gcnet import get_loaders
    from gcnet_missing_m3.train_gcnet import TrainConfig

    source = CHECKPOINT_ROOT / f"seed_{seed}"
    output = REMOTE / f"seed_{seed}.json"
    if output.exists():
        print(f"SKIP seed={seed}", flush=True)
        return
    config = TrainConfig(**json.loads((source / "config.json").read_text()))
    if config.train_rate_mode != "cyclic" or tuple(config.train_missing_rates) != tuple(i / 10 for i in range(8)):
        raise ValueError("source is not the original all-rate cyclic cfg84 model")
    device = torch.device("cuda:0")
    loaders = get_loaders(
        audio_root=_roots()[0], text_root=_roots()[1], video_root=_roots()[2],
        num_folder=1, dataset="CMUMOSI", batch_size=config.batch_size,
        num_workers=0, seed=seed, validation_fraction=config.validation_fraction,
        evaluation_protocol=config.evaluation_protocol,
    )
    _, _, test_loaders, adim, tdim, vdim = loaders
    dimensions = (adim, tdim, vdim)
    model = _build_model(config, dimensions).to(device)
    rows = {}
    for name, pattern in PATTERNS.items():
        rate = CHECKPOINT_RATE[name]
        checkpoint = source / f"best_miss_{rate.replace('.', 'p')}.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        result = _evaluate_pattern(
            model, test_loaders[0], pattern, dimensions, device
        )
        rows[name] = {
            **result,
            "checkpoint_rate": rate,
            "checkpoint_epoch": int(state["epoch"]),
        }
        print(
            f"seed={seed} pattern={name} wf1={100*result['weighted_f1']:.3f}",
            flush=True,
        )
    payload = {
        "label": "INTERNAL DIAGNOSTIC ONLY; NOT A FORMAL PAPER RESULT",
        "seed": seed,
        "protocol": "evaluation-only fixed observed-set ablation",
        "checkpoint_rule": "nearest official rate to exact fixed missing fraction",
        "source": str(source),
        "patterns": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def summarize() -> None:
    rows = []
    for seed in SEEDS:
        payload = json.loads((REMOTE / f"seed_{seed}.json").read_text())
        for pattern, metrics in payload["patterns"].items():
            rows.append({
                "seed": seed,
                "pattern": pattern,
                "observed_count": sum(PATTERNS[pattern]),
                "checkpoint_rate": metrics["checkpoint_rate"],
                "checkpoint_epoch": metrics["checkpoint_epoch"],
                "weighted_f1": 100 * metrics["weighted_f1"],
                "accuracy": 100 * metrics["accuracy"],
                "mae": metrics["mae"],
                "correlation": metrics["correlation"],
            })
    with (REMOTE / "per_seed_pattern.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = []
    full = [row["weighted_f1"] for row in rows if row["pattern"] == "ALV"]
    full_mean = float(np.mean(full))
    for pattern in PATTERNS:
        selected = [row for row in rows if row["pattern"] == pattern]
        values = np.asarray([row["weighted_f1"] for row in selected])
        summary.append({
            "pattern": pattern,
            "observed_count": sum(PATTERNS[pattern]),
            "mean_wf1": float(values.mean()),
            "sample_sd": float(values.std(ddof=1)),
            "drop_from_alv": float(values.mean() - full_mean),
            "mean_accuracy": float(np.mean([row["accuracy"] for row in selected])),
            "mean_mae": float(np.mean([row["mae"] for row in selected])),
        })
    with (REMOTE / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    print(json.dumps(summary, indent=2))


def launch() -> None:
    REMOTE.mkdir(parents=True, exist_ok=True)
    pending = list(SEEDS)
    running = []
    while pending or running:
        while pending and len(running) < len(GPUS):
            seed = pending.pop(0); gpu = GPUS[len(running)]
            log = (REMOTE / f"seed_{seed}.log").open("w")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(REPO),
                       OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
            child = subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__)), "--seed", str(seed)],
                cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
            )
            running.append((child, log, seed, gpu))
        child, log, seed, gpu = running.pop(0)
        code = child.wait(); log.close()
        if code:
            raise RuntimeError(f"seed {seed} failed on GPU {gpu}")
    summarize()


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--launch", action="store_true")
    group.add_argument("--seed", type=int, choices=SEEDS)
    group.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.launch: launch()
    elif args.summarize: summarize()
    else: evaluate_seed(args.seed)


if __name__ == "__main__":
    main()
