from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "gcnet"))

import config
from model import GraphModel as BaselineGraphModel
from gcnet_modality_jepa.dataloader_iemocap import IEMOCAPDataset
from gcnet_modality_jepa.loss import MaskedCELoss, MaskedReconLoss
from gcnet_modality_jepa.model import ModalityJEPAGraphModel
from gcnet_modality_jepa.parity import (
    compare_shared_gradients,
    compare_shared_tensors,
    load_shared_backbone,
    miss0_jepa_loss,
)
from gcnet_modality_jepa.train_gcnet import generate_inputs


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict miss=0 parity test on real IEMOCAP batches."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[66, 67, 68])
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fold", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--hidden", type=int, default=100)
    parser.add_argument("--windowp", type=int, default=6)
    parser.add_argument("--windowf", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audio-feature", default="wav2vec-large-c-UTT")
    parser.add_argument("--text-feature", default="deberta-large-4-UTT")
    parser.add_argument("--video-feature", default="manet_UTT")
    parser.add_argument(
        "--comparison",
        choices=("baseline-jepa", "baseline-baseline"),
        default="baseline-jepa",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_fixed_batches(args: argparse.Namespace) -> tuple[list[list[Any]], tuple[int, int, int]]:
    feature_root = Path(config.PATH_TO_FEATURES["IEMOCAPSix"])
    dataset = IEMOCAPDataset(
        label_path=config.PATH_TO_LABEL["IEMOCAPSix"],
        audio_root=str(feature_root / args.audio_feature),
        text_root=str(feature_root / args.text_feature),
        video_root=str(feature_root / args.video_feature),
    )
    test_session = args.fold - 1
    train_indices = [
        index
        for index, conversation_id in enumerate(dataset.vids)
        if int(conversation_id[4]) - 1 != test_session
    ]
    loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=0,
    )
    batches = []
    for batch_index, batch in enumerate(loader):
        batches.append(batch)
        if batch_index + 1 == args.steps:
            break
    if len(batches) != args.steps:
        raise RuntimeError(f"requested {args.steps} batches, found {len(batches)}")
    return batches, dataset.get_featDim()


def prepare_batch(
    batch: list[Any], device: torch.device
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    tensors = [tensor.to(device) for tensor in batch[:9]]
    audio_host, text_host, visual_host = tensors[:3]
    audio_guest, text_guest, visual_guest = tensors[3:6]
    qmask, umask, labels = tensors[6:9]
    input_features = generate_inputs(
        audio_host,
        text_host,
        visual_host,
        audio_guest,
        text_guest,
        visual_guest,
        qmask,
    )
    sequence_length, batch_size = audio_host.shape[:2]
    ones = torch.ones(sequence_length, batch_size, 1, device=device)
    input_masks = generate_inputs(ones, ones, ones, ones, ones, ones, qmask)
    lengths = [int(row.sum().item()) for row in umask]
    return input_features, input_masks, qmask, umask, labels, lengths


def capture_rng() -> tuple[torch.Tensor, list[torch.Tensor]]:
    return torch.get_rng_state(), torch.cuda.get_rng_state_all()


def restore_rng(state: tuple[torch.Tensor, list[torch.Tensor]]) -> None:
    torch.set_rng_state(state[0])
    torch.cuda.set_rng_state_all(state[1])


def predictor_gradient_norm(model: nn.Module) -> float:
    if not hasattr(model, "modality_predictor"):
        return 0.0
    squared_norm = 0.0
    for parameter in model.modality_predictor.parameters():
        if parameter.grad is not None:
            squared_norm += parameter.grad.detach().pow(2).sum().item()
    return squared_norm**0.5


def paired_step(
    baseline: nn.Module,
    candidate: nn.Module,
    baseline_optimizer: torch.optim.Optimizer,
    candidate_optimizer: torch.optim.Optimizer,
    batch: list[Any],
    dimensions: tuple[int, int, int],
    device: torch.device,
) -> dict[str, float]:
    input_features, input_masks, qmask, umask, labels, lengths = prepare_batch(
        batch, device
    )
    cls_loss = MaskedCELoss().to(device)
    rec_loss = MaskedReconLoss().to(device)
    baseline_optimizer.zero_grad(set_to_none=True)
    candidate_optimizer.zero_grad(set_to_none=True)

    rng = capture_rng()
    baseline_logits, baseline_reconstruction, _ = baseline(
        input_features, qmask, umask, lengths
    )
    restore_rng(rng)
    if isinstance(candidate, ModalityJEPAGraphModel):
        candidate_logits, candidate_reconstruction, _, predictions = candidate(
            input_features, qmask, umask, lengths, predict_modalities=False
        )
        if predictions is not None:
            raise AssertionError("predictor forward was not skipped at miss=0")
    else:
        candidate_logits, candidate_reconstruction, _ = candidate(
            input_features, qmask, umask, lengths
        )

    baseline_flat = baseline_logits.transpose(0, 1).contiguous().view(-1, 6)
    candidate_flat = candidate_logits.transpose(0, 1).contiguous().view(-1, 6)
    labels_flat = labels.view(-1)
    adim, tdim, vdim = dimensions
    baseline_total = cls_loss(baseline_flat, labels_flat, umask) + rec_loss(
        baseline_reconstruction,
        input_features,
        input_masks,
        umask,
        adim,
        tdim,
        vdim,
    )
    candidate_total = cls_loss(candidate_flat, labels_flat, umask) + rec_loss(
        candidate_reconstruction,
        input_features,
        input_masks,
        umask,
        adim,
        tdim,
        vdim,
    )
    jepa_zero, jepa_loss_gradient_norm = miss0_jepa_loss(candidate)

    baseline_total.backward()
    candidate_total.backward()
    logits_max_abs = torch.max(
        torch.abs(baseline_logits - candidate_logits)
    ).item()
    reconstruction_max_abs = torch.max(
        torch.abs(baseline_reconstruction[0] - candidate_reconstruction[0])
    ).item()
    shared_gradient_max_abs = compare_shared_gradients(baseline, candidate)
    predictor_grad_norm = predictor_gradient_norm(candidate)
    loss_abs_diff = abs(baseline_total.item() - candidate_total.item())

    baseline_optimizer.step()
    candidate_optimizer.step()
    shared_weight_max_abs = compare_shared_tensors(baseline, candidate)
    return {
        "logits_max_abs": logits_max_abs,
        "reconstruction_max_abs": reconstruction_max_abs,
        "loss_abs_diff": loss_abs_diff,
        "shared_gradient_max_abs": shared_gradient_max_abs,
        "shared_weight_max_abs": shared_weight_max_abs,
        "jepa_loss": jepa_zero.item(),
        "jepa_loss_gradient_norm": jepa_loss_gradient_norm,
        "predictor_gradient_norm": predictor_grad_norm,
    }


def run_seed(
    seed: int,
    args: argparse.Namespace,
    batches: list[list[Any]],
    dimensions: tuple[int, int, int],
    device: torch.device,
) -> dict[str, Any]:
    seed_everything(seed)
    adim, tdim, vdim = dimensions
    model_arguments = dict(
        base_model="LSTM",
        adim=adim,
        tdim=tdim,
        vdim=vdim,
        D_e=args.hidden,
        graph_hidden_size=args.hidden // 2,
        n_speakers=2,
        window_past=args.windowp,
        window_future=args.windowf,
        n_classes=6,
        dropout=args.dropout,
        time_attn=False,
        no_cuda=False,
    )
    baseline = BaselineGraphModel(**model_arguments).to(device).train()
    if args.comparison == "baseline-jepa":
        candidate = ModalityJEPAGraphModel(
            **model_arguments, predictor_dropout=0.1
        ).to(device).train()
        load_shared_backbone(baseline, candidate)
    else:
        candidate = BaselineGraphModel(**model_arguments).to(device).train()
        candidate.load_state_dict(baseline.state_dict())
    initial_shared_weight_max_abs = compare_shared_tensors(baseline, candidate)
    baseline_optimizer = torch.optim.Adam(
        baseline.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    candidate_optimizer = torch.optim.Adam(
        candidate.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    steps = [
        paired_step(
            baseline,
            candidate,
            baseline_optimizer,
            candidate_optimizer,
            batch,
            dimensions,
            device,
        )
        for batch in batches
    ]
    maxima = {
        name: max(step[name] for step in steps)
        for name in steps[0]
    }
    tolerance = 1e-6
    passed = (
        initial_shared_weight_max_abs == 0.0
        and maxima["logits_max_abs"] < tolerance
        and maxima["shared_gradient_max_abs"] < tolerance
        and maxima["shared_weight_max_abs"] < tolerance
        and maxima["jepa_loss"] == 0.0
        and maxima["jepa_loss_gradient_norm"] == 0.0
        and maxima["predictor_gradient_norm"] == 0.0
    )
    return {
        "seed": seed,
        "initial_shared_weight_max_abs": initial_shared_weight_max_abs,
        "steps": steps,
        "maxima": maxima,
        "passed": passed,
    }


def main() -> None:
    args = parse_arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("this real-batch parity test requires CUDA")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device("cuda:0")
    batches, dimensions = load_fixed_batches(args)
    records = [run_seed(seed, args, batches, dimensions, device) for seed in args.seeds]
    result = {
        "dataset": "IEMOCAPSix",
        "missing_rate": 0.0,
        "fold": args.fold,
        "steps_per_seed": args.steps,
        "batch_size": args.batch_size,
        "feature_dimensions": {
            "audio": dimensions[0],
            "text": dimensions[1],
            "visual": dimensions[2],
        },
        "comparison": args.comparison,
        "seeds": records,
        "all_passed": all(record["passed"] for record in records),
        "criterion": "all paired logits, gradients, and post-step shared weights < 1e-6",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
