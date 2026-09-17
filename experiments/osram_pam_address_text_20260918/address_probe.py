"""Probe whether current visible A/V + OSRAM Base/Gap contains Text-target information.

This is a diagnostic only.  It does not train PAM, alter checkpoints, or make a
new experiment claim.  It compares direct regressors from the same input
features used by PAM-E's query to:

1. full 256d Teacher Text targets; and
2. a 32d teacher-text subspace trained with the existing
   ``TextSubspacePretrainer`` Stage-1 objective.

Held-out test centered cosine and SmoothL1 are compared with trivial
pattern-conditioned mean predictors.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from gcnet_missing_m3 import train_gcnet as tr
from gcnet_missing_m3.model import MissingM3GraphModel, MODALITIES
from gcnet_missing_m3.text_subspace import TextSubspacePretrainer
from experiments.osram_causal_readout_20260910.run import FEATURES
from experiments.osram_pam_episodic_text_20260917.run import configuration

RATES = (0.1, 0.3, 0.5, 0.7)
PATTERNS = {1: "V", 4: "A", 5: "AV"}


def build_model(cfg, dims, shape, device):
    adim, tdim, vdim = dims
    model = MissingM3GraphModel(
        cfg.base_model,
        adim,
        tdim,
        vdim,
        cfg.hidden,
        cfg.hidden // 2,
        n_speakers=int(shape["num_speakers"]),
        window_past=cfg.window_past,
        window_future=cfg.window_future,
        n_classes=int(shape["num_classes"]),
        dropout=cfg.dropout,
        time_attn=cfg.time_attention,
        no_cuda=device.type != "cuda",
        latent_dim=cfg.latent_dim,
        num_experts=cfg.num_experts,
        top_k=cfg.top_k,
        projector_dropout=cfg.projector_dropout,
        predictor_dropout=cfg.predictor_dropout,
        fusion_type=cfg.fusion_type,
        local_context_residual=cfg.local_context_residual,
        local_fusion_hidden_dim=cfg.local_fusion_hidden_dim,
        local_fusion_dropout=cfg.local_fusion_dropout,
        graph_branch_mode=cfg.graph_branch_mode,
        mmoe_variant=cfg.mmoe_variant,
        target_private_rank=cfg.target_private_rank,
        classification_completion=cfg.classification_completion,
        representation_type=cfg.representation_type,
        node_interaction_residual=cfg.node_interaction_residual,
        readout_type=cfg.readout_type,
        readout_rank=cfg.readout_rank,
        recurrent_padding_mode=cfg.recurrent_padding_mode,
        postgraph_sequence_mode=cfg.postgraph_sequence_mode,
        graph_message_calibration=cfg.graph_message_calibration,
        graph_second_layer=cfg.graph_second_layer,
        postgraph_bilstm_ablation=cfg.postgraph_bilstm_ablation,
        backbone_type=cfg.backbone_type,
        osram_output_dim=cfg.osram_output_dim,
        osram_num_heads=cfg.osram_num_heads,
        osram_key_dim=cfg.osram_key_dim,
        osram_value_dim=cfg.osram_value_dim,
        osram_read_ridge=cfg.osram_read_ridge,
        osram_write_ridge=cfg.osram_write_ridge,
        osram_predictor_mode=cfg.osram_predictor_mode,
        osram_ablation=cfg.osram_ablation,
        osram_emotion_ablation=cfg.osram_emotion_ablation,
        osram_query_availability=cfg.osram_query_availability,
        osram_bidirectional=cfg.osram_bidirectional,
        osram_forward_slot_reuse=cfg.osram_forward_slot_reuse,
        osram_readout_fusion=cfg.osram_readout_fusion,
        osram_write_step=cfg.osram_write_step,
        completion_path=cfg.completion_path,
        pam_key_dim=cfg.pam_key_dim,
        pam_loss_weight=cfg.pam_loss_weight,
        complete_state_jepa=cfg.training_objective == "complete-state",
        write_state_completion=cfg.training_objective == "write-state",
        future_state_jepa=cfg.training_objective == "future-state",
        teacher_mode=cfg.teacher_mode,
        teacher_checkpoint=cfg.teacher_checkpoint,
        target_space=cfg.target_space,
        text_subspace_checkpoint=cfg.text_subspace_checkpoint,
        training_objective=cfg.training_objective,
    ).to(device)
    return model


def load_checkpoint_model(cfg, dims, shape, checkpoint, device):
    model = build_model(cfg, dims, shape, device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model


@torch.no_grad()
def collect_pam_queries(model, loader, cfg, dims, schedule, device):
    """Collect PAM-E's current-observation address features and Teacher targets."""
    xs, ys, pats = [], [], []
    for raw in loader:
        data = tr._move_batch(raw, device)
        view = tr._prepare_view(data, schedule, 0, dims)
        availability, umask = view["availability"], view["umask"]
        valid = umask.T.bool()
        encoded, latents = model.observed_set(
            view["incomplete"], availability, umask
        )
        base, gap = model.osram.causal_read_contexts(
            encoded, latents, availability, view["qmask"], umask, view["lengths"]
        )
        teacher = model.encode_teacher_targets([view["complete"]])
        a_mask = availability[..., 0]
        v_mask = availability[..., 2]
        t_mask = availability[..., 1]
        keep = valid & (t_mask <= 0) & ((a_mask > 0) | (v_mask > 0))
        if not bool(keep.any()):
            continue
        x = torch.cat(
            (
                latents["audio"][keep] * a_mask[keep].unsqueeze(-1),
                latents["visual"][keep] * v_mask[keep].unsqueeze(-1),
                torch.stack((a_mask[keep], v_mask[keep]), dim=-1),
                base[keep],
                gap[keep][:, 1, :],
            ),
            dim=-1,
        )
        pattern_id = (a_mask[keep].long() * 4 + v_mask[keep].long())
        xs.append(x.cpu())
        ys.append(teacher["text"][keep].cpu())
        pats.append(pattern_id.cpu())
    if not xs:
        raise RuntimeError("No T-missing A/V-source rows collected")
    return torch.cat(xs), torch.cat(ys), torch.cat(pats)


@torch.no_grad()
def collect_teacher_all(model, loader, cfg, dims, device):
    """Collect complete Teacher latents/labels for Stage-1 subspace training."""
    data = {m: [] for m in MODALITIES}
    labels = []
    schedule = tr._build_schedule(cfg, "train", 0.0)
    for raw in loader:
        batch = tr._move_batch(raw, device)
        view = tr._prepare_view(batch, schedule, 0, dims)
        valid = view["umask"].T.bool()
        teacher = model.encode_teacher_targets([view["complete"]])
        for name in MODALITIES:
            data[name].append(teacher[name][valid].cpu())
        labels.append(view["labels"].T[valid].cpu())
    return {name: torch.cat(value) for name, value in data.items()}, torch.cat(labels)


def train_subspace(train_data, train_labels, val_data, val_labels, device, epochs=100):
    model = TextSubspacePretrainer(latent_dim=256, subspace_dim=32, hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best = float("inf")
    best_state = None
    n = train_labels.shape[0]
    generator = torch.Generator().manual_seed(7723)
    for _ in range(epochs):
        model.train()
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, 256):
            index = order[start:start + 256]
            batch = {name: train_data[name][index].to(device) for name in MODALITIES}
            labels = train_labels[index].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(batch, labels, beta=1.0)["total"]
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_batch = {name: val_data[name].to(device) for name in MODALITIES}
            val_loss = float(model.loss(val_batch, val_labels.to(device), beta=1.0)["total"])
        if val_loss < best:
            best = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    model.load_state_dict(best_state)
    projector = model.projector.eval().requires_grad_(False)
    return projector, best


class MLPProbe(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_probe(x_train, y_train, x_val, y_val, device, seed=0):
    torch.manual_seed(seed)
    model = MLPProbe(x_train.shape[1], y_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best = float("inf")
    best_state = None
    patience = 0
    batch_size = 128
    n = x_train.shape[0]
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(250):
        model.train()
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, batch_size):
            index = order[start:start + batch_size]
            xb = x_train[index].to(device)
            yb = y_train[index].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.smooth_l1_loss(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(F.smooth_l1_loss(model(x_val.to(device)), y_val.to(device)))
        if val_loss < best:
            best = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 30:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best


def centered_cosine(prediction, target):
    prediction = prediction - prediction.mean(0, keepdims=True)
    target = target - target.mean(0, keepdims=True)
    return (
        (prediction * target).sum(-1)
        / (np.linalg.norm(prediction, axis=-1) * np.linalg.norm(target, axis=-1) + 1e-8)
    )


def evaluate(prediction, target, pattern, train_target, train_pattern):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    pattern = np.asarray(pattern)
    metrics = {}
    for name, value in (("probe", prediction),):
        metrics[name] = {
            "centered_cosine": float(centered_cosine(value, target).mean()),
            "raw_cosine": float(
                ((value * target).sum(-1) / (np.linalg.norm(value, axis=-1) * np.linalg.norm(target, axis=-1) + 1e-8)).mean()
            ),
            "smooth_l1": float(F.smooth_l1_loss(torch.from_numpy(value), torch.from_numpy(target)).item()),
            "pred_norm": float(np.linalg.norm(value, axis=-1).mean()),
            "target_norm": float(np.linalg.norm(target, axis=-1).mean()),
        }
    # pattern-conditioned train mean baseline
    train_target = np.asarray(train_target, dtype=np.float64)
    train_pattern = np.asarray(train_pattern)
    baseline = np.stack([
        train_target[train_pattern == pattern_id].mean(0)
        if np.any(train_pattern == pattern_id)
        else train_target.mean(0)
        for pattern_id in pattern
    ])
    metrics["pattern_mean"] = {
        "centered_cosine": float(centered_cosine(baseline, target).mean()),
        "smooth_l1": float(F.smooth_l1_loss(torch.from_numpy(baseline), torch.from_numpy(target)).item()),
    }
    metrics["zero"] = {
        "centered_cosine": float(centered_cosine(np.zeros_like(target), target).mean()),
        "smooth_l1": float(F.smooth_l1_loss(torch.zeros_like(torch.from_numpy(target)), torch.from_numpy(target)).item()),
    }
    for pattern_id, name in PATTERNS.items():
        selected = pattern == pattern_id
        if bool(selected.any()):
            metrics.setdefault("per_pattern", {})[name] = {
                "n": int(selected.sum()),
                "centered_cosine": float(centered_cosine(prediction[selected], target[selected]).mean()),
                "smooth_l1": float(F.smooth_l1_loss(torch.from_numpy(prediction[selected]), torch.from_numpy(target[selected])).item()),
            }
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("/data2/yb/remote_experiments/osram_pam_episodic_text_20260917/mosi/seed_66/best_miss_0p5.pt"))
    parser.add_argument("--output", type=Path,
                        default=Path("/data2/yb/remote_experiments/osram_pam_episodic_text_20260917/address_probe_seed66.json"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg, _, _ = configuration(args.seed)
    cfg = replace(cfg, device=args.device)
    shape = tr._resolve_task_contract(cfg.dataset, cfg.mosi_task_mode)
    device = torch.device(args.device)
    loaders = tr.get_loaders(
        audio_root=str(FEATURES / "wav2vec-large-c-UTT"),
        text_root=str(FEATURES / "deberta-large-4-UTT"),
        video_root=str(FEATURES / "manet_UTT"),
        num_folder=int(shape["num_folds"]),
        dataset=cfg.dataset,
        batch_size=cfg.batch_size,
        num_workers=0,
        seed=cfg.seed,
        validation_fraction=cfg.validation_fraction,
        evaluation_protocol=cfg.evaluation_protocol,
    )
    train_loaders, val_loaders, test_loaders, adim, tdim, vdim = loaders
    dims = (adim, tdim, vdim)
    train_loader = train_loaders[cfg.fold - 1]
    val_loader = val_loaders[cfg.fold - 1]
    test_loader = test_loaders[cfg.fold - 1]

    model = load_checkpoint_model(cfg, dims, shape, args.checkpoint, device)

    # Teacher latents for Stage-1 subspace.
    train_all, train_labels = collect_teacher_all(model, train_loader, cfg, dims, device)
    val_all, val_labels = collect_teacher_all(model, val_loader, cfg, dims, device)
    projector, subspace_val_loss = train_subspace(
        train_all, train_labels, val_all, val_labels, device
    )

    collections = {"train": {}, "validation": {}, "test": {}}
    for split, loader in (("train", train_loader), ("validation", val_loader), ("test", test_loader)):
        for rate in RATES:
            schedule = tr._build_schedule(cfg, split, rate)
            x, y, pattern = collect_pam_queries(model, loader, cfg, dims, schedule, device)
            collections[split][rate] = (x, y, pattern)
    # Concatenate rates within each split.
    data = {}
    for split, rate_items in collections.items():
        xs, ys, pats = [], [], []
        for rate, (x, y, pattern) in rate_items.items():
            xs.append(x)
            ys.append(y)
            pats.append(pattern)
        data[split] = (torch.cat(xs), torch.cat(ys), torch.cat(pats))

    x_train, y_train, pat_train = data["train"]
    x_val, y_val, _ = data["validation"]
    x_test, y_test, pat_test = data["test"]
    # Use per-split feature standardization; probe fitting is diagnostic only.
    mean = x_train.mean(0, keepdim=True)
    std = x_train.std(0, keepdim=True).clamp_min(1e-6)
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std

    y_sub_train = projector(y_train.to(device)).cpu()
    y_sub_val = projector(y_val.to(device)).cpu()
    y_sub_test = projector(y_test.to(device)).cpu()

    report = {
        "label": "INTERNAL DIAGNOSTIC ONLY; NOT A FORMAL PAPER RESULT",
        "seed": args.seed,
        "checkpoint": str(args.checkpoint),
        "rates": list(RATES),
        "n_train": int(x_train.shape[0]),
        "n_validation": int(x_val.shape[0]),
        "n_test": int(x_test.shape[0]),
        "subspace_validation_loss": subspace_val_loss,
        "target_spaces": {},
    }
    for name, y_train_target, y_val_target, y_test_target in (
        ("full-text-256", y_train, y_val, y_test),
        ("predictable-subspace-32", y_sub_train, y_sub_val, y_sub_test),
    ):
        probe, val_loss = train_probe(
            x_train, y_train_target, x_val, y_val_target, device, seed=args.seed
        )
        with torch.no_grad():
            prediction = probe(x_test.to(device)).cpu().numpy()
        report["target_spaces"][name] = {
            "probe_validation_loss": val_loss,
            "test": evaluate(
                prediction,
                y_test_target.numpy(),
                pat_test.numpy(),
                y_train_target.numpy(),
                pat_train.numpy(),
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
