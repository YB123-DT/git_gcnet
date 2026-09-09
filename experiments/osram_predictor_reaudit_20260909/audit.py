"""Frozen-checkpoint, natural miss=.5 predictor audit; never trains a model."""
import argparse
import csv
import hashlib
import importlib.util
import inspect
import json
from dataclasses import asdict
from pathlib import Path

import torch
from gcnet_missing_m3 import train_gcnet as tr
from gcnet_missing_m3.model import MissingM3GraphModel, MODALITIES

OLD = Path(__file__).resolve().parents[1] / "missing_m3_mosi_latent_diagnostic_20260831/analyze_checkpoint.py"
spec = importlib.util.spec_from_file_location("old_predictor_audit", OLD)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
PATTERNS = {"A": (1, 0, 0), "V": (0, 0, 1), "AV": (1, 0, 1)}


@torch.no_grad()
def audit(path, feature_root):
    cp = torch.load(path, map_location="cpu", weights_only=False)
    cfg = tr.TrainConfig(**cp["config"])
    assert cfg.backbone_type == "osram" and not cfg.osram_bidirectional
    assert not cfg.classification_completion
    tr.set_random_seed(cfg.seed)
    contract = tr._resolve_task_contract(cfg.dataset, cfg.mosi_task_mode)
    paths = [str(feature_root / f) for f in
             ("wav2vec-large-c-UTT", "deberta-large-4-UTT", "manet_UTT")]
    _, _, loaders, adim, tdim, vdim = tr.get_loaders(
        audio_root=paths[0], text_root=paths[1], video_root=paths[2],
        num_folder=int(contract["num_folds"]), dataset=cfg.dataset,
        batch_size=cfg.batch_size, num_workers=0, seed=cfg.seed,
        validation_fraction=cfg.validation_fraction,
        evaluation_protocol=cfg.evaluation_protocol)
    names = inspect.signature(MissingM3GraphModel.__init__).parameters
    kwargs = {k: v for k, v in asdict(cfg).items() if k in names}
    kwargs.update(adim=adim, tdim=tdim, vdim=vdim, D_e=cfg.hidden,
                  graph_hidden_size=cfg.hidden // 2,
                  n_speakers=int(contract["num_speakers"]),
                  n_classes=int(contract["num_classes"]),
                  time_attn=cfg.time_attention, no_cuda=True)
    model = MissingM3GraphModel(**kwargs).eval()
    model.load_state_dict(cp["model"], strict=True)
    collected = {}
    digest = hashlib.sha256()
    ids = []
    schedule = tr._build_schedule(cfg, "test", .5)
    for data in loaders[cfg.fold - 1]:
        view = tr._prepare_view(data, schedule, epoch=0, dimensions=(adim, tdim, vdim))
        availability = view["availability"]
        valid = view["umask"].T.bool()
        digest.update(availability[valid].numpy().tobytes())
        ids.extend(str(x) for x in view["conversation_ids"])
        logits, _, _, pred = model([view["incomplete"]], availability,
            view["qmask"], view["umask"], view["lengths"], predict_missing=True)
        # A predictor audit must not change the deployed classifier output.
        without = model([view["incomplete"]], availability, view["qmask"],
                        view["umask"], view["lengths"], predict_missing=False)[0]
        assert torch.equal(logits, without)
        teacher = model.encode_teacher_targets([view["complete"]])
        for pattern, bits in PATTERNS.items():
            cohort = (availability == availability.new_tensor(bits)).all(-1) & valid
            for q, name in enumerate(MODALITIES):
                if bits[q]:
                    continue
                selected = cohort & pred.target_mask[..., q]
                for branch, values in (("regression", pred.reg_predictions),
                                       ("contrastive", pred.cl_predictions)):
                    key = (pattern, name, branch)
                    pairs = collected.setdefault(key, ([], []))
                    pairs[0].append(values[..., q, :][selected].cpu())
                    pairs[1].append(teacher[name][selected].cpu())
    assert all(torch.equal(value.cpu(), cp["model"][key])
               for key, value in model.state_dict().items())
    rows = []
    for (pattern, target, branch), (ps, ts) in collected.items():
        p, t = torch.cat(ps), torch.cat(ts)
        assert len(p) >= 2 and torch.isfinite(p).all() and torch.isfinite(t).all()
        metrics = old._metrics(p, t, cfg.temperature,
                               cfg.seed + 100 * MODALITIES.index(target))
        metrics["std_ratio"] = metrics["channel_std"] / max(metrics["target_channel_std"], 1e-12)
        metrics["retrieval_over_chance"] = metrics["retrieval_top1"] / metrics["chance"]
        rows.append(dict(seed=cfg.seed, write_step=cfg.osram_write_step, rate=.5,
                         epoch=cp["epoch"], pattern=pattern, target=target,
                         branch=branch, **metrics))
    return {"checkpoint": str(path), "mask_sha256": digest.hexdigest(),
            "conversation_ids": ids, "config": asdict(cfg),
            "state_unchanged": True, "logits_unchanged": True, "rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in range(66, 71):
        pair = []
        for variant, path in (
            ("eta06", args.root / f"osram_write_step_train_20260909/mosi/seed_{seed}/best.pt"),
            ("eta1", args.root / f"osram_forward_only_mosi_20260908/seed_{seed}/best.pt")):
            result = audit(path, args.feature_root)
            (args.output / f"{variant}_seed_{seed}.json").write_text(json.dumps(result, indent=2))
            pair.append(result)
            results.extend(result["rows"])
            print(f"{variant} seed {seed}: {len(result['rows'])} groups; state/logits unchanged", flush=True)
        assert pair[0]["mask_sha256"] == pair[1]["mask_sha256"]
        assert pair[0]["conversation_ids"] == pair[1]["conversation_ids"]
    with (args.output / "per_seed.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
