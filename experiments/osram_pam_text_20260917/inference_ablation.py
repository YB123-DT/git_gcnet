"""Four inference modes for a fixed PAM-T checkpoint and fixed test masks.

Normal / Zero / Shuffle / Oracle-Fusion, plus prior-write coverage and
|s_normal - s_zero|.  Evaluation only; no training and no checkpoint selection.
"""
import argparse
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from gcnet_missing_m3 import train_gcnet as tr
from gcnet_missing_m3.model import MissingM3GraphModel

FEATURE_NAMES = ("wav2vec-large-c-UTT", "deberta-large-4-UTT", "manet_UTT")
RATES = tuple(f"{i/10:.1f}" for i in range(8))
SEEDS = (66, 67, 68, 69, 70)
MODES = ("Normal", "Zero", "Shuffle", "Oracle-Fusion")


def weighted_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    mask = labels != 0
    if not bool(mask.any()):
        return float("nan")
    return float(
        f1_score(
            (labels[mask] > 0).astype(int),
            (predictions[mask] > 0).astype(int),
            average="weighted",
        )
    )


def build_model(cfg: tr.TrainConfig, dims, device: torch.device):
    names = inspect.signature(MissingM3GraphModel.__init__).parameters
    kwargs = {key: value for key, value in cfg.__dict__.items() if key in names}
    kwargs.update(
        adim=dims[0],
        tdim=dims[1],
        vdim=dims[2],
        D_e=cfg.hidden,
        graph_hidden_size=cfg.hidden // 2,
        n_speakers=1,
        n_classes=1,
        time_attn=cfg.time_attention,
    )
    return MissingM3GraphModel(**kwargs).to(device).eval()


def collect_pass(model, views, dims, device):
    """Run one normal forward pass and collect PAM state and text latents."""
    normal_overrides, zero_overrides, oracle_overrides = [], [], []
    original_zhats, masks, labels, normal_preds = [], [], [], []
    normal_labels = []
    missing_vectors, oracle_vectors = [], []
    coverage_num = 0
    coverage_den = 0
    for view in views:
        with torch.no_grad():
            logits, _, _, _ = model(
                [view["incomplete"]],
                view["availability"],
                view["qmask"],
                view["umask"],
                view["lengths"],
                predict_missing=False,
            )
        pam = model.last_pam_outputs
        z_hat = pam["z_hat_text"].detach()
        tm = pam["text_missing_mask"]
        prior = torch.cumsum(pam["write_mask"].int(), dim=0) > 0
        prior_before = torch.zeros_like(prior)
        prior_before[1:] = prior[:-1]
        coverage_num += int((prior_before & tm).sum().item())
        coverage_den += int(tm.sum().item())
        real_text = model.observed_set.projectors["text"](
            view["complete"][..., dims[0] : dims[0] + dims[1]]
        ).detach()
        zero = torch.zeros_like(z_hat)
        oracle = torch.where(tm.unsqueeze(-1), real_text, z_hat)
        normal_overrides.append(z_hat)
        zero_overrides.append(zero)
        oracle_overrides.append(oracle)
        original_zhats.append(z_hat)
        masks.append(tm)
        # Collect all T-missing vectors in global order for shuffle.
        for t, b in tm.nonzero(as_tuple=False).tolist():
            missing_vectors.append(z_hat[t, b].cpu())
            oracle_vectors.append(real_text[t, b].cpu())
        # predictions/labels for this batch, only valid rows
        valid_t = view["umask"].transpose(0, 1).bool()
        labels_t = view["labels"].transpose(0, 1)[valid_t]
        labels.append(labels_t.detach().cpu().numpy())
        normal_preds.append(logits[valid_t].detach().cpu().numpy())
    return dict(
        normal_overrides=normal_overrides,
        zero_overrides=zero_overrides,
        oracle_overrides=oracle_overrides,
        original_zhats=original_zhats,
        masks=masks,
        labels=np.concatenate(labels) if labels else np.zeros(0),
        normal_preds=np.concatenate(normal_preds) if normal_preds else np.zeros(0),
        missing_vectors=missing_vectors,
        oracle_vectors=oracle_vectors,
        coverage_num=coverage_num,
        coverage_den=coverage_den,
    )


def shuffled_overrides(original_zhats, masks, missing_vectors, seed=1234):
    if not missing_vectors:
        return [z.clone() for z in original_zhats]
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(missing_vectors), generator=generator).tolist()
    shuffled_vectors = [missing_vectors[i] for i in order]
    result = []
    cursor = 0
    for z_hat, tm in zip(original_zhats, masks):
        clone = z_hat.clone()
        for t, b in tm.nonzero(as_tuple=False).tolist():
            clone[t, b] = shuffled_vectors[cursor].to(clone.device)
            cursor += 1
        result.append(clone)
    return result


def run_mode(model, views, overrides, device):
    predictions = []
    labels = []
    original_forward = model.pam_text_memory.forward
    try:
        for view, override in zip(views, overrides):
            override = override.detach().to(device)

            def patched(latents, availability, umask, override=override):
                out = dict(original_forward(latents, availability, umask))
                out["z_hat_text"] = override
                return out

            model.pam_text_memory.forward = patched
            with torch.no_grad():
                logits, _, _, _ = model(
                    [view["incomplete"]],
                    view["availability"],
                    view["qmask"],
                    view["umask"],
                    view["lengths"],
                    predict_missing=False,
                )
            valid_t = view["umask"].transpose(0, 1).bool()
            labels_t = view["labels"].transpose(0, 1)[valid_t]
            predictions.append(logits[valid_t].detach().cpu().numpy())
            labels.append(labels_t.detach().cpu().numpy())
    finally:
        model.pam_text_memory.forward = original_forward
    return np.concatenate(predictions), np.concatenate(labels)


def evaluate_seed(root, seed, device):
    seed_root = Path(root) / "mosi" / f"seed_{seed}"
    checkpoint = torch.load(seed_root / "best_miss_0p0.pt", map_location="cpu", weights_only=False)
    cfg = tr.TrainConfig(**checkpoint["config"])
    roots = [str(Path(cfg.data_root) / name) if False else None for name in FEATURE_NAMES]
    # Resolve feature roots from the standard reproduction layout.
    feature_root = Path("/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset/CMUMOSI/features")
    roots = [str(feature_root / name) for name in FEATURE_NAMES]
    train, validation, test, adim, tdim, vdim = tr.get_loaders(
        audio_root=roots[0],
        text_root=roots[1],
        video_root=roots[2],
        num_folder=1,
        dataset=cfg.dataset,
        batch_size=cfg.batch_size,
        num_workers=0,
        seed=cfg.seed,
        validation_fraction=cfg.validation_fraction,
        evaluation_protocol=cfg.evaluation_protocol,
    )
    dims = (adim, tdim, vdim)
    test_loader = test[0]
    batches = [tr._move_batch(raw, device) for raw in test_loader]
    rows = []
    for rate in RATES:
        ckpt = torch.load(seed_root / f"best_miss_{rate.replace('.', 'p')}.pt", map_location="cpu", weights_only=False)
        model = build_model(tr.TrainConfig(**ckpt["config"]), dims, device)
        model.load_state_dict(ckpt["model"], strict=True)
        schedule = tr._build_schedule(cfg, "test", float(rate))
        views = [tr._prepare_view(batch, schedule, 0, dims) for batch in batches]
        collected = collect_pass(model, views, dims, device)
        shuffle = shuffled_overrides(
            collected["original_zhats"], collected["masks"], collected["missing_vectors"]
        )
        mode_outputs = {}
        mode_outputs["Normal"] = (collected["normal_preds"], collected["labels"])
        mode_outputs["Zero"] = run_mode(model, views, collected["zero_overrides"], device)
        mode_outputs["Shuffle"] = run_mode(model, views, shuffle, device)
        mode_outputs["Oracle-Fusion"] = run_mode(model, views, collected["oracle_overrides"], device)
        for mode in MODES:
            preds, labels = mode_outputs[mode]
            rows.append(
                dict(
                    group="PAM-T",
                    seed=seed,
                    rate=rate,
                    mode=mode,
                    weighted_f1=weighted_f1(labels, preds),
                    prior_write_coverage=(collected["coverage_num"] / collected["coverage_den"]
                                          if collected["coverage_den"] else None),
                    t_missing_count=collected["coverage_den"],
                )
            )
            if mode == "Zero":
                pass
        # Normal - Zero per seed/rate
        normal = [r for r in rows if r["seed"] == seed and r["rate"] == rate and r["mode"] == "Normal"][0]
        zero = [r for r in rows if r["seed"] == seed and r["rate"] == rate and r["mode"] == "Zero"][0]
        rows[-len(MODES) + 1]["normal_minus_zero_abs"] = abs(normal["weighted_f1"] - zero["weighted_f1"])
    return rows


def summarize(rows):
    groups = {}
    for mode in MODES:
        vals = [r["weighted_f1"] for r in rows if r["mode"] == mode]
        groups[mode] = dict(mean=float(np.nanmean(vals)) * 100.0, sd=float(np.nanstd(vals, ddof=1)) * 100.0)
    cov = [float(r["prior_write_coverage"]) for r in rows if r["mode"] == "Normal" and r.get("prior_write_coverage") is not None]
    summary = {
        "modes": groups,
        "prior_write_coverage_mean": float(np.mean(cov)) if cov else None,
        "normal_minus_zero_abs_mean": float(np.nanmean([abs(r["weighted_f1"] - [x for x in rows if x["seed"] == r["seed"] and x["rate"] == r["rate"] and x["mode"] == "Zero"][0]["weighted_f1"]) for r in rows if r["mode"] == "Normal"])) * 100.0,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data2/yb/remote_experiments/osram_pam_text_20260917"))
    parser.add_argument("--output", type=Path, default=Path("/data2/yb/remote_experiments/osram_pam_text_20260917/inference_ablation"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    all_rows = []
    for seed in SEEDS:
        rows = evaluate_seed(args.root, seed, device)
        all_rows.extend(rows)
        print(f"seed {seed} done", flush=True)
    (args.output / "per_seed_rate.csv").write_text(
        "group,seed,rate,mode,weighted_f1,prior_write_coverage,t_missing_count,normal_minus_zero_abs\n"
        + "\n".join(
            f"{r['group']},{r['seed']},{r['rate']},{r['mode']},{r['weighted_f1']},"
            f"{r.get('prior_write_coverage','')},{r.get('t_missing_count','')},{r.get('normal_minus_zero_abs','')}"
            for r in all_rows
        )
        + "\n"
    )
    summary = summarize(all_rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
