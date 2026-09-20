"""Five-session cross-validation runner for the current cfg84 OSRAM model."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from gcnet_missing_m3.train_gcnet import TrainConfig, run_experiment


REMOTE_ROOT = Path("/data2/yb/remote_experiments")
OUTPUT_ROOT = REMOTE_ROOT / "osram_iemocap_cfg84_nojepa_5session_20260919"
FEATURE_ROOT = Path(
    "/data2/yb/paper/GCNet_TPAMI_modality_jepa_20260818/dataset/IEMOCAP/features"
)
DATASETS = {4: "IEMOCAPFour", 6: "IEMOCAPSix"}
FOLDS = (1, 2, 3, 4, 5)
LABEL = "INTERNAL DIAGNOSTIC ONLY; NOT A FORMAL PAPER RESULT"
SELECTION_METRIC = "accuracy"


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _source_config(classes: int, seed: int) -> dict:
    candidates = (
        REMOTE_ROOT / f"osram_iemocap{classes}_20260906/seed_{seed}/config.json",
        REMOTE_ROOT / f"osram_iemocap{classes}_20260906/raw/seed_{seed}/config.json",
    )
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError("IEMOCAP source config not found: " + ", ".join(map(str, candidates)))
    return json.loads(path.read_text())


def configuration(classes: int, seed: int, fold: int) -> TrainConfig:
    settings = _source_config(classes, seed)
    settings.update(
        {
            "seed": seed,
            "fold": fold,
            "dataset": DATASETS[classes],
            "epochs": 100,
            "train_rate_mode": "cyclic",
            "training_objective": "emotion-only",
            "checkpoint_selection": "test-oracle-per-rate",
            "evaluate_test": True,
            "osram_bidirectional": False,
            "osram_write_step": 0.6,
            "osram_readout_fusion": "flat",
            "fusion_type": "mean",
            "osram_output_dim": 1600,
            "osram_num_heads": 8,
            "osram_key_dim": 64,
            "osram_value_dim": 64,
            "disable_unused_aux_modules": True,
            "initial_backbone_checkpoint": None,
            "classification_completion": False,
            "completion_path": "none",
            "target_space": "all-modalities",
            "teacher_mode": "ema",
            "teacher_checkpoint": None,
        }
    )
    return TrainConfig(**settings)


def train_fold(classes: int, seed: int, fold: int) -> None:
    output = OUTPUT_ROOT / f"iemocap{classes}" / f"seed_{seed}" / f"fold_{fold}"
    provenance = output / "PROVENANCE.json"
    metrics_path = output / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        if (
            metrics.get("selection_protocol") == "per-rate-test-oracle"
            and metrics.get("selection_metric") == SELECTION_METRIC
        ):
            print(f"SKIP complete IEMOCAP-{classes} seed={seed} fold={fold}", flush=True)
            return
        if metrics.get("selection_protocol") == "per-rate-test-oracle":
            raise RuntimeError(
                f"existing IEMOCAP output uses selection_metric="
                f"{metrics.get('selection_metric')!r}; expected {SELECTION_METRIC!r}. "
                "Use a fresh output root instead of relabeling old checkpoints."
            )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    cfg = configuration(classes, seed, fold)
    _write(output / "config.json", asdict(cfg))
    _write(
        provenance,
        {
            "status": "training",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "label": LABEL,
            "dataset": DATASETS[classes],
            "classes": classes,
            "seed": seed,
            "fold": fold,
            "session_test": fold,
            "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "selection_protocol": "per-rate-test-oracle",
            "selection_metric": SELECTION_METRIC,
            "training_objective": "emotion-only",
            "configuration_delta": {
                "osram_output_dim": 1600,
                "osram_num_heads": 8,
                "osram_key_dim": 64,
                "osram_value_dim": 64,
                "osram_bidirectional": False,
                "osram_write_step": 0.6,
                "fold_protocol": "S1-S5 leave-one-session-out",
            },
        },
    )
    roots = [str(FEATURE_ROOT / name) for name in (
        "wav2vec-large-c-UTT",
        "deberta-large-4-UTT",
        "manet_UTT",
    )]
    print(
        f"START IEMOCAP-{classes} seed={seed} fold={fold} "
        f"(test=S{fold}) GPU={os.environ.get('CUDA_VISIBLE_DEVICES')}",
        flush=True,
    )
    try:
        run_experiment(cfg, *roots, output_dir=str(output))
        metrics = json.loads(metrics_path.read_text())
        if metrics.get("selection_protocol") != "per-rate-test-oracle":
            raise RuntimeError("unexpected checkpoint selection protocol")
        if metrics.get("selection_metric") != SELECTION_METRIC:
            raise RuntimeError("IEMOCAP must select epochs by accuracy")
        payload = json.loads(provenance.read_text())
        payload.update(
            status="complete",
            completed_utc=datetime.now(timezone.utc).isoformat(),
            selected_epoch_by_rate=metrics.get("selected_epoch_by_rate"),
            selected_accuracy_by_rate=metrics.get("selected_accuracy_by_rate"),
            selected_weighted_f1_by_rate=metrics.get("selected_weighted_f1_by_rate"),
        )
        _write(provenance, payload)
        print(f"COMPLETE IEMOCAP-{classes} seed={seed} fold={fold}", flush=True)
    except BaseException as error:
        payload = json.loads(provenance.read_text())
        payload.update(status="failed", error=f"{type(error).__name__}: {error}")
        _write(provenance, payload)
        raise


def launch(classes: int, seed: int) -> None:
    started = time.time()
    for fold in FOLDS:
        train_fold(classes, seed, fold)
    summary = {
        "status": "complete",
        "dataset": DATASETS[classes],
        "classes": classes,
        "seed": seed,
        "folds": list(FOLDS),
        "fold_protocol": "S1-S5 leave-one-session-out",
        "selection_protocol": "per-rate-test-oracle",
        "selection_metric": SELECTION_METRIC,
        "elapsed_seconds": time.time() - started,
    }
    _write(OUTPUT_ROOT / f"iemocap{classes}" / f"seed_{seed}" / "SUMMARY.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", type=int, choices=(4, 6), required=True)
    parser.add_argument("--seed", type=int, default=66)
    args = parser.parse_args()
    launch(args.classes, args.seed)


if __name__ == "__main__":
    main()
