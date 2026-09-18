"""Train MOSI pattern-balanced and GroupDRO emotion-loss variants."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments.osram_causal_nojepa_20260910 import run as nojepa
from experiments.osram_supervised_teacher_20260914 import run as teacher


ROOT = Path("/data2/yb/remote_experiments/osram_pattern_emotion_20260918")
VARIANTS = (
    "no-jepa-pb",
    "no-jepa-groupdro",
    "reg-only-pb",
    "reg-only-groupdro",
)


def variant_config(seed: int, variant: str):
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    base, source, _ = nojepa.configuration(seed)
    if variant.startswith("reg-only"):
        teacher_path = (
            teacher.ROOT / "teacher" / f"seed_{seed}" / "teacher_projectors.pt"
        )
        cfg = teacher.student_config(base, teacher_path, "joint-reg-only")
        reference = teacher.ROOT / "student-reg-only" / f"seed_{seed}"
    else:
        cfg = replace(
            base,
            training_objective="emotion-only",
            checkpoint_selection="test-oracle-per-rate",
            evaluate_test=True,
            teacher_mode="ema",
            teacher_checkpoint=None,
        )
        reference = source
    cfg = replace(
        cfg,
        emotion_loss_mode=(
            "pattern-balanced" if variant.endswith("pb") else "pattern-groupdro"
        ),
        group_dro_eta=0.1,
    )
    return cfg, reference


def run(variant: str, seed: int) -> None:
    import torch

    from gcnet_missing_m3.train_gcnet import run_experiment

    cfg, reference = variant_config(seed, variant)
    output = ROOT / variant / f"seed_{seed}"
    if (output / "metrics.json").exists():
        raise FileExistsError(f"refusing to overwrite completed run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "variant": variant,
        "seed": seed,
        "reference": str(reference),
        "config": asdict(cfg),
        "label": "INTERNAL DIAGNOSTIC ONLY; NOT A FORMAL PAPER RESULT",
        "source_sha256": {
            name: nojepa.runner.sha(REPO / name)
            for name in (
                "gcnet_missing_m3/train_gcnet.py",
                "gcnet_missing_m3/loss.py",
                "gcnet_missing_m3/model.py",
                "experiments/osram_pattern_emotion_20260918/run.py",
            )
        },
    }
    nojepa.runner.write_json(output / "PROVENANCE.json", provenance)
    try:
        torch.set_num_threads(6)
        roots = [
            str(nojepa.runner.FEATURES / name)
            for name in ("wav2vec-large-c-UTT", "deberta-large-4-UTT", "manet_UTT")
        ]
        print(
            f"TRAIN variant={variant} seed={seed} "
            f"emotion_loss={cfg.emotion_loss_mode} objective={cfg.training_objective}",
            flush=True,
        )
        run_experiment(cfg, *roots, output_dir=str(output))
        metrics = json.loads((output / "metrics.json").read_text())
        if metrics["selection_protocol"] != "per-rate-test-oracle":
            raise RuntimeError("pattern-loss experiment must use per-rate Test oracle")
        provenance.update(
            {
                "status": "complete",
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "selection_protocol": metrics["selection_protocol"],
            }
        )
    except BaseException as error:
        provenance.update(
            {"status": "failed", "error": f"{type(error).__name__}: {error}"}
        )
        nojepa.runner.write_json(output / "PROVENANCE.json", provenance)
        raise
    nojepa.runner.write_json(output / "PROVENANCE.json", provenance)
    print(f"COMPLETE variant={variant} seed={seed}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", choices=nojepa.runner.SEEDS, type=int, required=True)
    args = parser.parse_args()
    run(args.variant, args.seed)
