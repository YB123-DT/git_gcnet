"""Run OSRAM with the original batch-level GCNet random-mask sampler."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess


REMOTE = Path(os.environ.get("OSRAM_REMOTE_ROOT", "/data2/yb/remote_experiments"))
REFERENCE = REMOTE / "osram_iemocap_cfg84_nojepa_5session_20260919"
IEMOCAP_DATASET = Path(os.environ.get(
    "IEMOCAP_DATASET_ROOT",
    "/data2/yb/paper/GCNet_TPAMI_modality_jepa_20260818/dataset",
))
OUTPUT_ROOT = Path(os.environ["OSRAM_GCNET_MASK_OUTPUT"])


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _discard_checkpoints(output: Path) -> dict[str, int]:
    checkpoints = sorted(output.glob("best_miss_*.pt"))
    size = sum(path.stat().st_size for path in checkpoints)
    for path in checkpoints:
        path.unlink()
    return {"count": len(checkpoints), "bytes": size}


def _iemocap_config(classes: int, seed: int, fold: int):
    from gcnet_missing_m3.train_gcnet import TrainConfig

    path = REFERENCE / f"iemocap{classes}" / f"seed_{seed}" / f"fold_{fold}" / "config.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    settings = json.loads(path.read_text())
    settings.update(
        epochs=100,
        seed=seed,
        fold=fold,
        dataset="IEMOCAPFour" if classes == 4 else "IEMOCAPSix",
        missing_mask_protocol="gcnet-random-mask-v1",
        osram_meaningful_block="none",
    )
    return TrainConfig(**settings)


def run_iemocap(classes: int, seed: int, fold: int) -> None:
    output = OUTPUT_ROOT / f"iemocap{classes}" / f"seed_{seed}" / f"fold_{fold}"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    os.environ["GCNET_DATASET_ROOT"] = str(IEMOCAP_DATASET)
    os.environ["GCNET_CACHE_ROOT"] = str(OUTPUT_ROOT / f"cache_iemocap{classes}")
    config = _iemocap_config(classes, seed, fold)
    feature_root = IEMOCAP_DATASET / "IEMOCAP" / "features"
    roots = [feature_root / name for name in (
        "wav2vec-large-c-UTT", "deberta-large-4-UTT", "manet_UTT"
    )]
    provenance = {
        "status": "running", "started_utc": _now(), "pid": os.getpid(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "protocol": "original GCNet random_mask; per-rate Test-oracle; internal diagnostic",
        "config": asdict(config),
    }
    _write(output / "PROVENANCE.json", provenance)
    try:
        from gcnet_missing_m3.train_gcnet import run_experiment
        run_experiment(config, *(str(path) for path in roots), output_dir=str(output))
        provenance.update(
            status="complete", finished_utc=_now(),
            discarded_checkpoints=_discard_checkpoints(output),
        )
    except BaseException as error:
        provenance.update(status="failed", finished_utc=_now(), error=repr(error))
        raise
    finally:
        _write(output / "PROVENANCE.json", provenance)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", type=int, choices=(4, 6), required=True)
    parser.add_argument("--seed", type=int, choices=(66, 67, 68), required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    args = parser.parse_args()
    run_iemocap(args.classes, args.seed, args.fold)


if __name__ == "__main__":
    main()
