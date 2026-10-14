"""Frozen MELD write/retention audit; never trains or changes a checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import replace
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from experiments.osram_meld_flat_source_combinations_20261013.evaluate import (
    _build_model, _sha256, _write,
)
from experiments.osram_meld_gcnet_random_no_skip_evidence_20261014.evaluate import (
    CANONICAL_MASK_PROTOCOL, replay_view_from_availability, scatter_saved_availability,
)


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class WriteCapture:
    """Wrap the actual joint-write call without altering its arguments/result."""

    def __init__(self, osram, umask):
        self.osram = osram
        self.times = umask.bool().any(dim=0).nonzero().flatten().tolist()
        self.snapshots = []

    def __enter__(self):
        self.original = self.osram.block_write
        self.had_instance_method = "block_write" in self.osram.__dict__
        self.instance_method = self.osram.__dict__.get("block_write")

        def capture(memory, keys, values, availability, beta=None):
            index = len(self.snapshots)
            if index >= len(self.times):
                raise RuntimeError("More than one causal scan encountered")
            before = memory.detach().clone()
            result = self.original(memory, keys, values, availability, beta=beta)
            self.snapshots.append((
                self.times[index], before, keys.detach().clone(),
                values.detach().clone(), availability.detach().clone(),
                result.detach().clone(),
            ))
            return result

        self.osram.block_write = capture
        return self

    def __exit__(self, *_args):
        if self.had_instance_method:
            self.osram.block_write = self.instance_method
        else:
            del self.osram.block_write


def guard_gpu(uuid):
    if not uuid.startswith("GPU-") or os.environ.get("CUDA_VISIBLE_DEVICES") != uuid:
        raise ValueError("CUDA_VISIBLE_DEVICES must equal the explicit --gpu-uuid")
    listing = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits",
    ], text=True)
    matches = [line.split(",") for line in listing.splitlines()
               if line.split(",")[-1].strip() == uuid]
    if len(matches) != 1 or matches[0][0].strip() == "4":
        raise ValueError("GPU UUID not found uniquely or identifies forbidden host GPU 4")
    return {"host_index": int(matches[0][0]), "uuid": uuid, "logical_device": "cuda:0"}


def evaluate_rate(model, loader, config, dimensions, saved, rate, stream, max_batches):
    from gcnet_missing_m3.train_gcnet import _collect_predictions, _move_batch
    from experiments.osram_meld_write_retention_20261014.audit import audit_batch

    cursor = batches = row_count = 0
    ids = []
    saved_availability = np.asarray(saved["availability"])
    saved_predictions = np.asarray(saved["predictions"])
    saved_labels = np.asarray(saved["labels"])
    if len(saved_availability) != len(saved_predictions) or len(saved_labels) != len(saved_predictions):
        raise ValueError("Saved availability/prediction/label counts disagree")
    with torch.no_grad():
        for raw in loader:
            if max_batches is not None and batches >= max_batches:
                break
            data = _move_batch(raw, next(model.parameters()).device)
            start = cursor
            availability, cursor = scatter_saved_availability(saved_availability, cursor, data[7])
            view = replay_view_from_availability(data, availability, dimensions)
            with WriteCapture(model.osram, view["umask"]) as capture:
                logits, _, _, completion = model(
                    [view["incomplete"]], view["availability"], view["qmask"],
                    view["umask"], view["lengths"], predict_missing=False,
                )
            if completion is not None or len(capture.snapshots) != len(capture.times):
                raise RuntimeError("Expected one complete causal scan without completion")
            pred, labels, _ = _collect_predictions(
                config.dataset, logits, view["labels"], view["umask"], config.mosi_task_mode,
            )
            np.testing.assert_array_equal(labels, saved_labels[start:cursor])
            np.testing.assert_array_equal(pred, saved_predictions[start:cursor])
            sample_ids = [str(value) for value in data[-1]]
            if len(sample_ids) != view["umask"].shape[0]:
                raise ValueError("Conversation ID count does not match batch")
            # Capture has been removed before counterfactual block_write replay.
            rows = audit_batch(model.osram, capture.snapshots, sample_ids, rate)
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                row_count += 1
            ids.extend(sample_ids)
            batches += 1
            stream.flush()
            print(f"rate={rate:.1f} batches={batches} utterances={cursor} rows={row_count}", flush=True)
    if max_batches is None and cursor != len(saved_availability):
        raise ValueError("Saved predictions were not fully consumed")
    return {
        "batches": batches, "utterances": cursor, "rows": row_count,
        "conversation_ids": ids, "saved_predictions_exact": True,
        "saved_labels_exact": True, "saved_count": len(saved_labels),
        "full_saved_coverage": cursor == len(saved_labels),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "dataset-root", "meld-pickle", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--rates", nargs="+", type=float, default=[i / 10 for i in range(8)])
    parser.add_argument("--max-batches", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.max_batches is not None and args.max_batches < 1:
        parser.error("--max-batches must be positive")
    if len(set(args.rates)) != len(args.rates) or any(r not in [i / 10 for i in range(8)] for r in args.rates):
        parser.error("--rates must be unique values from 0.0 through 0.7")
    # Do not silently substitute no-skip/legacy checkpoints, or allocate a GPU
    # and build loaders when the exact canonical checkpoint has been removed.
    required = [args.source / "config.json", args.meld_pickle]
    for rate in args.rates:
        suffix = f"{rate:.1f}".replace(".", "p")
        required.extend((args.source / f"best_miss_{suffix}.pt",
                         args.source / f"predictions_miss_{suffix}.npz"))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Exact source artifacts required; no fallback:\n" + "\n".join(missing))
    gpu = guard_gpu(args.gpu_uuid)
    if not args.meld_pickle.is_file():
        raise FileNotFoundError(args.meld_pickle)
    os.environ["GCNET_DATASET_ROOT"] = str(args.dataset_root)
    os.environ["GCNET_MELD_PICKLE"] = str(args.meld_pickle)
    from gcnet_missing_m3.train_gcnet import TrainConfig, _dataset_shape
    from gcnet_modality_jepa.train_gcnet import get_loaders

    config_path = args.source / "config.json"
    source_config = json.loads(config_path.read_text())
    config = TrainConfig(**source_config)
    if not (
        config.dataset == "MELD" and config.seed == 66
        and config.missing_mask_protocol == CANONICAL_MASK_PROTOCOL
        and config.backbone_type == "osram" and not config.osram_bidirectional
        and config.osram_readout_fusion == "flat"
        and config.osram_meaningful_block == "none"
        and config.training_objective == "emotion-only"
        and not config.osram_disable_local_skip
    ):
        raise ValueError("Source must be canonical seed66 causal MELD no-JEPA Flat with Local skip")
    device = torch.device("cuda:0")
    config = replace(config, device=str(device))
    torch.set_num_threads(2)
    feature_root = args.dataset_root / "MELD" / "features"
    roots = [feature_root / name for name in (
        "wav2vec-large-c-UTT", "deberta-large-4-UTT", "manet_UTT",
    )]
    # MELD loads all three features from the checked pickle; these compatibility
    # roots are unused by its loader and need not exist on disk.
    _, _, test, ad, td, vd = get_loaders(
        audio_root=str(roots[0]), text_root=str(roots[1]), video_root=str(roots[2]),
        num_folder=int(_dataset_shape(config.dataset)["num_folds"]),
        dataset=config.dataset, batch_size=config.batch_size, num_workers=0,
        seed=config.seed, validation_fraction=config.validation_fraction,
        evaluation_protocol=config.evaluation_protocol,
    )
    model = _build_model(config, (ad, td, vd), device)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    summary = {
        "status": "running", "training": False, "seed": config.seed, "gpu": gpu,
        "protocol": "INTERNAL DIAGNOSTIC ONLY; per-rate Test-oracle checkpoint; exact saved masks",
        "source": str(args.source), "source_config": source_config,
        "source_config_sha256": _sha256(config_path),
        "meld_pickle_sha256": _sha256(args.meld_pickle),
        "missing_mask_protocol": CANONICAL_MASK_PROTOCOL,
        "command": sys.argv, "max_batches": args.max_batches, "requested_rates": args.rates,
        "torch_version": torch.__version__, "rates": {},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_diff": subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT, text=True),
        "code_sha256": {str(p.relative_to(ROOT)): _sha256(p) for p in (
            Path(__file__), Path(__file__).with_name("audit.py"),
            ROOT / "gcnet_missing_m3/osram.py", ROOT / "gcnet_missing_m3/model.py",
        )},
    }
    _write(args.output / "SUMMARY.json", summary)
    try:
        for rate in args.rates:
            suffix = f"{rate:.1f}".replace(".", "p")
            checkpoint = args.source / f"best_miss_{suffix}.pt"
            predictions = args.source / f"predictions_miss_{suffix}.npz"
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(state["model"], strict=True)
            checkpoint_epoch = state.get("epoch")
            del state
            before = state_digest(model)
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            tick = time.monotonic()
            output_name = f"audit_miss_{suffix}.jsonl.gz"
            with np.load(predictions) as saved, gzip.open(args.output / output_name, "wt") as stream:
                result = evaluate_rate(model, test[config.fold - 1], config, (ad, td, vd),
                                       saved, rate, stream, args.max_batches)
            torch.cuda.synchronize(device)
            after = state_digest(model)
            if before != after or any(p.grad is not None for p in model.parameters()):
                raise RuntimeError("Frozen model changed or accumulated gradients")
            summary["rates"][f"{rate:.1f}"] = {
                **result, "seconds": time.monotonic() - tick,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "checkpoint_sha256": _sha256(checkpoint),
                "checkpoint_epoch": checkpoint_epoch,
                "saved_predictions_sha256": _sha256(predictions),
                "parameter_state_sha256_before": before, "parameter_state_sha256_after": after,
                "rows_file": output_name,
            }
            _write(args.output / "SUMMARY.json", summary)
        summary["status"] = "partial_smoke" if args.max_batches is not None else (
            "complete" if len(args.rates) == 8 else "partial_rates")
    except BaseException as error:
        summary.update(status="failed", error=repr(error))
        raise
    finally:
        summary["seconds"] = time.monotonic() - started
        _write(args.output / "SUMMARY.json", summary)


if __name__ == "__main__":
    main()
