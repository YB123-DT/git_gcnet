"""One-epoch real MOSI Text-Core smoke; not a full experiment."""

from dataclasses import replace
import json
from pathlib import Path

from .run import FEATURES, configuration


def main():
    import torch

    from gcnet_missing_m3.train_gcnet import run_experiment

    cfg, _ = configuration(66)
    cfg = replace(cfg, epochs=1)
    output = Path("/tmp/osram_text_core_smoke")
    if output.exists():
        import shutil

        shutil.rmtree(output)
    torch.set_num_threads(6)
    roots = [
        str(FEATURES / name)
        for name in ("wav2vec-large-c-UTT", "deberta-large-4-UTT", "manet_UTT")
    ]
    result = run_experiment(cfg, *roots, output_dir=str(output))
    assert result["training_objective"] == "emotion-only"
    assert result["selection_protocol"] == "per-rate-test-oracle"
    for rate, metrics in result["test"].items():
        assert metrics["text_core_real_slot"] is not None
        assert metrics["text_core_pred_target_std_ratio"] >= 0.0
    print(json.dumps({"status": "ok", "parameter_count": result["parameter_count"]}))


if __name__ == "__main__":
    main()
