"""Run causal or noncausal ComP on the ECERC MELD feature pickle."""

import os
import runpy
import sys
import types
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
COMP_SOURCE = REPOSITORY / "external_repos" / "ComP_causal" / "ComP"
WORKSPACE = Path(os.environ.get("PAPER_WORKSPACE", "/data2/yb/paper"))
MELD_PICKLE = Path(os.environ.get(
    "MELD_ECERC_PICKLE",
    str(WORKSPACE / "baseline_sources_20261006" / "ECERC" / "data" / "meld"
        / "meld_emotion_semantic_features_roberta.pkl"),
))
OUTPUT_ROOT = Path(os.environ.get(
    "COMP_OUTPUT_ROOT",
    str(WORKSPACE / "05_reproduction" / "runs" / "ComP_MELD_20261012"),
))


def install_runtime_config():
    config = types.ModuleType("config")
    config.PATH_TO_LABEL = {"MELD": str(MELD_PICKLE)}
    config.PATH_TO_FEATURES = {"MELD": str(MELD_PICKLE.parent)}
    config.MODEL_DIR = str(OUTPUT_ROOT / "model")
    config.LOG_DIR = str(OUTPUT_ROOT / "log")
    config.NPZ_DIR = str(OUTPUT_ROOT / "npz")
    sys.modules["config"] = config


def main():
    if not MELD_PICKLE.is_file():
        raise FileNotFoundError(MELD_PICKLE)
    install_runtime_config()
    sys.path.insert(0, str(COMP_SOURCE))
    sys.argv[0] = str(COMP_SOURCE / "train_mr.py")
    runpy.run_path(str(COMP_SOURCE / "train_mr.py"), run_name="__main__")


if __name__ == "__main__":
    main()
