"""Run the vendored causal ComP with workspace-local data and output paths."""

from __future__ import annotations

import os
import runpy
import sys
import types
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
COMP_REPO = REPOSITORY / "external_repos" / "ComP_causal"
WORKSPACE = Path(os.environ.get("PAPER_WORKSPACE", "/data2/yb/paper"))
DATA_ROOT = Path(
    os.environ.get("COMP_DATA_ROOT", str(WORKSPACE / "GCNet_TPAMI" / "dataset"))
)
OUTPUT_ROOT = Path(
    os.environ.get(
        "COMP_OUTPUT_ROOT",
        str(WORKSPACE / "05_reproduction" / "runs" / "ComP_causal" / "saved"),
    )
)


def install_runtime_config() -> None:
    data_dir = {
        "CMUMOSI": str(DATA_ROOT / "CMUMOSI"),
        "CMUMOSEI": str(DATA_ROOT / "CMUMOSEI"),
        "IEMOCAPSix": str(DATA_ROOT / "IEMOCAP"),
        "IEMOCAPFour": str(DATA_ROOT / "IEMOCAP"),
    }
    config = types.ModuleType("config")
    config.DATA_DIR = data_dir
    config.SAVED_ROOT = str(OUTPUT_ROOT)
    config.PATH_TO_FEATURES = {
        name: os.path.join(path, "features") for name, path in data_dir.items()
    }
    config.PATH_TO_LABEL = {
        "CMUMOSI": os.path.join(data_dir["CMUMOSI"], "CMUMOSI_features_raw_2way.pkl"),
        "CMUMOSEI": os.path.join(data_dir["CMUMOSEI"], "CMUMOSEI_features_raw_2way.pkl"),
        "IEMOCAPSix": os.path.join(data_dir["IEMOCAPSix"], "IEMOCAP_features_raw_6way.pkl"),
        "IEMOCAPFour": os.path.join(data_dir["IEMOCAPFour"], "IEMOCAP_features_raw_4way.pkl"),
    }
    config.MODEL_DIR = str(OUTPUT_ROOT / "model")
    config.LOG_DIR = str(OUTPUT_ROOT / "log")
    config.NPZ_DIR = str(OUTPUT_ROOT / "npz")
    sys.modules["config"] = config


def main() -> None:
    install_runtime_config()
    source_dir = COMP_REPO / "ComP"
    sys.path.insert(0, str(source_dir))
    sys.argv[0] = str(source_dir / "train_mr.py")
    runpy.run_path(str(source_dir / "train_mr.py"), run_name="__main__")


if __name__ == "__main__":
    main()
