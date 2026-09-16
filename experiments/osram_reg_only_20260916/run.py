"""Run one fixed-teacher regression-only Stage-2 candidate job."""
import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments.osram_supervised_teacher_20260914 import run as base


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int, choices=base.runner.SEEDS)
    args = parser.parse_args()
    base.run("student", args.seed, "joint-reg-only")
