"""Run one completed GCNet version through the shared trainer."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

from versions.registry import VERSION_NAMES, resolve


ROOT = Path(__file__).resolve().parent
TRAINER = ROOT / "common" / "gcnet" / "train_gcnet.py"
LOCKED_FLAGS = {
    "branch_fusion": "--branch-fusion",
    "graph_conv_variant": "--graph-conv-variant",
    "reconstruction_target": "--reconstruction-target",
}


def _contains_flag(arguments, flag):
    return any(value == flag or value.startswith(flag + "=") for value in arguments)


def build_command(version, user_arguments):
    locked = resolve(version)
    for key, flag in LOCKED_FLAGS.items():
        if _contains_flag(user_arguments, flag):
            raise ValueError(
                "locked argument {} is selected by version {!r}".format(
                    flag, version
                )
            )

    command = [sys.executable, str(TRAINER)]
    for key in sorted(locked):
        command.extend([LOCKED_FLAGS[key], str(locked[key])])
    command.extend(user_arguments)
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--version", required=True, choices=VERSION_NAMES)
    known, remaining = parser.parse_known_args(argv)
    command = build_command(known.version, remaining)

    env = os.environ.copy()
    python_paths = [str(ROOT), str(ROOT / "common" / "gcnet")]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    return subprocess.call(
        command,
        cwd=str(ROOT / "common" / "gcnet"),
        env=env,
    )


if __name__ == "__main__":
    raise SystemExit(main())
