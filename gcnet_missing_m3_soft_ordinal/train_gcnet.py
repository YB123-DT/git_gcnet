"""Locked soft-ordinal entry point over the shared Missing-M3 trainer."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from gcnet_missing_m3 import train_gcnet as base_train


def main(argv: Sequence[str] | None = None) -> None:
    """Run the shared trainer with the treatment task mode locked."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if any(
        argument == "--mosi-task-mode"
        or argument.startswith("--mosi-task-mode=")
        for argument in arguments
    ):
        raise ValueError("soft-ordinal version owns mosi_task_mode")
    base_train.main(
        [*arguments, "--mosi-task-mode", "soft-ordinal"]
    )


if __name__ == "__main__":
    main()
