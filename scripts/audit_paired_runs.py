#!/usr/bin/env python3
"""Audit whether two GCNet run manifests form a fair experimental pair."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from gcnet_modality_jepa.run_manifest import (  # noqa: E402
    ManifestValidationError,
    audit_paired_manifests,
    load_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("jepa", type=Path)
    args = parser.parse_args()
    try:
        baseline = load_manifest(args.baseline)
        jepa = load_manifest(args.jepa)
        mismatches = audit_paired_manifests(baseline, jepa)
    except ManifestValidationError as error:
        print("manifest validation failed: {}".format(error))
        return 2
    if mismatches:
        print("paired-run audit failed:")
        for mismatch in mismatches:
            print("- {}".format(mismatch))
        return 1
    print("paired-run audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
