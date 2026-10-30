"""Create/verify a small immutable source bundle, never archive experiment data."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

PACKAGES = ("gcnet_missing_m3/", "gcnet_modality_jepa/", "gcnet_jepa_replacement/",
            "gcnet_plci_jepa/", "experiments/cam_unified_20261030/")
EXTRA = ("config.py", "experiments/__init__.py",
         "experiments/osram_frozen_nested_gcnet_20261025/__init__.py",
         "experiments/osram_frozen_nested_gcnet_20261025/model.py")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root):
    root = Path(root)
    manifest = json.loads((root / "SOURCE.json").read_text())
    failures = [name for name, expected in manifest["files"].items()
                if not (root / name).is_file() or digest(root / name) != expected]
    if failures:
        raise ValueError("Immutable source changed: " + str(failures))
    return manifest


def create(root, archive):
    root, archive = Path(root), Path(archive)
    if archive.exists():
        raise FileExistsError(archive)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    files = subprocess.check_output(["git", "ls-files"], cwd=root, text=True).splitlines()
    selected = sorted(name for name in files if (
        (name.startswith(PACKAGES) and name.endswith((".py", ".md")))
        or name in EXTRA or (name.startswith("tests/test_cam_unified_") and name.endswith(".py"))))
    # Never deploy uncommitted behavior. Unrelated user work remains untouched.
    changed = subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=root, text=True).splitlines()
    if set(changed) & set(selected):
        raise ValueError("Selected source must first be committed")
    if not any(name.endswith("/train.py") and "/cam_unified_" in name for name in selected):
        raise ValueError("Unified training entry has not been committed")
    manifest = dict(version=1, source_commit=sha, files={name: digest(root / name) for name in selected})
    payload = (json.dumps(manifest, indent=2) + "\n").encode()
    with tarfile.open(archive, "x:gz") as tar:
        for name in selected:
            tar.add(root / name, arcname=name, recursive=False)
        info = tarfile.TarInfo("SOURCE.json")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    return dict(source_commit=sha, source_files=len(selected), archive=str(archive),
                archive_sha256=digest(archive), bytes=archive.stat().st_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify == bool(args.archive):
        parser.error("Choose exactly one of --archive or --verify")
    result = verify(args.root) if args.verify else create(args.root, args.archive)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
