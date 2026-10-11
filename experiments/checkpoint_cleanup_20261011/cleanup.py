"""Delete only the user's approved batches 2/3/4, retaining hashed evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket

ROOTS = [
    Path('/data2/yb/remote_experiments/osram_medium_flat_nested_20261009/runs'),
    Path('/data1/yb/remote_experiments/osram_nested_direct_capacity_20261011/runs'),
    Path('/data1/yb/remote_experiments/osram_nested_direct_capacity_more_20261011/lane0/runs'),
    Path('/data1/yb/remote_experiments/osram_nested_direct_capacity_more_20261011/lane1/runs'),
]
WEIGHTS = {'.pt', '.pth', '.ckpt', '.safetensors'}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, data):
    with path.open('w') as stream:
        json.dump(data, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    # Fail closed if any live Python process references these experiment batches.
    for process in Path('/proc').iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            parts = (process / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if parts and b'python' in Path(parts[0].decode(errors='replace')).name.encode():
            command = b' '.join(parts).decode(errors='replace')
            assert not any(str(r.parent) in command for r in ROOTS), command
    files, evidence, runs = [], [], []
    for root in ROOTS:
        assert root.is_dir() and not root.is_symlink()
        children = sorted(root.glob('*/seed_*'))
        assert children
        for run in children:
            assert not run.is_symlink()
            provenance = json.loads((run / 'PROVENANCE.json').read_text())
            assert provenance['status'] == 'complete' and provenance['outputs_verified']
            assert (run / 'metrics.json').is_file() and (run / 'config.json').is_file()
            assert len(list(run.glob('predictions_miss_*.npz'))) == 8
            runs.append(str(run))
            for path in sorted(run.rglob('*')):
                assert not path.is_symlink(), str(path)
                if not path.is_file():
                    continue
                assert path.resolve().is_relative_to(root.resolve())
                stat = path.stat()
                row = dict(path=str(path), size=stat.st_size, sha256=sha(path),
                           inode=stat.st_ino, device=stat.st_dev, links=stat.st_nlink)
                (files if path.suffix in WEIGHTS else evidence).append(row)
    assert len(runs) == 18 and len(files) == 647, (len(runs), len(files))
    report = dict(status='inventoried', host=socket.gethostname(), runs=runs,
                  checkpoints=files, retained_evidence=evidence,
                  requested_batches=[2, 3, 4], script_sha256=sha(Path(__file__)),
                  free_before={p: shutil.disk_usage(p).free for p in ('/data1', '/data2')})
    save(args.output / 'MANIFEST.json', report)
    print('Inventory saved:', len(files), 'checkpoints, bytes:', sum(r['size'] for r in files), flush=True)
    if not args.execute:
        return
    # No recursive directory deletion. Validate exact inode/content immediately
    # before unlink; a partial failure remains traceable to the durable manifest.
    for row in files:
        path = Path(row['path'])
        stat = path.stat()
        assert (stat.st_ino, stat.st_dev, stat.st_size) == (row['inode'], row['device'], row['size'])
        assert sha(path) == row['sha256']
        path.unlink()
    for row in evidence:
        path = Path(row['path'])
        assert path.is_file() and sha(path) == row['sha256'], str(path)
    assert all(not Path(row['path']).exists() for row in files)
    report.update(status='complete', deleted_files=len(files),
                  deleted_logical_bytes=sum(r['size'] for r in files),
                  evidence_verified=len(evidence),
                  free_after={p: shutil.disk_usage(p).free for p in ('/data1', '/data2')})
    save(args.output / 'MANIFEST.json', report)
    summary = {k: v for k, v in report.items() if k not in ('checkpoints', 'retained_evidence')}
    save(args.output / 'SUMMARY.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
