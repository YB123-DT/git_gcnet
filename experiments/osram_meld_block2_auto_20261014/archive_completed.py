"""Move explicitly named completed runs across local filesystems without loss.

Operations-only helper: never imported by the immutable training workers.
Original paths remain usable through symlinks; all file bytes are verified.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def inventory(root):
    records = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError(f'Unexpected existing symlink: {path}')
        if path.is_file():
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(chunk)
            records[str(path.relative_to(root))] = digest.hexdigest()
    return records


def archive(root, destination, jobs):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if root == destination or root in destination.parents or destination in root.parents:
        raise ValueError('Archive must be outside the original run tree')
    records = []
    for job in jobs:
        if not job.startswith('hpo_') or '/' in job or '..' in job:
            raise ValueError('Only explicit HPO job names are accepted')
        source, target = root / job, destination / job
        if source.is_symlink():
            raise ValueError(f'Already archived: {source}')
        summary = json.loads((source / 'SUMMARY.json').read_text())
        if summary.get('status') != 'complete':
            raise ValueError(f'Run has not completed: {source}')
        if any(not (source / f'best_miss_0p{i}.pt').is_file() for i in range(8)):
            raise ValueError(f'Missing protected winning weights: {source}')
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
        before = inventory(source)
        records.append((source, target, before))
    destination.mkdir(parents=True, exist_ok=True)
    for source, target, before in records:
        shutil.move(str(source), str(target))
        source.symlink_to(target, target_is_directory=True)
        after = inventory(target)
        if before != after:
            raise RuntimeError(f'Archive verification failed; retain both paths: {target}')
        print(json.dumps({'source': str(source), 'destination': str(target),
                          'files': len(after), 'sha256': after, 'verified': True}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--jobs', nargs='+', required=True)
    args = parser.parse_args()
    archive(args.root, args.destination, args.jobs)
