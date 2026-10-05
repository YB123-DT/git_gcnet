"""Record hashes of an already extracted git archive, never a live worktree."""
import argparse
from pathlib import Path

from experiments.osram_core20_20261005.run import sha, write, now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if (root / '.git').exists() or (root / 'SNAPSHOT.json').exists():
        raise ValueError('Use a fresh immutable git-archive directory')
    files = sorted(p for p in root.rglob('*') if p.is_file()
                   and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts)
    write(root / 'SNAPSHOT.json', dict(code_commit=args.commit, created_utc=now(),
          source_sha256={str(p.relative_to(root)): sha(p) for p in files}))


if __name__ == '__main__':
    main()
