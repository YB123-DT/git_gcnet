"""Select/reserve at most the remaining twenty; never starts a process.

The resulting manifest is consumed unchanged by the existing round-two
snapshot, CUDA readiness and single-run entry points. No new trainer.
"""
import argparse
import copy
import fcntl
import json
from pathlib import Path

from experiments.osram_meaningful20_20261003.manifest import read, write
from experiments.osram_meaningful20_round2_20261004.manifest import validate_round


def select(catalog, candidates, registry):
    from gcnet_missing_m3.meaningful_new40_registry import NEW40_METHODS
    cards = catalog.get('cards', [])
    by_id = {card['id']: card for card in cards}
    if len(cards) != 40 or set(by_id) != set(NEW40_METHODS):
        raise ValueError('Catalog must contain exactly the forty implemented IDs')
    if (not candidates or len(candidates) > 20 or len(set(candidates)) != len(candidates)
            or not set(candidates) <= set(by_id)):
        raise ValueError('Select 1..20 unique implemented candidates')
    reserved = {m['id'] for r in registry.get('rounds', {}).values()
                for m in r.get('methods', [])}
    limit = min(60, registry.get('max_distinct_trained_methods', 60))
    if len(reserved | set(candidates)) > limit:
        raise ValueError('Selection exceeds the global sixty-method training cap')
    manifest = {
        'round_id': '3_additional40_selection', 'target_count': 20,
        'status': 'source_accepted', 'label': 'INTERNAL DIAGNOSTIC ONLY',
        'cards': [copy.deepcopy(by_id[name]) for name in candidates],
    }
    validate_round(manifest)
    return manifest


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, default=Path(__file__).with_name('CATALOG.json'))
    parser.add_argument('--registry', type=Path, default=root / 'experiments/osram_method_registry.json')
    parser.add_argument('--candidate', action='append', required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not args.dry_run and args.output is None:
        parser.error('--output is required without --dry-run')
    if args.dry_run:
        print(json.dumps(select(read(args.catalog), args.candidate, read(args.registry)), indent=2))
        return
    with args.registry.with_suffix('.selection.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.output.exists():
            raise FileExistsError('Never overwrite a previously selected manifest')
        registry = read(args.registry)
        manifest = select(read(args.catalog), args.candidate, registry)
        reservation = registry['rounds'].setdefault('3', {
            'status': 'reserved_not_trained', 'methods': [],
            'source_directory': 'osram_new40_20261004',
        })
        known = {m['id'] for m in reservation['methods']}
        reservation['methods'].extend(copy.deepcopy(c) for c in manifest['cards'] if c['id'] not in known)
        # Reserve first: if writing the manifest fails, slots remain reserved,
        # never allowing a second selection to exceed the experiment budget.
        write(args.registry, registry)
        write(args.output, manifest)


if __name__ == '__main__':
    main()
