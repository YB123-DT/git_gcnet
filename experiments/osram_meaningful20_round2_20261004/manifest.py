"""Immutable accepted round-two subsets; snapshot/data mechanics are shared."""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import subprocess

from experiments.osram_meaningful20_20261003.manifest import (
    REQUIRED_SNAPSHOT_JSON, build_data_manifest, create_snapshot, digest_json,
    now, read, sha, verify_snapshot, write)

PACKAGE = 'experiments/osram_meaningful20_round2_20261004'
TARGET_COUNT = 20


def validate_round(manifest, prior_design_hashes=()):
    cards = manifest.get('cards', [])
    if (manifest.get('status') != 'source_accepted' or not manifest.get('round_id')
            or manifest.get('target_count') != TARGET_COUNT or not 1 <= len(cards) <= TARGET_COUNT):
        raise ValueError('Round two requires an immutable accepted subset with target_count=20')
    ids = [card.get('id') for card in cards]
    designs = [card.get('design_sha256') for card in cards]
    if (len(set(ids)) != len(cards) or len(set(designs)) != len(cards)
            or any(not isinstance(name, str) or not re.fullmatch('[a-z0-9_]+', name) for name in ids)):
        raise ValueError('Unique safe candidate IDs and distinct designs are required')
    for card in cards:
        novelty = card.get('novelty_audit', {})
        novelty_keys = {'closest_prior_ids', 'substantive_difference',
                        'same_round_overlap', 'source_code_evidence'}
        if (card.get('accepted') is not True or not re.fullmatch('[0-9a-f]{64}', card.get('design_sha256', ''))
                or not all(card.get('source', {}).get(key) for key in ('paper', 'code'))
                or card['design_sha256'] in prior_design_hashes
                or not isinstance(novelty, dict) or not novelty_keys <= novelty.keys()
                or not novelty['closest_prior_ids'] or not novelty['substantive_difference']
                or not novelty['source_code_evidence']):
            raise ValueError('Only accepted, evidenced, genuinely new designs can run')
    return ids


def validate_extension(previous, current):
    old_ids, new_ids = validate_round(previous), validate_round(current)
    if previous['round_id'] != current['round_id'] or new_ids[:len(old_ids)] != old_ids:
        raise ValueError('A manifest revision may only append accepted candidates in the same round')
    if current['cards'][:len(old_ids)] != previous['cards']:
        raise ValueError('An accepted candidate cannot change under an existing run')
    return new_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    snapshot = sub.add_parser('snapshot')
    for key in ('source', 'manifest', 'output'):
        snapshot.add_argument('--' + key, type=Path, required=True)
    snapshot.add_argument('--committed-file', nargs='*', default=[])
    snapshot.add_argument('--extra-json', nargs='*', default=[])
    data = sub.add_parser('data')
    data.add_argument('--dataset-root', type=Path, required=True)
    data.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'data':
        if args.output.exists(): raise FileExistsError('Data manifest is immutable')
        write(args.output, build_data_manifest(args.dataset_root))
        from experiments.osram_meaningful20_20261003.run import validate_data_manifest
        validate_data_manifest(args.output)
        return
    source = args.source.resolve()
    validate_round(read(args.manifest))
    manifest_name = str(args.manifest.resolve().relative_to(source))
    files = set(subprocess.check_output(['git', 'ls-files', '*.py'], cwd=source, text=True).splitlines())
    files.update(REQUIRED_SNAPSHOT_JSON)
    files.update((manifest_name, 'experiments/osram_method_registry.json'))
    files.update(args.extra_json)
    files.update(str(path.relative_to(source)) for path in (source / PACKAGE).glob('*.json')
                 if path.name != 'STATUS.json')
    for folder in ('gcnet_missing_m3', PACKAGE):
        files.update(str(path.relative_to(source)) for path in (source / folder).rglob('*.py'))
    dirty = set(subprocess.check_output(['git', 'diff', 'HEAD', '--name-only'], cwd=source, text=True).splitlines()) & files
    if dirty - set(args.committed_file):
        raise ValueError(f'Commit round-two source before freezing: {sorted(dirty - set(args.committed_file))}')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    overrides = {name: subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=source)
                 for name in args.committed_file}
    create_snapshot(source, args.output, files=files, code_commit=commit, committed_overrides=overrides)


if __name__ == '__main__': main()
