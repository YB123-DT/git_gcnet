"""Source/design manifests and immutable execution snapshots (standard library)."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

FAMILIES = ('graph', 'hypergraph', 'grouping', 'set_context', 'optimization', 'feature_reasoning')
REQUIRED_SNAPSHOT_JSON = (
    'experiments/osram_current_history_relation_20261003/reference/config.json',
    'experiments/osram_meaningful20_20261003/ROUND1.json',
    'experiments/osram_meaningful20_20261003/BASELINE_AUDIT.json',
    *(f'experiments/osram_meaningful20_20261003/{family}.json' for family in FAMILIES),
)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def validate_round(manifest, prior_design_hashes=()):
    cards = manifest.get('cards', [])
    if manifest.get('status') != 'source_accepted' or not manifest.get('round_id') or len(cards) != 20:
        raise ValueError('A round needs exactly twenty accepted source/design cards')
    ids = [card.get('id') for card in cards]
    designs = [card.get('design_sha256') for card in cards]
    if (len(set(ids)) != 20 or len(set(designs)) != 20
            or any(not isinstance(i, str) or not re.fullmatch(r'[a-z0-9_]+', i) for i in ids)):
        raise ValueError('Twenty unique safe candidate IDs and design hashes are required')
    for card in cards:
        if not card.get('accepted') or not re.fullmatch('[0-9a-f]{64}', card.get('design_sha256', '')):
            raise ValueError('Rejected/unreviewed designs cannot enter a round')
        if not all(card.get('source', {}).get(key) for key in ('paper', 'code')):
            raise ValueError('Primary paper and inspected author-code evidence are required')
        if card['design_sha256'] in prior_design_hashes:
            raise ValueError('An earlier design cannot fill a new-round quota')
    return ids


def freeze_cards(directory, round_id):
    directory = Path(directory)
    cards = []
    for family in FAMILIES:
        data = read(directory / f'{family}.json')
        entries = data.get('accepted', data.get('candidates', data.get('cards', data.get('papers', []))))
        for card in entries:
            decision = str(card.get('decision', card.get('accept_or_reject', 'accept' if 'accepted' in data else ''))).lower()
            if not decision.startswith('accept'): continue
            source = {'paper': card.get('url'),
                      'code': card.get('code_url') or card.get('codeURL') or card.get('code_evidence')}
            cards.append({'id': card['id'], 'family': family, 'accepted': True,
                          'design_sha256': digest_json(card), 'source': source,
                          'card': card, 'card_file_sha256': sha(directory / f'{family}.json')})
    manifest = {'round_id': round_id, 'status': 'source_accepted', 'cards': cards,
                'created_at': now(), 'label': 'INTERNAL TEST-ORACLE ADAPTIVE SEARCH; NOT A PAPER CLAIM'}
    validate_round(manifest)
    return manifest


def build_data_manifest(dataset_root):
    dataset_root = Path(dataset_root).resolve()
    roots = [dataset_root / 'CMUMOSI/features' / name for name in
             ('wav2vec-large-c-UTT', 'deberta-large-4-UTT', 'manet_UTT')]
    label = dataset_root / 'CMUMOSI/CMUMOSI_features_raw_2way.pkl'
    if not label.is_file() or any(not root.is_dir() for root in roots):
        raise ValueError('Canonical MOSI labels/splits and all three feature trees are required')
    files = {str(label): sha(label)}
    for root in roots:
        paths = sorted(p for p in root.rglob('*') if p.is_file())
        if not paths: raise ValueError(f'Empty feature tree: {root}')
        for path in paths:
            if not path.resolve().is_relative_to(dataset_root): raise ValueError('Feature escapes dataset root')
            files[str(path.resolve())] = sha(path)
    return {'dataset_root': str(dataset_root), 'feature_roots': [str(p) for p in roots],
            'split_files': [str(label)], 'files': files, 'created_at': now()}


def create_snapshot(source, destination, *, files, code_commit, scoped_diff='', committed_overrides=None):
    source, destination = Path(source).resolve(), Path(destination).absolute()
    if not re.fullmatch('[0-9a-f]{40}', code_commit):
        raise ValueError('Snapshot requires a full forty-character code commit')
    if destination.exists(): raise FileExistsError(destination)
    committed_overrides = committed_overrides or {}
    if set(committed_overrides) - set(files): raise ValueError('Override must name an included source file')
    checked = []
    for name in sorted(set(files)):
        relative = Path(name)
        path = source / relative
        if relative.is_absolute() or '..' in relative.parts or path.is_symlink() or not path.is_file():
            raise ValueError(f'Unsafe/missing snapshot source: {name}')
        if not path.resolve().is_relative_to(source): raise ValueError('Source escapes repository')
        override = committed_overrides.get(name)
        if override is not None and not isinstance(override, bytes): raise ValueError('Committed source must be bytes')
        checked.append((name, path, sha(path) if override is None else hashlib.sha256(override).hexdigest(), override))
    destination.mkdir(parents=True)
    hashes = {}
    for name, path, expected, override in checked:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if override is None: shutil.copyfile(path, target)
        else: target.write_bytes(override)
        if sha(target) != expected or (override is None and sha(path) != expected):
            raise ValueError('Source changed during snapshot copy')
        hashes[name] = expected
    record = {'code_commit': code_commit, 'source_sha256': hashes, 'created_at': now(),
              'scoped_diff': scoped_diff, 'source_root': str(source),
              'committed_overrides': sorted(committed_overrides)}
    write(destination / 'SNAPSHOT.json', record)
    verify_snapshot(destination)
    return record


def verify_snapshot(root):
    root = Path(root).resolve()
    record = read(root / 'SNAPSHOT.json')
    if not record.get('source_sha256'): raise ValueError('Empty source snapshot')
    for name, expected in record['source_sha256'].items():
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file() or sha(path) != expected:
            raise ValueError(f'Immutable source hash mismatch: {name}')
    extras = {str(path.relative_to(root)) for path in root.rglob('*.py')} - record['source_sha256'].keys()
    if extras: raise ValueError(f'Unrecorded executable source: {sorted(extras)}')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    freeze = sub.add_parser('freeze')
    freeze.add_argument('--cards', type=Path, required=True)
    freeze.add_argument('--round-id', required=True)
    freeze.add_argument('--output', type=Path, required=True)
    snapshot = sub.add_parser('snapshot')
    snapshot.add_argument('--source', type=Path, required=True)
    snapshot.add_argument('--output', type=Path, required=True)
    snapshot.add_argument('--allow-dirty', nargs='*', default=[])
    snapshot.add_argument('--committed-file', nargs='*', default=[],
                          help='Use HEAD bytes for explicitly excluded unrelated dirty files')
    data = sub.add_parser('data')
    data.add_argument('--dataset-root', type=Path, required=True)
    data.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'freeze':
        if args.output.exists(): raise FileExistsError('Round manifest is immutable')
        write(args.output, freeze_cards(args.cards, args.round_id))
    elif args.action == 'data':
        if args.output.exists(): raise FileExistsError('Data manifest is immutable')
        write(args.output, build_data_manifest(args.dataset_root))
        from .run import validate_data_manifest
        validate_data_manifest(args.output)
    else:
        source = args.source.resolve()
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
        tracked = subprocess.check_output(['git', 'ls-files', '*.py'], cwd=source, text=True).splitlines()
        files = set(tracked)
        files.update(REQUIRED_SNAPSHOT_JSON)
        for folder in ('gcnet_missing_m3', 'experiments/osram_meaningful20_20261003'):
            files.update(str(p.relative_to(source)) for p in (source / folder).rglob('*.py'))
        files.update(str(p.relative_to(source)) for p in (source / 'tests').glob('test_meaningful*.py'))
        dirty = set(subprocess.check_output(['git', 'diff', 'HEAD', '--name-only'], cwd=source, text=True).splitlines()) & files
        if set(args.allow_dirty) & set(args.committed_file): raise ValueError('Conflicting snapshot policies')
        if dirty - set(args.allow_dirty) - set(args.committed_file):
            raise ValueError(f'Commit source edits or explicitly scope dirty paths: {sorted(dirty)}')
        included_dirty = dirty - set(args.committed_file)
        diff = subprocess.check_output(['git', 'diff', 'HEAD', '--', *sorted(included_dirty)], cwd=source, text=True) if included_dirty else ''
        overrides = {name: subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=source)
                     for name in args.committed_file}
        create_snapshot(source, args.output, files=files, code_commit=commit, scoped_diff=diff,
                        committed_overrides=overrides)


if __name__ == '__main__': main()
