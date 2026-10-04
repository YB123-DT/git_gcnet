"""Shared durable queue with append-only subsets and a full-twenty ranking gate."""
from __future__ import annotations
import fcntl
from pathlib import Path

from experiments.osram_meaningful20_20261003 import queue as shared
from .manifest import now, read, sha, validate_extension, validate_round, write


def phase(state, ids, baseline):
    state['accepted_count'] = len(ids)
    state['target_count'] = 20
    if len(ids) < 20:
        state['subset_complete'] = all(state['jobs'].get(f'{name}:66', {}).get('status') == 'complete' for name in ids)
        return
    shared._phase(state, ids, baseline)


def bind_manifest_version(args):
    """Called under both control and queue locks before the shared coordinator."""
    current = read(args.manifest)
    validate_round(current)
    spec = {'manifest_sha256': sha(args.manifest), 'baseline_audit_sha256': sha(args.baseline_audit),
            'data_manifest_sha256': sha(args.data_manifest), 'reference_root': str(args.reference_root.resolve())}
    path = args.root / 'QUEUE.json'
    state = read(path) if path.exists() else {'spec': spec, 'phase': 'screening', 'jobs': {}, 'created_at': now()}
    versions = state.setdefault('manifest_versions', [])
    for version in versions:
        if sha(version['path']) != version['sha256']:
            raise ValueError('A previously accepted manifest was modified')
    if versions:
        previous = versions[-1]
        validate_extension(read(previous['path']), current)
    elif state['jobs']:
        raise ValueError('Existing jobs lack an immutable round-two manifest lineage')
    if any(state['spec'].get(key) != value for key, value in spec.items() if key != 'manifest_sha256'):
        raise ValueError('Reference/data identity cannot change while extending a subset')
    if state['spec']['manifest_sha256'] != spec['manifest_sha256'] and state['phase'] != 'screening':
        raise ValueError('Cannot extend a round after promotion has begun')
    if not versions or versions[-1]['sha256'] != spec['manifest_sha256']:
        versions.append({'path': str(args.manifest.resolve()), 'sha256': spec['manifest_sha256']})
    state['spec'] = spec
    write(path, state)


def coordinate(args):
    if not 1 <= args.max_concurrent <= 3 or args.poll_seconds < 1:
        raise ValueError('Round two permits at most three concurrent formal runs')
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / 'round2-control.lock').open('a') as control:
        fcntl.flock(control, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (args.root / 'queue.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            bind_manifest_version(args)
        return shared.coordinate(args, round_validator=validate_round,
            runner_module='experiments.osram_meaningful20_round2_20261004.run', phase_fn=phase)


def parser():
    return shared.parser()


if __name__ == '__main__': coordinate(parser().parse_args())
