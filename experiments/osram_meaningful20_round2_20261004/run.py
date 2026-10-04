"""Shared full-resume runner, with each run pinned to its original subset."""
from __future__ import annotations
import fcntl
from pathlib import Path

from experiments.osram_meaningful20_20261003 import run as shared
from .manifest import now, read, validate_extension, validate_round, write

parser = shared.parser
completion_status = shared.completion_status
candidate_config = shared.candidate_config
bind_data_environment = shared.bind_data_environment
INPUT_PATHS = ('manifest', 'readiness', 'snapshot', 'reference_root', 'baseline_audit', 'data_manifest')
INPUT_HASHES = ('manifest_sha256', 'readiness_sha256', 'baseline_audit_sha256', 'data_manifest_sha256')


def bind_launch_inputs(args):
    """Appending accepted methods cannot change a previous run's resume identity."""
    shared.verify_launch_hashes(args)
    validate_round(read(args.manifest))
    binding = {key: str(getattr(args, key).resolve()) for key in INPUT_PATHS}
    binding.update({key: getattr(args, key) for key in INPUT_HASHES})
    binding.update(candidate=args.candidate, seed=args.seed, run_id=args.run_id)
    path = args.output.with_name(args.output.name + '.launch-inputs.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if path.exists():
            saved = read(path)
            changing = {'manifest', 'manifest_sha256'}
            if any(saved.get(key) != value for key, value in binding.items() if key not in changing):
                raise ValueError('A run cannot change its pinned source/data/readiness inputs')
            # Admission-only failures may retry before creating a checkpoint.
            # They retain their first binding just like full-state resumes.
            validate_extension(read(saved['manifest']), read(args.manifest))
            args.manifest = Path(saved['manifest'])
            args.manifest_sha256 = saved['manifest_sha256']
        elif args.resume:
            raise ValueError('Missing original round-two launch binding; refuse weights-only recovery')
        else:
            write(path, binding)
    shared.verify_launch_hashes(args)


def train(args):
    bind_launch_inputs(args)
    return shared.train(args, round_validator=validate_round)


def main():
    args = parser().parse_args()
    try:
        train(args)
    except BaseException as error:
        write(args.output.with_name(args.output.name + '.launch-failure.json'),
              {'run_id': args.run_id, 'at': now(), 'error': repr(error),
               'failure_category': 'interrupted' if isinstance(error, (KeyboardInterrupt, SystemExit))
               else 'preflight_or_execution'})
        raise


if __name__ == '__main__': main()
