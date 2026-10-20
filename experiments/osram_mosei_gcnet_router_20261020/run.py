"""Two finite MOSEI seed66 runs, one resource-admitted healthy GPU."""
import argparse
from pathlib import Path
import shutil

from experiments.osram_mosi_gcnet_router_3seed_20261019.run import make_job, run_queue, BAD_UUID

MODULE = 'experiments.osram_mosei_gcnet_router_20261020.worker'


def plan_jobs(output, source_commit, gpus):
    if len(gpus) != 1 or gpus[0] == BAD_UUID:
        raise ValueError('Require exactly one healthy GPU UUID; GPU4 forbidden')
    jobs = [make_job(MODULE, Path(output)/'runs'/'CMUMOSEI'/'seed_66'/mode,
        source_commit, gpus[0], mode, 66, 'CMUMOSEI') for mode in ('flat', 'joint')]
    for job in jobs:
        job['checkpoint_selection'] = 'test-oracle-per-rate'
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--poll-seconds', type=float, default=10)
    args = parser.parse_args()
    args.gpus, args.after_queue = [args.gpu_uuid], None
    if not args.gpu_uuid.startswith('GPU-') or args.poll_seconds <= 0:
        raise ValueError('Require actual GPU UUID and positive poll interval')
    ancestor = args.output
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free < 16 * 2**30:
        raise RuntimeError('Require16GiB free disk; never delete old runs automatically')
    from .report import write_report
    run_queue(args, plan_jobs(args.output, args.source_commit, args.gpus), write_report)


if __name__ == '__main__':
    main()
