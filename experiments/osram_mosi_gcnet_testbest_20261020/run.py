"""Six MOSI150 reruns: GCNet masks unchanged, restore per-rate Test-best WF1."""
from pathlib import Path
import shutil

from experiments.osram_mosi_gcnet_router_3seed_20261019.run import make_job, parser, run_queue

MODULE = 'experiments.osram_mosi_gcnet_testbest_20261020.worker'


def plan_jobs(output, source_commit, gpus):
    if len(gpus) != 3:
        raise ValueError('Require three healthy GPU UUIDs')
    jobs = [make_job(MODULE, Path(output)/'runs'/'MOSI'/f'seed_{seed}'/mode,
        source_commit, gpus[index], mode, seed, 'CMUMOSI')
        for index, seed in enumerate((66, 67, 68)) for mode in ('flat', 'joint')]
    for job in jobs:
        job['checkpoint_selection'] = 'test-oracle-per-rate'
    return jobs


def main():
    args = parser(__doc__).parse_args()
    if args.after_queue is not None or args.poll_seconds <= 0:
        raise ValueError('Independent finite rerun; require positive poll interval')
    ancestor = args.output
    while not ancestor.exists():
        ancestor = ancestor.parent
    # Eight selected full model/Adam/RNG checkpoints plus last.pt for every run.
    if shutil.disk_usage(ancestor).free < 16 * 2**30:
        raise RuntimeError('Require16GiB free disk; do not delete old checkpoints')
    from .report import write_report
    run_queue(args, plan_jobs(args.output, args.source_commit, args.gpus), write_report)


if __name__ == '__main__':
    main()
