"""Finite corrected IEMOCAP6 suite: five folds, three seeds, Flat/Router."""
from pathlib import Path
import shutil

from experiments.osram_mosi_gcnet_router_3seed_20261019.run import make_job, parser, run_queue

MODULE = 'experiments.osram_iemocap6_gcnet_testbest_20261020.worker'


def plan_jobs(output, source_commit, gpus):
    if len(gpus) != 3 or len(set(gpus)) != 3:
        raise ValueError('Require three distinct healthy GPU UUIDs')
    jobs = [make_job(MODULE, Path(output)/'runs'/'IEMOCAPSix'/f'fold_{fold}'/f'seed_{seed}'/mode,
        source_commit, gpus[index], mode, seed, 'IEMOCAPSix', fold=fold)
        for fold in range(1, 6) for index, seed in enumerate((66, 67, 68))
        for mode in ('flat', 'joint')]
    for job in jobs:
        job['checkpoint_selection'] = 'test-oracle-per-rate'
    return jobs


def main():
    args = parser(__doc__).parse_args()
    if args.poll_seconds <= 0:
        raise ValueError('Require a positive poll interval')
    ancestor = args.output
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free < 64 * 2**30:
        raise RuntimeError('Require64GiB free for30runs with eight fullBEST and last each')
    from .report import write_report
    run_queue(args, plan_jobs(args.output, args.source_commit, args.gpus), write_report)


if __name__ == '__main__':
    main()
