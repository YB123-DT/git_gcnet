"""One finite seed66 Flat/Router pair; no extra seeds or automatic HPO."""
import argparse
from pathlib import Path
import shutil

from experiments.osram_mosi_gcnet_router_3seed_20261019.run import BAD_UUID, make_job, run_queue

MODULE='experiments.osram_mosi_router_lr1e5_20261020.worker'


def plan_jobs(output, source_commit, gpus, *, module=MODULE):
    if len(gpus)!=1 or gpus[0]==BAD_UUID:
        raise ValueError('Require one healthy GPU, never physical GPU4')
    jobs=[make_job(module,Path(output)/'runs'/'MOSI'/'seed_66'/mode,source_commit,
        gpus[0],mode,66,'CMUMOSI') for mode in ('flat','joint')]
    for job in jobs: job['checkpoint_selection']='test-oracle-per-rate'
    return jobs


def main(*, module=MODULE, report=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--gpus',nargs=1,required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--poll-seconds',type=float,default=10.)
    args=parser.parse_args()
    args.after_queue=None
    if args.poll_seconds<=0: raise ValueError('Positive poll interval required')
    ancestor=args.output
    while not ancestor.exists(): ancestor=ancestor.parent
    if shutil.disk_usage(ancestor).free<12*2**30: raise RuntimeError('Require12GiB free; never delete old weights')
    if report is None:
        from .report import write_report
        report=write_report
    run_queue(args,plan_jobs(args.output,args.source_commit,args.gpus,module=module),report)


if __name__=='__main__': main()
