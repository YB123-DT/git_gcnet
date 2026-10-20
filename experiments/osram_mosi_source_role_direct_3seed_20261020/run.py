"""Wait for the existing seed66 run, then add ONLY seeds67/68."""
import argparse
from pathlib import Path
import shutil

from experiments.osram_mosi_gcnet_router_3seed_20261019.run import BAD_UUID,make_job,run_queue

UPSTREAM=Path('/data2/yb/remote_experiments/osram_mosi_source_role_direct_20261020/execution_v1/QUEUE.json')


def plan_jobs(output,source_commit,gpus):
    if len(gpus)!=1 or gpus[0]==BAD_UUID:raise ValueError('Require one healthy GPU, not physical4')
    return [make_job('experiments.osram_mosi_source_role_direct_3seed_20261020.worker',
        Path(output)/'runs'/'CMUMOSI'/f'seed_{seed}'/'joint',source_commit,gpus[0],'joint',seed,'CMUMOSI')
        for seed in (67,68)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--gpus',nargs=1,required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--poll-seconds',type=float,default=10.)
    args=parser.parse_args();args.after_queue=UPSTREAM
    if args.poll_seconds<=0:raise ValueError('Positive poll interval required')
    ancestor=args.output
    while not ancestor.exists():ancestor=ancestor.parent
    if shutil.disk_usage(ancestor).free<16*2**30:raise RuntimeError('Require16GiB; preserve all existing checkpoints')
    from .report import write_report
    run_queue(args,plan_jobs(args.output,args.source_commit,args.gpus),write_report)


if __name__=='__main__':main()
