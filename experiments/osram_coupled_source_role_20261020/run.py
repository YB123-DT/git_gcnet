"""One finite seed66 source-role job per dataset; existing Flat reused."""
import argparse
from pathlib import Path
import shutil

from experiments.osram_mosi_gcnet_router_3seed_20261019.run import BAD_UUID,make_job,run_queue


def plan_jobs(output,source_commit,gpus,dataset):
    if len(gpus)!=1 or gpus[0]==BAD_UUID:raise ValueError('Require one healthy GPU, never physical4')
    suffix={'CMUMOSI':'mosi','CMUMOSEI':'mosei'}[dataset]
    job=make_job('experiments.osram_coupled_source_role_20261020.'+suffix,
        Path(output)/'runs'/dataset/'seed_66'/'joint',source_commit,gpus[0],'joint',66,dataset)
    job['checkpoint_selection']='test-oracle-per-rate'
    return [job]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=('CMUMOSI','CMUMOSEI'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--gpus',nargs=1,required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--poll-seconds',type=float,default=10.)
    args=parser.parse_args();args.after_queue=None
    if args.poll_seconds<=0:raise ValueError('Positive poll interval required')
    ancestor=args.output
    while not ancestor.exists():ancestor=ancestor.parent
    if shutil.disk_usage(ancestor).free<12*2**30:raise RuntimeError('Require12GiB disk; never delete weights')
    from .report import write_report
    run_queue(args,plan_jobs(args.output,args.source_commit,args.gpus,args.dataset),write_report)


if __name__=='__main__':main()
