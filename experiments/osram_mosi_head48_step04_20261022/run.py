"""One resource-checked foreground worker under persistent tmux, then summarize."""
import argparse
from pathlib import Path
import shutil

from experiments.osram_mosi_gcnet_router_3seed_20261019.run import devices
from experiments.osram_text_path_router_3seed_20261019.resource_gate import available_gib
from .worker import main as train,MODE


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-commit',required=True)
    p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args(argv)
    gpu=devices([args.gpu_uuid])[args.gpu_uuid]
    if (gpu['total_mib']-gpu['used_mib']<6144 or available_gib()<64
            or shutil.disk_usage(args.output.parent).free<12*2**30):
        raise RuntimeError('Require6GiB GPU,64GiB RAM,12GiB disk; do not change batch/protocol')
    if args.output.exists() and not args.resume:
        raise FileExistsError('Fresh independent single-job output required')
    command=['--mode',MODE,'--phase','train','--seed','66',
        '--gpu-uuid',args.gpu_uuid,'--source-commit',args.source_commit,'--output',str(args.output)]
    train(command+(['--resume'] if args.resume else []))
    from .report import summarize
    summarize(args.output)


if __name__=='__main__':
    main()
