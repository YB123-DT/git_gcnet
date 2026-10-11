"""Four bounded extra capacity runs, two per healthy physical GPU0/1."""
import argparse
import sys
from experiments.osram_direct_input4_20261010 import dispatch

LANES={
    0: ('GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45',
        ('nested_direct_random_mlp128','nested_direct_random_dim128_mlp256')),
    1: ('GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049',
        ('nested_direct_random_dim128_mlp512','nested_direct_random_dim256_mlp512')),
}

if __name__=='__main__':
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--lane',type=int,choices=tuple(LANES),required=True)
    args,remaining=p.parse_known_args()
    dispatch.GPU_INDEX=args.lane
    dispatch.GPU,dispatch.METHODS=LANES[args.lane]
    sys.argv=[sys.argv[0],*remaining]
    dispatch.main()
