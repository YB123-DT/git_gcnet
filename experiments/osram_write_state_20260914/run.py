"""Five approved WSC-OSRAM MOSI seeds, inherited configs, no baseline retraining."""
import argparse
from dataclasses import replace
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.osram_causal_nojepa_20260910 import run as base

ROOT=Path('/data2/yb/remote_experiments/osram_write_state_20260914')
reference_configuration=base.configuration


def configuration(seed):
    cfg,source,_=reference_configuration(seed)
    cfg=replace(cfg,training_objective='write-state')
    delta={'training_objective':['joint','write-state'],
           'checkpoint_selection':['test-oracle','test-oracle-per-rate']}
    return cfg,source,delta


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--launch',action='store_true')
    g.add_argument('--seed',type=int,choices=base.runner.SEEDS)
    args=p.parse_args()
    base.ROOT,base.configuration=ROOT,configuration
    if args.launch:
        base.runner.ROOT,base.runner.configuration,base.runner.__file__=ROOT,configuration,__file__
        base.runner.launch(gpus=(5,5,5,6,6))
    else:
        base.train(args.seed)
