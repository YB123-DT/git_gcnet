"""Fixed-sum B/Gap interpolation; readout-only, frozen checkpoints."""
import argparse
from pathlib import Path
import sys
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.osram_cross_substitution_20260910.run import run

GRID={'t_minus1':-1.,'t_minus_half':-.5,'t_zero':0.,'t_half':.5,'normal':1.}


def interpolated_inputs(local,base,gap,availability,valid):
    missing=~availability.bool()
    eligible=valid & (missing.sum(-1)==1)
    active=gap*missing[...,None]
    single=active.sum(-2)
    u=(base+single)/2
    d=(base-single)/2
    out={}
    for mode,t in GRID.items():
        b,g=base.clone(),active.clone()
        if t!=1:
            # Exact original tensors at endpoints avoid cancellation artifacts.
            bt,gt=(single,base) if t==-1 else (u+t*d,u-t*d)
            torch.testing.assert_close((bt+gt)[eligible],(base+single)[eligible],rtol=1e-5,atol=1e-6)
            b[eligible]=bt[eligible]
            g[eligible]=(gt.unsqueeze(-2)*missing[...,None])[eligible]
        out[mode]=torch.cat([local,b,g.flatten(-2)],-1)
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    for seed in range(66,71):
        run(seed,args.output,input_fn=interpolated_inputs,modes=tuple(GRID))
