"""Build utility cache for all rates and splits for one seed."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import torch
from gcnet_missing_m3 import train_gcnet as tr
from experiments.osram_utility_text_retrieval_20260918.utility_common import build_reader, query_utility_from_batch

ROOT = Path('/data2/yb/remote_experiments/osram_utility_text_retrieval_20260918')
RATES = [round(i/10,1) for i in range(8)]

def rate_tag(rate): return f'{rate:.1f}'.replace('.','p')

def process_split(model,cfg,dims,loader,split,rate,device):
    schedule=tr._build_schedule(cfg,split,rate)
    pieces=[]
    for raw in loader:
        data=tr._move_batch(raw,device); view=tr._prepare_view(data,schedule,0,dims)
        with torch.no_grad(): out=query_utility_from_batch(model,cfg,view)
        if out is not None: pieces.append(out)
    if not pieces: return None
    max_c=max(int(p['values'].shape[1]) for p in pieces)
    padded=[]
    for p in pieces:
        item={}
        for k,v in p.items():
            if k in {'values','mask','scores','losses','utility'} and v.shape[1]<max_c:
                pad_shape=(v.shape[0],max_c-v.shape[1])+tuple(v.shape[2:])
                pad=torch.zeros(pad_shape,dtype=v.dtype)
                if k=='mask': pad=pad.to(dtype=torch.bool)
                v=torch.cat((v,pad),dim=1)
            item[k]=v
        padded.append(item)
    return {k:torch.cat([p[k] for p in padded],dim=0) for k in padded[0]}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--device',default='cuda'); args=ap.parse_args()
    device=torch.device(args.device)
    for rate in RATES:
        model,cfg,dims,shape,loaders=build_reader(args.seed,f'{rate:.1f}',device)
        for split,loader in (('train',loaders[0][cfg.fold-1]),('validation',loaders[1][cfg.fold-1]),('test',loaders[2][cfg.fold-1])):
            out=process_split(model,cfg,dims,loader,split,rate,device)
            if out is None: continue
            path=ROOT/'utility_cache'/f'seed_{args.seed}'/f'rate_{rate_tag(rate)}'/f'{split}.pt'
            path.parent.mkdir(parents=True,exist_ok=True)
            torch.save({'tensors':out,'seed':args.seed,'rate':rate,'split':split,'n_queries':int(out['labels'].shape[0]),'max_candidates':int(out['values'].shape[1])},path)
            print(json.dumps({'seed':args.seed,'rate':rate,'split':split,'n':int(out['labels'].shape[0])}),flush=True)
        del model
    print('DONE',args.seed,flush=True)
if __name__=='__main__': main()
