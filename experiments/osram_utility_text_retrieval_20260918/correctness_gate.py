"""Formal cache equivalence gate (seed66 rate0.5). Fail -> SystemExit(1)."""
import random, sys
from pathlib import Path
import numpy as np, torch
from sklearn.metrics import f1_score
from gcnet_missing_m3 import train_gcnet as tr
from experiments.osram_utility_text_retrieval_20260918.utility_common import build_reader

ROOT=Path('/data2/yb/remote_experiments/osram_utility_text_retrieval_20260918')
CACHE=ROOT/'utility_cache_v2'/'seed_66'/'rate_0p5'/'test.pt'
PAM_E=Path('/data2/yb/remote_experiments/osram_pam_episodic_text_20260917')
SEED=66; RATE=0.5

def fail(msg):
    print('GATE_FAIL',msg); raise SystemExit(1)

def main():
    device=torch.device('cuda')
    cp=torch.load(CACHE,map_location='cpu',weights_only=False)['tensors']
    model,cfg,dims,shape,loaders=build_reader(SEED,'0.5',device)
    loader=loaders[2][cfg.fold-1]; schedule=tr._build_schedule(cfg,'test',RATE)
    batches=[]
    for bi,raw in enumerate(loader):
        data=tr._move_batch(raw,device); view=tr._prepare_view(data,schedule,0,dims); batches.append((bi,data,view))
    # A: candidate-0 normal forward
    for bi,data,view in batches:
        with torch.no_grad():
            logits,_,_,_=model(view['incomplete'],view['availability'],view['qmask'],view['umask'],view['lengths'])
        scores=logits[...,0]
        rows=torch.where(cp['batch_index']==bi)[0]
        for row in rows.tolist():
            t=int(cp['query_t'][row]); b=int(cp['query_b'][row])
            diff=abs(float(scores[t,b])-float(cp['scores'][row,0]))
            if diff>=1e-5: fail(f'A row={row} diff={diff}')
    print('GATE_A_PASS')
    # B: 32 historical candidates direct full forward
    eligible=[(r,k) for r in range(len(cp['labels'])) for k in range(1,int(cp['mask'][r].sum()))]
    random.seed(42); random.shuffle(eligible)
    for row,k in eligible[:32]:
        bi=int(cp['batch_index'][row]); t=int(cp['query_t'][row]); b=int(cp['query_b'][row])
        data=dict((bi,data) for bi,data,_ in batches)[bi]
        view=tr._prepare_view(data,schedule,0,dims)
        avail=view['availability']; umask=view['umask']; qmask=view['qmask']; lengths=view['lengths']
        tiled_incomplete=view['incomplete'][:,b:b+1]
        tiled_avail=avail[:,b:b+1]
        tiled_qmask=qmask[b].unsqueeze(0)
        tiled_umask=umask[b].unsqueeze(0)
        override=torch.zeros((avail.shape[0],1,model.latent_dim),device=device)
        mask=torch.zeros((avail.shape[0],1),dtype=torch.bool,device=device)
        override[t,0]=cp['values'][row,k].to(device); mask[t,0]=True
        with torch.no_grad():
            logits,_,_,_=model(tiled_incomplete,tiled_avail,tiled_qmask,tiled_umask,[lengths[b]],pam_text_override=override,pam_override_mask=mask)
        direct=float(logits[t,0,0]); cached=float(cp['scores'][row,k])
        if abs(direct-cached)>=1e-5: fail(f'B row={row} k={k} direct={direct} cached={cached}')
    print('GATE_B_PASS')
    # C: candidate0-only full test W-F1 equals original PAM-E metric.
    labels=cp['full_labels'].numpy(); scores=cp['full_scores'].numpy()
    nz=labels!=0
    cache_wf1=f1_score(labels[nz]>0,scores[nz]>0,average='weighted')
    pam_wf1=__import__('json').load(open(PAM_E/'mosi'/f'seed_{SEED}'/'metrics.json'))['test']['0.5']['weighted_f1']
    if abs(float(cache_wf1)-float(pam_wf1))>=1e-6:
        fail(f'C cache={cache_wf1} pam={pam_wf1}')
    print('GATE_C_PASS')
    print('GATE_PASS')
if __name__=='__main__': main()
