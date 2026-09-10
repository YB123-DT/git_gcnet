"""Frozen readout directional audit; fixed eight control draws, two FD step sizes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn

REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO))
CACHE=Path('/data2/yb/remote_experiments/osram_magnitude_20260910')
KINDS=('real','shuffle','random','symmetric')


def directions(d,seed):
    n,dim=d.shape
    assert n>=2 and dim%2==0 and torch.count_nonzero(d[:,dim//2:])==0
    length=d.norm(dim=-1,keepdim=True);assert torch.all(length>1e-12)
    q=d/length
    gen=torch.Generator(device='cpu').manual_seed(seed)
    order=torch.randperm(n,generator=gen)
    donor=torch.empty(n,dtype=torch.long);donor[order]=order.roll(1)
    shuffled=q[donor.to(d.device)]
    r=torch.zeros_like(d)
    r[:,:dim//2]=torch.randn(n,dim//2,generator=gen,dtype=d.dtype).to(d.device)
    r=r/r.norm(dim=-1,keepdim=True)
    # Rescale each control to ||D_i|| before joint normalization; scale cancels.
    scale=2**.5
    return {'real':(-q/scale,q/scale),'shuffle':(-shuffled/scale,shuffled/scale),
            'random':(-r/scale,r/scale),'symmetric':(r/scale,r/scale)},donor


def finite_difference(f,b,g,pair,eps):
    with torch.no_grad():
        return ((f(b+eps*pair[0],g+eps*pair[1])-f(b-eps*pair[0],g-eps*pair[1]))/(2*eps)).abs()


class FrozenReadout(nn.Module):
    def __init__(self,cp):
        super().__init__();w=cp['model']
        out,inp=w['osram.emotion_adapter.1.weight'].shape
        latent=w['osram.local_skip.weight'].shape[1]
        self.adapter=nn.Sequential(nn.LayerNorm(inp),nn.Linear(inp,out),nn.GELU(),nn.Dropout(0),nn.Linear(out,out))
        self.local=nn.Linear(latent,out);self.norm=nn.LayerNorm(out);self.head=nn.Linear(out,1)
        for module,prefix in ((self.adapter,'osram.emotion_adapter.'),(self.local,'osram.local_skip.'),
                              (self.norm,'osram.emotion_norm.'),(self.head,'smax_fc.')):
            module.load_state_dict({k[len(prefix):]:v for k,v in w.items() if k.startswith(prefix)},strict=True)
        self.eval().requires_grad_(False)

    def forward(self,local,b,g,target):
        slots=[g if i==target else torch.zeros_like(g) for i in range(3)]
        x=torch.cat([local,b,*slots],-1)
        return self.head(self.norm(self.local(local)+self.adapter(x))).squeeze(-1)


def collect_local(root):
    from experiments.osram_cross_substitution_20260910.run import run
    root.mkdir(parents=True,exist_ok=False);(root/'locals').mkdir()
    def normal(local,b,g,a,valid):
        return {'normal':torch.cat([local,b,(g*(1-a)[...,None]).flatten(-2)],-1)}
    for seed in range(66,71):
        buffers={}
        def capture(s,rate,ctx,view,valid):
            buffers.setdefault(rate,[]).append(tuple(x.detach().cpu().numpy().copy() for x in
                (ctx['local'][valid],ctx['base'][valid],
                 (ctx['gap']*(1-view['availability'])[...,None]).sum(-2)[valid])))
        run(seed,root,input_fn=normal,modes=('normal',),context_observer=capture)
        for rate,items in buffers.items():
            local,b,g=[np.concatenate([x[i] for x in items]) for i in range(3)]
            with np.load(CACHE/'contexts'/f'seed{seed}_rate{rate:.1f}.npz') as old:
                assert np.array_equal(b,old['base']) and np.array_equal(g,old['gap'])
            np.savez_compressed(root/'locals'/f'seed{seed}_rate{rate:.1f}.npz',local=local)


def write_csv(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        w.writeheader();w.writerows(rows)


def analyze(root):
    from experiments.osram_causal_readout_20260910.run import FULL
    torch.set_num_threads(4);rows=[];groups=[];provenance=[]
    for seed in range(66,71):
        checkpoint=FULL/f'seed_{seed}'/'best.pt';cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model=FrozenReadout(cp).to(device='cuda',dtype=torch.float64)
        assert cp['config']['readout_type']=='shared'
        provenance.append(dict(seed=seed,checkpoint=str(checkpoint),sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()))
        for ri in range(1,8):
            name=f'seed{seed}_rate{ri/10:.1f}.npz'
            with np.load(CACHE/'contexts'/name) as cache,np.load(root/'locals'/name) as lc,np.load(root/name) as pred:
                b,g,y,a=(cache[k] for k in ('base','gap','labels','availability'))
                assert np.array_equal(y,pred['labels']) and np.array_equal(a,pred['availability'])
                for pattern,target in [('AT',2),('AV',1),('TV',0)]:
                    eligible=(a.sum(-1)==2)&(a[:,target]==0)&(y!=0)
                    norm=np.linalg.norm((b-g)/2,axis=-1)
                    ids=np.flatnonzero(eligible&(norm>1e-12))
                    record=dict(seed=seed,rate=ri/10,pattern=pattern,n=int(len(ids)),
                        zero_D=int((eligible&(norm<=1e-12)).sum()),total_eligible=int(eligible.sum()))
                    if len(ids)<2:
                        groups.append(record);continue
                    tensor=lambda x:torch.as_tensor(x,device='cuda',dtype=torch.float64)
                    bt=tensor(b[ids]).requires_grad_();gt=tensor(g[ids]).requires_grad_();local=tensor(lc['local'][ids])
                    f=lambda bb,gg:model(local,bb,gg,target)
                    score=f(bt,gt)
                    replay_error=float(np.max(abs(score.detach().cpu().numpy()-pred['normal'][ids])))
                    assert replay_error<1e-4, replay_error
                    gb,gg=torch.autograd.grad(score.sum(),(bt,gt))
                    d=(bt.detach()-gt.detach())/2
                    sensitivities={k:[] for k in KINDS};fds={};fd_errors=[];donors=None
                    for draw in range(8):
                        rng=int.from_bytes(hashlib.sha256(f'{seed}:{ri}:{pattern}:{draw}:directional-v1'.encode()).digest()[:8],'big')%(2**63-1)
                        direct,donor=directions(d,rng)
                        if draw==0:donors=ids[donor.numpy()]
                        for kind,pair in direct.items():
                            sensitivity=(gb*pair[0]+gg*pair[1]).sum(-1).abs().detach()
                            sensitivities[kind].append(sensitivity.cpu().numpy())
                            if draw==0:
                                for eps in (1e-3,5e-4):
                                    fd=finite_difference(f,bt.detach(),gt.detach(),pair,eps).cpu().numpy()
                                    fds[f'{kind}_fd_{eps}']=fd
                                    error=abs(fd-sensitivity.cpu().numpy());fd_errors.extend(error.tolist())
                    assert max(fd_errors)<5e-6,max(fd_errors)
                    values={k:np.stack(v).mean(0) for k,v in sensitivities.items()}
                    record.update(replay_fp64_max_abs_error=replay_error,fd_max_abs_error=max(fd_errors),
                        fd_mean_abs_error=float(np.mean(fd_errors)))
                    for k,v in values.items():
                        record[k+'_mean']=float(v.mean());record[k+'_median']=float(np.median(v))
                        record[k+'_p90']=float(np.quantile(v,.9))
                        if k!='real':
                            record['real_less_'+k]=float((values['real']<v).mean())
                    for j,i in enumerate(ids):
                        row=dict(seed=seed,rate=ri/10,pattern=pattern,sample_index=int(i),D_norm=float(norm[i]),
                            abs_score=float(abs(score[j].detach())),first_shuffle_donor_index=int(donors[j]),
                            **{k:float(v[j]) for k,v in values.items()},**{k:float(v[j]) for k,v in fds.items()})
                        for k in ('shuffle','random','symmetric'):
                            row[k+'_draw_sd']=float(np.std(np.stack(sensitivities[k])[:,j],ddof=1))
                        rows.append(row)
                    groups.append(record)
                    assert all(p.grad is None for p in model.parameters())
                    print(f'seed={seed} rate={ri/10} {pattern} n={len(ids)} real={record["real_mean"]:.5g} shuffle={record["shuffle_mean"]:.5g} random={record["random_mean"]:.5g} sym={record["symmetric_mean"]:.5g} FDerr={max(fd_errors):.2g}',flush=True)
    write_csv(root/'per_sample.csv',rows);write_csv(root/'per_seed_rate_pattern.csv',groups)
    summary=[]
    for pattern in ('ALL','AT','AV','TV'):
        for seed in range(66,71):
            # Each rate averages eligible samples, not equally weighting differently sized patterns.
            perrate=[]
            for ri in range(1,8):
                g=[r for r in groups if r['seed']==seed and r['rate']==ri/10 and (pattern=='ALL' or r['pattern']==pattern) and r['n']>=2]
                n=sum(r['n'] for r in g)
                perrate.append({k:sum(r[k+'_mean']*r['n'] for r in g)/n for k in KINDS})
            summary.append(dict(pattern=pattern,seed=seed,**{k:float(np.mean([r[k] for r in perrate])) for k in KINDS}))
    write_csv(root/'per_seed_summary.csv',summary)
    (root/'PROVENANCE.json').write_text(json.dumps(dict(checkpoints=provenance,draws=8,
        fd_epsilons=[1e-3,5e-4],precision='FP64 frozen readout; gradients only to Base/Gap',
        scope='nonzero-label exactly-one-missing; exclude zero D; shuffle same seed/rate/pattern with no self donor',
        normalization='joint unit norm of two slots; control pre-unit magnitude matched to sqrt(2)*norm(D_i)',
        random_space='forward256 only; symmetric and antisymmetric share each Gaussian draw',
        finite_difference='draw0 for all four direction types at both epsilons',
        selection='historical fixed Test-oracle checkpoint; no reselection'),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('collect','analyze'));p.add_argument('root',type=Path)
    args=p.parse_args();collect_local(args.root) if args.mode=='collect' else analyze(args.root)
