"""Frozen representation magnitude audit; no Jacobian, training, or geometry model."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))


def measure(base,gap):
    base=np.asarray(base,dtype=np.float64);gap=np.asarray(gap,dtype=np.float64)
    assert base.shape==gap.shape and base.ndim==2
    norm=lambda x:np.linalg.norm(x,axis=-1)
    bn,gn=norm(base),norm(gap)
    def ratio(num,den):
        return np.divide(num,den,out=np.full_like(num,np.nan),where=den>1e-12)
    def cosine(a,b):
        return np.clip(ratio((a*b).sum(-1),norm(a)*norm(b)),-1,1)
    centered_base=base-base.mean(0);centered_gap=gap-gap.mean(0)
    return dict(base_norm=bn,gap_norm=gn,difference_norm=norm(base-gap),sum_norm=norm(base+gap),
        cosine=cosine(base,gap),relative_difference=ratio(norm(base-gap),bn+gn),
        gap_base_norm_ratio=ratio(gn,bn),
        centered_base_norm=norm(centered_base),centered_gap_norm=norm(centered_gap),
        centered_cosine=cosine(centered_base,centered_gap))


def collect(root):
    import torch
    from experiments.osram_cross_substitution_20260910.run import run
    root.mkdir(parents=True,exist_ok=False)
    raw=root/'contexts';raw.mkdir()
    def normal_only(local,base,gap,a,valid):
        return {'normal':torch.cat([local,base,(gap*(1-a)[...,None]).flatten(-2)],-1)}
    for seed in range(66,71):
        buffers={}
        def capture(s,rate,ctx,view,valid):
            # Copy detached context; never feed diagnostics into model state/output.
            bucket=buffers.setdefault(rate,[])
            b=ctx['base'][valid].detach().cpu().numpy().copy()
            g=(ctx['gap']*(1-view['availability'])[...,None]).sum(-2)[valid].detach().cpu().numpy().copy()
            bucket.append((b,g))
        run(seed,root,input_fn=normal_only,modes=('normal',),context_observer=capture)
        for rate,chunks in buffers.items():
            d=np.load(root/f'seed{seed}_rate{rate:.1f}.npz')
            np.savez_compressed(raw/f'seed{seed}_rate{rate:.1f}.npz',
                base=np.concatenate([x[0] for x in chunks]),gap=np.concatenate([x[1] for x in chunks]),
                labels=d['labels'],availability=d['availability'])


def csv_write(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        w.writeheader();w.writerows(rows)


def analyze(root):
    groups=[];samples=[];hashes={}
    for seed in range(66,71):
        for ri in range(8):
            p=root/'contexts'/f'seed{seed}_rate{ri/10:.1f}.npz'
            hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
            with np.load(p) as d:
                a,y,b,g=(d[k] for k in ('availability','labels','base','gap'))
                assert b.shape==g.shape and len(y)==len(b) and np.isfinite(b).all() and np.isfinite(g).all()
                eligible=(a.sum(-1)==2)&(y!=0)
                for pattern,target in [('AT',2),('AV',1),('TV',0)]:
                    idx=np.flatnonzero(eligible&(a[:,target]==0));n=len(idx)
                    row=dict(seed=seed,rate=ri/10,pattern=pattern,n=n)
                    if n:
                        v=measure(b[idx],g[idx])
                        row.update(base_zero_fraction=float((v['base_norm']<=1e-12).mean()),
                                   gap_zero_fraction=float((v['gap_norm']<=1e-12).mean()))
                        for key,values in v.items():
                            finite=values[np.isfinite(values)]
                            row[key+'_valid_count']=len(finite)
                            row[key+'_mean']=float(finite.mean()) if len(finite) else None
                            row[key+'_median']=float(np.median(finite)) if len(finite) else None
                            row[key+'_p90']=float(np.quantile(finite,.9)) if len(finite) else None
                        for j,index in enumerate(idx):
                            samples.append(dict(seed=seed,rate=ri/10,pattern=pattern,sample_index=int(index),
                                **{key:float(values[j]) if np.isfinite(values[j]) else None for key,values in v.items()}))
                    groups.append(row)
    csv_write(root/'per_sample_magnitude.csv',samples);csv_write(root/'per_seed_rate_pattern.csv',groups)
    summary=[]
    for pattern in ('AT','AV','TV'):
        part=[r for r in groups if r['pattern']==pattern and r['n']]
        stats={}
        for key in part[0]:
            if key.endswith('_mean') or key.endswith('_fraction'):
                perseed=[]
                for seed in range(66,71):
                    values=[r[key] for r in part if r['seed']==seed and r.get(key) is not None]
                    if values:perseed.append(float(np.mean(values)))
                stats[key]=float(np.mean(perseed)) if perseed else None
                stats[key+'_seed_sd']=float(np.std(perseed,ddof=1)) if len(perseed)>1 else None
        summary.append(dict(pattern=pattern,**stats))
    csv_write(root/'summary.csv',summary)
    (root/'audit_metadata.json').write_text(json.dumps(dict(source_context_sha256=hashes,
        scope='same nonzero-label exactly-one-missing subset as prior interpolation',
        centered_similarity='subtract separate Base/Gap mean vectors within seed x rate x pattern; paired cosine',
        zero_policy='norm product <=1e-12 => undefined cosine; omit from cosine average, retain counts',
        aggregation='equal eligible rates within seed, then equal seeds; do not pool modalities or seeds before centering',
        precision='FP32 captured tensors; magnitude/cosine computations FP64',
        causal_half='512-d context includes zero backward half; retained as consumed by flat readout'),indent=2)+'\n')
    lines=['# Base/Gap representation magnitude audit','',
        '**INTERNAL DIAGNOSTIC ONLY.** Same five frozen MOSI Flat causal eta=.6 checkpoints.',
        'AT/AV/TV, exactly one missing and nonzero labels; rates .1–.7. Rate0 has no eligible samples.',
        'One normal eval forward captures Base and the unique active Gap before emotion fusion; no training or memory/readout alteration.',
        'Means weight rates equally within seed then seeds equally. Context has 512 dimensions, including the unchanged zero backward half.', '',
        '| Pattern | norm B | norm G | norm(B-G) | norm(B+G) | cos(B,G) | centered cosine | relative difference |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    keys=('base_norm','gap_norm','difference_norm','sum_norm','cosine','centered_cosine','relative_difference')
    for r in summary:
        lines.append('| '+r['pattern']+' | '+' | '.join('N/A' if r[k+'_mean'] is None else f"{r[k+'_mean']:.6f}" for k in keys)+' |')
    lines+=['','Relative difference = ||B-G||/(||B||+||G||), computed per sample before averaging.',
        'Centered cosine: independently subtract B and G mean vectors WITHIN seed/rate/pattern, then paired cosine; not CKA and not pooled centering.',
        'Zero-norm cosine is undefined, never filled with 0 or 1; counts and zero fractions are explicitly retained.', '',
        '| Pattern | Base zero fraction | Gap zero fraction |', '|---|---:|---:|']
    for r in summary:
        lines.append(f"| {r['pattern']} | {r['base_zero_fraction']:.6f} | {r['gap_zero_fraction']:.6f} |")
    lines+=['', 'Per-sample values, group mean/median/P90, valid counts and across-seed SD are saved in CSV.',
        'Checkpoint selection remains the historical eight-rate-mean Test oracle. No reselection, Jacobian, random-direction control or training.',
        'Norm/cosine evidence alone does not establish whether a difference encodes semantic information or why the downstream classifier is insensitive.']
    (root/'RESULT.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('collect','analyze'));p.add_argument('root',type=Path)
    args=p.parse_args();collect(args.root) if args.mode=='collect' else analyze(args.root)
