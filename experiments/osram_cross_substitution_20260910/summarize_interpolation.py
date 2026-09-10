"""Score-level odd/even decomposition; aggregate only after pairing samples."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

GRID={'t_minus1':-1.,'t_minus_half':-.5,'t_zero':0.,'t_half':.5,'normal':1.}


def write_csv(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(rows)


def analyze(root):
    samples=[];curve=[];components=[];hashes={}
    for seed in range(66,71):
        meta=json.loads((root/f'seed{seed}.json').read_text())
        for ri in range(8):
            rate=ri/10;path=root/f'seed{seed}_rate{rate:.1f}.npz'
            hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path) as z:
                a=z['availability'];y=z['labels'];ok=(a.sum(-1)==2)&(y!=0)
                f={k:z[k].astype('float64') for k in GRID}
                assert all(np.isfinite(x).all() for x in f.values())
                odd1=(f['normal']-f['t_minus1'])/2
                even1=(f['normal']+f['t_minus1'])/2-f['t_zero']
                oddhalf=(f['t_half']-f['t_minus_half'])/2
                evenhalf=(f['t_half']+f['t_minus_half'])/2-f['t_zero']
                for i in np.flatnonzero(ok):
                    samples.append(dict(seed=seed,rate=rate,sample_index=int(i),
                        pattern=''.join(m for m,v in zip('ATV',a[i]) if v),label=float(y[i]),
                        **{k:float(v[i]) for k,v in f.items()},odd1=float(odd1[i]),even1=float(even1[i]),
                        oddhalf=float(oddhalf[i]),evenhalf=float(evenhalf[i])))
                for pattern,select in [('ALL',ok),('AT',ok&(a[:,2]==0)),('AV',ok&(a[:,1]==0)),('TV',ok&(a[:,0]==0))]:
                    n=int(select.sum())
                    for mode,t in GRID.items():
                        r=dict(seed=seed,rate=rate,pattern=pattern,t=t,n=n)
                        if n:
                            delta=(f[mode]-f['normal'])[select]
                            metric=next(r['metrics'] for r in meta['rows'] if r['rate']==rate and r['group']==('one_missing' if pattern=='ALL' else pattern) and r['mode']==mode)
                            r.update(weighted_f1=100*metric['weighted_f1'],
                                score_delta_mean=float(delta.mean()),score_mae=float(abs(delta).mean()),
                                score_p90_abs=float(np.quantile(abs(delta),.9)),
                                sign_flip_percent=100*float(((f[mode][select]>0)!=(f['normal'][select]>0)).mean()))
                        curve.append(r)
                    for name,values in [('odd1',odd1),('even1',even1),('oddhalf',oddhalf),('evenhalf',evenhalf)]:
                        r=dict(seed=seed,rate=rate,pattern=pattern,component=name,n=n)
                        if n:
                            x=values[select]
                            r.update(mean=float(x.mean()),mean_abs=float(abs(x).mean()),
                                     p90_abs=float(np.quantile(abs(x),.9)))
                        components.append(r)
    write_csv(root/'per_sample_components.csv',samples)
    write_csv(root/'per_seed_rate_curve.csv',curve)
    write_csv(root/'per_seed_rate_components.csv',components)
    grouped=[];component_grouped=[]
    for pattern in ('ALL','AT','AV','TV'):
        for t in GRID.values():
            perseed=[]
            for seed in range(66,71):
                g=[r for r in curve if r['seed']==seed and r['pattern']==pattern and r['t']==t and r['n']]
                perseed.append({k:float(np.mean([r[k] for r in g])) for k in ('weighted_f1','score_delta_mean','score_mae','score_p90_abs','sign_flip_percent')})
            grouped.append(dict(pattern=pattern,t=t,**{k:float(np.mean([r[k] for r in perseed])) for k in perseed[0]},
                weighted_f1_seed_sd=float(np.std([r['weighted_f1'] for r in perseed],ddof=1))))
        for name in ('odd1','even1','oddhalf','evenhalf'):
            g=[r for r in components if r['pattern']==pattern and r['component']==name and r['n']]
            component_grouped.append(dict(pattern=pattern,component=name,
                **{k:float(np.mean([r[k] for r in g])) for k in ('mean','mean_abs','p90_abs')}))
    write_csv(root/'curve_summary.csv',grouped);write_csv(root/'component_summary.csv',component_grouped)
    (root/'analysis_metadata.json').write_text(json.dumps(dict(source_sha256=hashes,
        outcome='raw regression score; not probabilities or class logits',
        sample_filter='exactly one missing and label != 0',
        aggregation='equal rates .1-.7 within seed then equal seeds; quantiles averaged by group, not pooled',
        repeated_samples='same utterance may recur across rates/seeds; no independence assumption'),indent=2)+'\n')
    lines=['# Fixed-sum Base/Gap interpolation','',
        '**INTERNAL DIAGNOSTIC ONLY.** Frozen MOSI Flat causal eta=.6, seeds66–70.',
        'No training or new epoch selection. Existing historical eight-rate-mean Test-selected best.pt checkpoints.',
        'Local/memory unchanged; only the Base and unique active Gap emotion slots are redistributed.',
        'U=(B+G)/2, D=(B-G)/2; B(t)=U+tD, G(t)=U-tD. Endpoints copy original tensors exactly.',
        'Evidence sum conserved within FP32 tolerance (1e-6 absolute / 1e-5 relative).',
        'Scores are raw regression predictions; sign threshold zero; nonzero-label AT/AV/TV samples only.',
        'All means: equal rates .1–.7 within seed, then equal seeds. Rate0 has no eligible samples.',
        'SD is across five within-seed rate means. Subset W-F1 is not the full-test benchmark score.']
    for pattern in ('ALL','AT','AV','TV'):
        lines+=['',f'## {pattern}','',
            '| t | W-F1 mean ± SD (%) | Score MAE vs t=1 | P90 abs deviation (group-averaged) | Sign flips vs t=1 |',
            '|---|---:|---:|---:|---:|']
        for r in grouped:
            if r['pattern']==pattern:
                lines.append(f"| {r['t']} | {r['weighted_f1']:.3f} ± {r['weighted_f1_seed_sd']:.3f} | {r['score_mae']:.5f} | {r['score_p90_abs']:.5f} | {r['sign_flip_percent']:.3f}% |")
        lines+=['','| Component | Signed mean | Mean abs | P90 abs (group-averaged) |','|---|---:|---:|---:|']
        for r in component_grouped:
            if r['pattern']==pattern:
                lines.append(f"| {r['component']} | {r['mean']:+.6f} | {r['mean_abs']:.6f} | {r['p90_abs']:.6f} |")
    lines+=['','O(t)=(f(t)-f(-t))/2; E(t)=(f(t)+f(-t))/2-f(0). Applied per sample BEFORE aggregation.',
        'Odd/even here are output sensitivities along the chosen redistribution path, not proven internal modules.',
        'Constant W-F1 does not imply unchanged scores or individual predictions. Endpoint parity alone does not establish exchange symmetry.',
        'Small even sensitivity does not identify why a separately retrained gated architecture failed; bottleneck, projections and optimization remain unseparated explanations.',
        'No prespecified equivalence threshold or population significance claim. Inspect pattern/rate/seed tables and distributions.']
    (root/'RESULT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);analyze(p.parse_args().root)
