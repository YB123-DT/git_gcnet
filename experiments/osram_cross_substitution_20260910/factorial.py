"""Per-sample 2x2 decomposition of saved signed regression decision margins."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

MAPPING={'bb':'no_gap','bg':'move_base_to_gap','gb':'move_gap_to_base','gg':'no_base'}


def effects(bb,bg,gb,gg):
    return dict(grand=(bb+bg+gb+gg)/4,
                content=((bb+bg)-(gb+gg))/2,
                slot=((bg+gg)-(bb+gb))/2,
                interaction=(bg-bb)-(gg-gb))


def summarize(rows):
    if not rows:
        return dict(n=0)
    get=lambda k:np.array([r[k] for r in rows])
    result=dict(n=len(rows))
    for key in ('content','slot','interaction'):
        x=get(key)
        result.update({key+'_mean':float(x.mean()),key+'_mean_abs':float(abs(x).mean()),
            key+'_median_abs':float(np.median(abs(x))),key+'_p90_abs':float(np.quantile(abs(x),.9)),
            key+'_positive_fraction':float((x>0).mean())})
    i,c,s=get('interaction'),get('content'),get('slot')
    denom=np.mean(i*i+4*c*c+4*s*s)
    result['interaction_factor_energy_fraction']=float(np.mean(i*i)/denom) if denom else 0.
    result['additive_margin_mae']=float(abs(i).mean()/4)
    result['additive_sign_disagreement_fraction']=float(get('additive_sign_disagreement_fraction').mean())
    return result


def save_csv(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def analyze(root,output):
    rows=[];sources={}
    for seed in range(66,71):
        for ri in range(8):
            path=root/f'seed{seed}_rate{ri/10:.1f}.npz'
            sources[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path) as d:
                labels=d['labels'];availability=d['availability']
                selected=(availability.sum(-1)==2)&(labels!=0)
                indices=np.flatnonzero(selected)
                sign=np.sign(labels[selected])
                cells={k:sign*d[v][selected] for k,v in MAPPING.items()}
                assert all(np.isfinite(v).all() for v in cells.values())
                out=effects(**cells)
                disagreements=[]
                for name,c,s in (('bb',1,-1),('bg',1,1),('gb',-1,-1),('gg',-1,1)):
                    additive=out['grand']+c*out['content']/2+s*out['slot']/2
                    reconstructed=additive+c*s*out['interaction']/4
                    np.testing.assert_allclose(reconstructed,cells[name],rtol=1e-5,atol=1e-6)
                    disagreements.append((additive>0)!=(cells[name]>0))
                fraction=np.stack(disagreements).mean(0)
                for j,index in enumerate(indices):
                    pattern=''.join(m for m,a in zip('ATV',availability[index]) if a)
                    rows.append(dict(seed=seed,rate=ri/10,sample_index=int(index),pattern=pattern,
                        label=float(labels[index]),**{k:float(v[j]) for k,v in cells.items()},
                        **{k:float(v[j]) for k,v in out.items()},
                        additive_sign_disagreement_fraction=float(fraction[j])))
    output.mkdir(parents=True,exist_ok=False)
    save_csv(output/'per_sample.csv',rows)
    groups=[]
    for seed in range(66,71):
        for ri in range(8):
            for pattern in ('ALL','AT','AV','TV'):
                part=[r for r in rows if r['seed']==seed and r['rate']==ri/10 and (pattern=='ALL' or r['pattern']==pattern)]
                groups.append(dict(seed=seed,rate=ri/10,pattern=pattern,**summarize(part)))
    save_csv(output/'per_seed_rate_pattern.csv',groups)
    rate_summary=[]
    for ri in range(1,8):
        g=[x for x in groups if x['rate']==ri/10 and x['pattern']=='ALL']
        assert all(x['n'] for x in g)
        rate_summary.append(dict(rate=ri/10,**{k:float(np.mean([x[k] for x in g])) for k in g[0] if k not in ('seed','rate','pattern','n')},n=sum(x['n'] for x in g)))
    save_csv(output/'per_rate.csv',rate_summary)
    seed_summary=[]
    for seed in range(66,71):
        g=[x for x in groups if x['seed']==seed and x['pattern']=='ALL' and x['n']]
        seed_summary.append(dict(seed=seed,**{k:float(np.mean([x[k] for x in g])) for k in g[0] if k not in ('seed','rate','pattern','n')},n=sum(x['n'] for x in g)))
    save_csv(output/'per_seed.csv',seed_summary)
    overall={k:float(np.mean([x[k] for x in seed_summary])) for k in seed_summary[0] if k not in ('seed','n')}
    valid_groups=[g for g in groups if g['pattern']=='ALL' and g['n']]
    metadata=dict(outcome='sign(nonzero ground-truth sentiment) * raw regression score; uncalibrated decision margin',
        mapping=MAPPING,source_sha256=sources,sample_rate_seed_records=len(rows),
        unique_sample_warning='The same test utterance recurs across rates/seeds. Rows are not independent subjects.',
        overall_equal_seed_equal_rate=overall,
        positive_interaction_groups=sum(g['interaction_mean']>0 for g in valid_groups),
        valid_seed_rate_groups=len(valid_groups))
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    lines=['# Saved-output 2×2 factorial decomposition','',
        '**INTERNAL DIAGNOSTIC ONLY. No model execution or training.**',
        'MOSI five frozen Flat checkpoints; exactly-one-missing AT/AV/TV and nonzero labels.',
        'Y = sign(label) × raw sentiment regression score. Higher is better. This is a signed decision margin, not calibrated class-logit margin or NLL.', '',
        'Cells: BB=Base-only; BG=Move B→Gap; GB=Move G→Base; GG=Gap-only. Local is unchanged.',
        'Content effect C=((BB+BG)−(GB+GG))/2; positive favors Base content.',
        'Slot effect S=((BG+GG)−(BB+GB))/2; positive favors Gap slot.',
        'Interaction I=(BG−BB)−(GG−GB).',
        'Exact reconstruction: Y(c,s)=grand+c*C/2+s*S/2+c*s*I/4, with c=+1 for B and s=+1 for Gap slot.',
        'Thus the sample-wise residual of the best additive 2×2 representation is ±I/4.', '',
        'Means below weight rates .1–.7 equally within each seed and then seeds equally. Rate0 has no eligible samples.', '',
        '| Effect | Signed mean | Mean absolute | Median absolute (group-averaged) | P90 absolute (group-averaged) |',
        '|---|---:|---:|---:|---:|']
    for key in ('content','slot','interaction'):
        lines.append(f"| {key} | {overall[key+'_mean']:+.6f} | {overall[key+'_mean_abs']:.6f} | {overall[key+'_median_abs']:.6f} | {overall[key+'_p90_abs']:.6f} |")
    lines+=['','| Rate | C mean | S mean | I mean | Mean abs I | Additive sign disagreement |','|---|---:|---:|---:|---:|---:|']
    for x in rate_summary:
        lines.append(f"| {x['rate']} | {x['content_mean']:+.5f} | {x['slot_mean']:+.5f} | {x['interaction_mean']:+.5f} | {x['interaction_mean_abs']:.5f} | {100*x['additive_sign_disagreement_fraction']:.2f}% |")
    lines+=['','| Seed | I mean | Mean abs I |','|---|---:|---:|']
    for x in seed_summary:
        lines.append(f"| {x['seed']} | {x['interaction_mean']:+.5f} | {x['interaction_mean_abs']:.5f} |")
    lines+=['',f"Interaction means positive in {metadata['positive_interaction_groups']}/{len(valid_groups)} seed/rate groups.",
        f"Additive approximation margin MAE: {overall['additive_margin_mae']:.6f}.",
        f"Interaction share of within-four-cell factorial energy (group-averaged): {100*overall['interaction_factor_energy_fraction']:.2f}%.",
        f"Ignoring interaction changes decision sign in {100*overall['additive_sign_disagreement_fraction']:.2f}% of four-cell cases (group-averaged). This is not an F1 delta.", '',
        'Energy uses E[I²]/E[4C²+4S²+I²]; it is descriptive, not population explained variance.',
        'No equivalence threshold was prespecified. Small signed mean alone cannot establish negligible interaction.',
        'Regression-score scale, frozen checkpoint selection and repeated samples limit inference; no significance/independence claims are made.',
        'See per_sample.csv for full paired effects, per_seed_rate_pattern.csv for AT/AV/TV and all rates, and metadata.json for all input hashes.']
    (output/'RESULT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args();analyze(args.root,args.output)
