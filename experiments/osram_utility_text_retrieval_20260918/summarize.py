"""Summarize utility-retrieval results and combine with frozen PAM-E outputs."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score

ROOT=Path('/data2/yb/remote_experiments/osram_utility_text_retrieval_20260918')
PAM_E=Path('/data2/yb/remote_experiments/osram_pam_episodic_text_20260917')
RATES=[f'{i/10:.1f}' for i in range(8)]
SEEDS=(66,67,68,69,70)
MODES=('Original','Uniform','Soft','Top1','Top2','Oracle-Best','Oracle-Soft')
METHODS=('U1','U2','U3')
PATTERNS=('A','T','V','AT','AV','TV','ATV')
PATTERN_IDS={1:'V',2:'T',3:'TV',4:'A',5:'AV',6:'AT',7:'ATV'}
HIGH={'0.5','0.6','0.7'}
T_MISSING={'A','V','AV'}; T_PRESENT={'T','AT','TV','ATV'}

def weighted_f1(labels,pred):
    labels=np.asarray(labels); pred=np.asarray(pred)
    nonzero=labels!=0
    labels=labels[nonzero]>0; pred=pred[nonzero]>0
    if labels.size==0: return 0.0
    return float(f1_score(labels,pred,average='weighted'))

def tag(rate): return rate.replace('.','p')

def load_npz(path):
    d=np.load(path); return {k:d[k] for k in d.files}

def combine(method,mode,seed,rate):
    pam=load_npz(PAM_E/'mosi'/f'seed_{seed}'/f'predictions_miss_{tag(rate)}.npz')
    avail=pam['availability']; mask=(avail[:,1]==0)&((avail[:,0]>0)|(avail[:,2]>0))
    pred=pam['predictions'].copy()
    artifact=ROOT/'retrieval_artifacts'/method/f'seed_{seed}'/f'rate_{tag(rate)}.npz'
    if mask.sum()==0 or not artifact.exists():
        return pam['labels'],pred,avail
    if mode != 'Original':
        art=load_npz(artifact)
        if mask.sum()!=len(art['labels']):
            raise ValueError(f'mask/cache mismatch {method} {seed} {rate}: {mask.sum()} vs {len(art["labels"])}')
        pred[mask]=art[f'pred_{mode}']
    return pam['labels'],pred,avail

def aggregate_method(mode):
    rows=[]; pat_rows=[]; retrieval=[]
    for method in METHODS:
        for seed in SEEDS:
            for rate in RATES:
                try:
                    labels,pred,avail=combine(method,mode,seed,rate)
                except FileNotFoundError:
                    continue
                rows.append({'method':method,'mode':mode,'seed':seed,'rate':rate,'weighted_f1':weighted_f1(labels,pred)*100})
                pids=avail[:,0].astype(int)*4+avail[:,1].astype(int)*2+avail[:,2].astype(int)
                for pid,name in PATTERN_IDS.items():
                    sel=pids==pid
                    pat_rows.append({'method':method,'mode':mode,'seed':seed,'rate':rate,'pattern':name,'n':int(sel.sum()),'weighted_f1':weighted_f1(labels[sel],pred[sel])*100 if sel.any() else None})
    return rows,pat_rows

def retrieval_metrics():
    rows=[]
    for method in METHODS:
        for seed in SEEDS:
            for rate in RATES:
                p=ROOT/'retrieval_artifacts'/method/f'seed_{seed}'/f'rate_{tag(rate)}.npz'
                if not p.exists(): continue
                a=load_npz(p); r=a['r']; mask=a['mask']; losses=a['losses']; scores=a['scores']
                # gold better = smaller loss
                gold=losses.copy(); gold[~mask]=1e9
                probs=np.exp(r-r.max(1,keepdims=True)); probs=np.where(mask,probs,0); probs=probs/probs.sum(1,keepdims=True).clip(min=1e-12)
                top1=probs.argmax(1)
                oracle=gold.argmin(1)
                hit=(top1==oracle).mean()
                # NDCG@3 with relevance = positive utility
                util=scores[:, :1]-scores
                util=np.where(mask,np.maximum(util,0),0)
                def ndcg(k):
                    vals=[]
                    for i in range(len(probs)):
                        order=np.argsort(-probs[i])[:k]
                        dcg=sum((2**util[i,j]-1)/math.log2(pos+2) for pos,j in enumerate(order))
                        ideal_order=np.argsort(-util[i])[:k]
                        idcg=sum((2**util[i,j]-1)/math.log2(pos+2) for pos,j in enumerate(ideal_order))
                        vals.append(dcg/idcg if idcg>0 else 0.0)
                    return float(np.mean(vals))
                rows.append({'method':method,'seed':seed,'rate':rate,'candidate_count_mean':float(mask.sum(1).mean()),'bank_nonempty_coverage':float((mask.sum(1)>1).mean()),'ndcg@1':ndcg(1),'ndcg@3':ndcg(3),'top1_utility':float(util[np.arange(len(top1)),top1].mean()),'top1_oracle_hit':float(hit),'candidate0_prob_mean':float(probs[:,0].mean()),'candidate0_select_rate':float((top1==0).mean())})
    return rows

def agg(rows):
    out={}
    for method in METHODS:
        for rate in RATES:
            vals=[r['weighted_f1'] for r in rows if r['method']==method and r['rate']==rate]
            if vals: out.setdefault(method,{})[rate]={'mean':float(np.mean(vals)),'std':float(np.std(vals,ddof=1)) if len(vals)>1 else 0.0}
    for method in out:
        allv=[out[method][r]['mean'] for r in RATES if r in out[method]]
        highv=[out[method][r]['mean'] for r in HIGH if r in out[method]]
        out[method]['macro8']=float(np.mean(allv)); out[method]['high']=float(np.mean(highv))
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    summaries={}
    for mode in MODES:
        rows,pat=aggregate_method(mode); summaries[mode]={'rows':rows,'pattern':pat}
    ret=retrieval_metrics()
    with open(args.output/'per_seed_rate.csv','w') as f:
        f.write('method,mode,seed,rate,weighted_f1\n')
        for mode in MODES:
            for r in summaries[mode]['rows']: f.write(f"{r['method']},{mode},{r['seed']},{r['rate']},{r['weighted_f1']}\n")
    with open(args.output/'pattern_per_seed.csv','w') as f:
        f.write('method,mode,seed,rate,pattern,n,weighted_f1\n')
        for mode in MODES:
            for r in summaries[mode]['pattern']: f.write(f"{r['method']},{mode},{r['seed']},{r['rate']},{r['pattern']},{r['n']},{r['weighted_f1']}\n")
    with open(args.output/'retrieval_metrics.csv','w') as f:
        f.write('method,seed,rate,candidate_count_mean,bank_nonempty_coverage,ndcg@1,ndcg@3,top1_utility,top1_oracle_hit,candidate0_prob_mean,candidate0_select_rate\n')
        for r in ret: f.write(','.join(str(r[k]) for k in r)+'\n')
    overall={'per_mode':{mode:agg(summaries[mode]['rows']) for mode in MODES},'retrieval':{}}
    for method in METHODS:
        sub=[r for r in ret if r['method']==method]
        if sub:
            overall['retrieval'][method]={k:float(np.mean([r[k] for r in sub])) for k in sub[0] if k not in ('method','seed','rate')}
    (args.output/'overall_summary.json').write_text(json.dumps(overall,indent=2,default=float)+'\n')
    print(json.dumps(overall,indent=2,default=float))
if __name__=='__main__': main()
