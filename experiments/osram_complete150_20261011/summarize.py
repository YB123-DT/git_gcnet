"""Audit completed47 outputs and compute equal-fold/rate/seed metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics as st
import numpy as np


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def scores(path, classification):
    m=read(path/'metrics.json');rates={};hashes={}
    for i in range(8):
        r=str(i/10);p=path/f'predictions_miss_0p{i}.npz'
        rates[r]={'ACC':100*m['test'][r]['accuracy'],'WF1':100*m['test'][r]['weighted_f1']}
        if classification:
            with np.load(p) as x:y=x['labels'].reshape(-1);pred=x['predictions'].reshape(-1)
            c=4 if classification=='IEMOCAPFour' else 6
            assert all(np.sum(y==j)>0 for j in range(c))
            assert abs(np.mean(y==pred)-m['test'][r]['accuracy'])<1e-10
            rates[r]['UA']=100*float(np.mean([np.mean(pred[y==j]==j) for j in range(c)]))
        hashes[p.name]=sha(p)
    return dict(rates=rates,mean8={k:st.mean(v[k] for v in rates.values()) for k in rates['0.0']},
                high={k:st.mean(rates[str(i/10)][k] for i in (5,6,7)) for k in rates['0.0']},
                prediction_sha256=hashes,metrics_sha256=sha(path/'metrics.json'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    q=read(a.root/'QUEUE.json');assert q['status']=='complete'
    done=[r for r in q['tasks'] if r['status']=='complete'];assert len(done)==47
    assert sum(r['status']=='cancelled_by_user' for r in q['tasks'])==15
    rows=[]
    for row in done:
        root=a.root/'runs'/row['id'];pr=read(root/'PROVENANCE.json');task=pr['task'];path=root/'stage150'
        assert pr['status']=='complete' and pr['outputs_verified']
        assert len(read(path/'history.json'))==150
        traces=list((path/'learning_rates').glob('epoch_*.json'));assert len(traces)==50
        assert all(all(v==.0001 for v in read(f)['rates']) for f in traces)
        for n,h in pr['artifact_sha256'].items():
            assert (path/n).is_file()
            if not n.endswith('.pt'):assert sha(path/n)==h,n
        classification=task['dataset'] if task['dataset'].startswith('IEMOCAP') else None
        final=scores(path,classification)
        assert all(final['prediction_sha256'][n]==pr['artifact_sha256'][n] for n in final['prediction_sha256'])
        original=Path(task['original']) if task['original'] else root/'stage100'
        old=scores(original,classification)
        historical=scores(Path(task['reference']),classification)
        rows.append(dict(id=row['id'],dataset=task['dataset'],model=task['model'],seed=task['seed'],fold=task['fold'],
                         final=final,source100=old,historical_flat100=historical,source=str(path)))
    groups={}
    for dataset,model in sorted({(r['dataset'],r['model']) for r in rows}):
        group=[r for r in rows if (r['dataset'],r['model'])==(dataset,model)];seeds=[]
        for seed in sorted({r['seed'] for r in group}):
            folds=[r for r in group if r['seed']==seed];assert len(folds)==(5 if dataset.startswith('IEMOCAP') else 1)
            entry=dict(seed=seed)
            for kind in ('final','source100','historical_flat100'):
                entry[kind]={scope:{k:st.mean(f[kind][scope][k] for f in folds)
                                   for k in folds[0][kind][scope]} for scope in ('mean8','high')}
            seeds.append(entry)
        groups[dataset+'_'+model]=dict(seeds=seeds,
            mean={kind:{scope:{k:st.mean(s[kind][scope][k] for s in seeds)
                              for k in seeds[0][kind][scope]} for scope in ('mean8','high')}
                  for kind in ('final','source100','historical_flat100')})
    report=dict(label='INTERNAL DIAGNOSTIC ONLY',status='complete',completed=47,cancelled=15,
                protocol='per-rate BEST; mean over five folds then eight rates then seeds; MOSI only newly run67/68 here',
                verification='150histories,50LR traces,all required artifacts present,non-weight hashes checked; no new inference',
                queue=q,groups=groups,runs=rows)
    (a.root/'FINAL_SUMMARY.json').write_text(json.dumps(report,indent=2))
    for key,g in groups.items():
        print(key,json.dumps(g),flush=True)


if __name__=='__main__':main()
