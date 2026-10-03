"""Offline Context Audit of frozen alpha0 versus alpha1; no model imports."""
import csv
import hashlib
import json
from pathlib import Path
import pickle
import statistics as st
from collections import defaultdict
import numpy as np
from sklearn.metrics import f1_score

ROOT=Path(__file__).resolve().parents[2]
EPS=1e-6


def classify(y,local,memory):
    gain=(y-local)**2-(y-memory)**2
    direction='help' if gain>EPS else ('harm' if gain < -EPS else 'near_zero')
    if y==0: transition='neutral_excluded'
    else:
        a=(local>0)==(y>0);b=(memory>0)==(y>0)
        transition='both_correct' if a and b else ('both_wrong' if not a and not b else ('wrong_to_correct' if b else 'correct_to_wrong'))
    return gain,direction,transition


def stats(rows):
    n=len(rows); result=dict(n=n,mean_G=st.mean(r['G'] for r in rows),
        median_G=st.median(r['G'] for r in rows))
    for key in ('help','harm','near_zero'):
        result[key]=sum(r['mse_effect']==key for r in rows)
        result[key+'_percent']=100*result[key]/n
    for key in ('wrong_to_correct','correct_to_wrong','both_correct','both_wrong','neutral_excluded'):
        result[key]=sum(r['polarity_transition']==key for r in rows)
    assert sum(result[k] for k in ('wrong_to_correct','correct_to_wrong','both_correct','both_wrong','neutral_excluded'))==n
    keep=[r for r in rows if r['label']!=0]
    result['binary_n']=len(keep)
    if keep:
        y=[r['label']>0 for r in keep]
        result['local_wf1']=100*f1_score(y,[r['local_prediction']>0 for r in keep],average='weighted',zero_division=0)
        result['memory_wf1']=100*f1_score(y,[r['memory_prediction']>0 for r in keep],average='weighted',zero_division=0)
    return result


def csvwrite(path,rows):
    with path.open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def main():
    source=ROOT/'experiments/osram_cfg84_history_scale_20260929/results'
    fine=ROOT/'experiments/osram_cfg84_history_scale_fine_20260929'
    out=Path(__file__).parent/'results';out.mkdir(exist_ok=False)
    metadata=ROOT/'dataset/CMUMOSI/CMUMOSI_features_raw_2way.pkl'
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(metadata)=='931ee50f3a77b2eb18786840a880a439f3ae6542bf2dfb5e14ecc9690702af84'
    with metadata.open('rb') as stream: ids,labels,*_=pickle.load(stream,encoding='latin1')
    uidmap={uid:(str(cid),t,float(labels[cid][t])) for cid,items in ids.items() for t,uid in enumerate(items)}
    with (fine/'prediction_analysis/reconstructed_sample_ids.csv').open() as stream:
        mappings=list(csv.DictReader(stream))
    logs=json.loads((source/'rows.json').read_text())
    logs={(r['seed'],r['rate'],r['alpha']):r for r in logs}
    hashes={};allrows=[];cells=[]
    for seed in (66,67,68):
        mapping=sorted((r for r in mappings if int(r['seed'])==seed),key=lambda r:int(r['artifact_row']))
        assert len(mapping)==686 and len({r['utterance_id'] for r in mapping})==686
        for rate in [i/10 for i in range(8)]:
            pair=[]
            for alpha in (0.,1.):
                p=source/f'seed_{seed}_miss_{rate:.1f}_alpha_{alpha}.npz';hashes[str(p.relative_to(ROOT))]=sha(p)
                with np.load(p) as data: pair.append({k:data[k].copy() for k in data.files})
            a,b=pair
            assert np.array_equal(a['labels'],b['labels']) and np.array_equal(a['availability'],b['availability'])
            # Link reconstructed IDs to the separately audited alpha1 artifacts.
            with np.load(fine/'results'/f'seed_{seed}_miss_{rate:.1f}_alpha_1.0.npz') as f:
                assert all(np.array_equal(b[k],f[k]) for k in ('predictions','labels','availability'))
            l0,l1=logs[seed,rate,0.],logs[seed,rate,1.]
            assert l0['upstream_sha256']==l1['upstream_sha256']
            assert l0['metrics']['mask_sha256']==l1['metrics']['mask_sha256']
            current=[]
            for i,m in enumerate(mapping):
                y=float(b['labels'][i]);p0=float(a['predictions'][i]);p1=float(b['predictions'][i])
                assert np.isfinite([y,p0,p1]).all() and y==float(m['label'])
                cid,t,label=uidmap[m['utterance_id']]
                assert abs(y-label)<1e-6
                gain,effect,transition=classify(y,p0,p1)
                row=dict(seed=seed,rate=rate,utterance_id=m['utterance_id'],conversation_id=cid,utterance_index=t,
                    pattern=''.join(x for x,v in zip('ATV',b['availability'][i]) if v),label=y,
                    local_prediction=p0,memory_prediction=p1,local_mse=(y-p0)**2,memory_mse=(y-p1)**2,
                    G=gain,mse_effect=effect,polarity_transition=transition,
                    position='first' if t==0 else ('2-5' if t<5 else '6+'),
                    local_margin='0-.25' if abs(p0)<=.25 else ('.25-1' if abs(p0)<=1 else '1+'),
                    label_strength='neutral' if y==0 else ('weak<=1' if abs(y)<=1 else 'strong>1'))
                current.append(row)
            s=stats(current)
            assert abs(s['local_wf1']-100*l0['metrics']['weighted_f1'])<1e-10
            assert abs(s['memory_wf1']-100*l1['metrics']['weighted_f1'])<1e-10
            for dimension in ('all','pattern','position','local_margin','label_strength'):
                groups=defaultdict(list)
                for row in current:groups['ALL' if dimension=='all' else row[dimension]].append(row)
                for group,items in groups.items():cells.append(dict(seed=seed,rate=rate,dimension=dimension,group=group,**stats(items)))
            allrows.extend(current)
    csvwrite(out/'utterances.csv',allrows);csvwrite(out/'by_seed_rate_group.csv',cells)
    aggregates=[]
    groups=defaultdict(list)
    for c in cells:groups[c['dimension'],c['group']].append(c)
    for (dim,group),items in groups.items():
        # Rate macro within seed, then seed macro; group-empty cells excluded explicitly.
        means={k:st.mean(st.mean(c[k] for c in items if c['seed']==s) for s in (66,67,68)
                         if any(c['seed']==s for c in items)) for k in ('mean_G','help_percent','harm_percent','near_zero_percent')}
        aggregates.append(dict(dimension=dim,group=group,cells=len(items),n=sum(c['n'] for c in items),**means,
            **{k:sum(c[k] for c in items) for k in ('wrong_to_correct','correct_to_wrong','both_correct','both_wrong','neutral_excluded')}))
    csvwrite(out/'aggregate.csv',aggregates)
    (out/'AUDIT.json').write_text(json.dumps(dict(training=False,inference=False,epsilon=EPS,source_hashes=hashes,
        sample_mapping_sha256=sha(fine/'prediction_analysis/reconstructed_sample_ids.csv'),metadata_sha256=sha(metadata)),indent=2)+'\n')
    assert all(sha(ROOT/p)==h for p,h in hashes.items())
    report=['# Context Audit — MOSI original cfg84 Flat','',
        'Offline reuse only:seed66/67/68 ×8rates, same checkpoint/mask/upstream per pair.',
        'Local-only means alpha0 at the original Flat historical inputs; Local adapter and skip remain.',
        'It is NOT a separately trained Local-only model. Memory calculations run unchanged in source inference.',
        'G=(y-local)^2-(y-memory)^2. MSE includes neutral labels; polarity excludes label0, threshold prediction>0.',
        'Near-zero:abs(G)<=1e-6. Mean G and percentages use rate-macro then seed-macro.',
        'Counts pool repeated seed/rate exposures, not independent utterances. Empty group/rate cells excluded.',
        'Internal Test-oracle label-assisted diagnosis; no learned selector, new fitting, or new inference.','',
        '|Group|N exposures|Mean G|MSE help %|MSE harm %|Near-zero %|Local wrong→Memory right|Local right→Memory wrong|Both right|Both wrong|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in aggregates:
        report.append(f"|{r['dimension']}:{r['group']}|{r['n']}|{r['mean_G']:.5f}|{r['help_percent']:.2f}|{r['harm_percent']:.2f}|{r['near_zero_percent']:.2f}|{r['wrong_to_correct']}|{r['correct_to_wrong']}|{r['both_correct']}|{r['both_wrong']}|")
    report+=['','Patterns describe CURRENT availability under random missing, not persistent whole-conversation masks.',
        'Local-margin groups are descriptive and not calibrated uncertainty. Label-strength groups require labels.',
        'MSE improvement does not imply polarity correction, nor does MSE worsening imply wrong classification.',
        'These observations identify strata, not semantic utterance types or a deployable rule for memory usage.','']
    (out/'RESULT.md').write_text('\n'.join(report));print('\n'.join(report))


if __name__=='__main__':main()
