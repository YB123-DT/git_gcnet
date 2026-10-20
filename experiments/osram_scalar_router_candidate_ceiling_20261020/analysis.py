"""Read saved scalar predictions only; no Torch, model loading or inference."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

RATES=tuple(f'{i/10:.1f}' for i in range(8))
TAGS=('full','candidate_B','candidate_G')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_predictions(a):
    y=a['labels'];n=len(y);c=a['candidates'];p=a['prediction'];w=a['weights'];e=a['eligible']
    if y.shape!=(n,) or c.shape!=(n,3,1) or p.shape!=(n,1) or w.shape!=(n,3) or e.shape!=(n,) or e.dtype!=np.bool_:
        raise ValueError('Wrong saved scalar array shapes/types')
    if not all(np.isfinite(a[k]).all() for k in ('labels','candidates','prediction','weights','availability')):
        raise ValueError('Nonfinite saved predictions')
    availability=a['availability']
    if availability.shape!=(n,3) or not np.isin(availability,[0,1]).all() or not availability.any(1).all():
        raise ValueError('Invalid availability')
    expected=np.zeros(n,dtype=bool)
    ids=a['conversation_ids'];indices=a['utterance_indices']
    if ids.shape!=(n,) or indices.shape!=(n,) or len(set(zip(ids.tolist(),indices.tolist())))!=n:
        raise ValueError('Invalid or duplicate utterance identity')
    for cid in np.unique(ids):
        rows=np.flatnonzero(ids==cid);rows=rows[np.argsort(indices[rows])]
        if not np.array_equal(indices[rows],np.arange(len(rows))):
            raise ValueError('Expected complete ordered conversation, not filtered rows')
        seen=False
        for i in rows:
            expected[i]=seen and not bool(availability[i,1])
            seen=seen or bool(availability[i,1])
    if not np.array_equal(expected,e):raise ValueError('Strict-past Router eligibility mismatch')
    if (not np.array_equal(c[~e,1:],np.repeat(c[~e,:1],2,axis=1))
            or not np.array_equal(p[~e],c[~e,0])
            or not np.array_equal(w[~e],np.tile([1,0,0],((~e).sum(),1)))):
        raise ValueError('Nonidentical inactive Router candidates')
    if (w<0).any() or not np.allclose(w.sum(1),1,atol=1e-6,rtol=0):
        raise ValueError('Invalid convex mixture weights')
    mixed=(c*w[...,None]).sum(1)
    if not np.allclose(p,mixed,atol=2e-6,rtol=1e-6):raise ValueError('Saved mixture reconstruction differs')
    lo=c[...,0].min(1);hi=c[...,0].max(1);pred=p[:,0]
    violation=float(np.maximum(np.maximum(lo-pred,pred-hi),0).max(initial=0))
    if violation>2e-6:raise ValueError('Outside candidate convex hull')
    unanimous=(c[...,0]>0).all(1)|(~(c[...,0]>0)).all(1)
    if (unanimous & ((pred>0)!=(c[:,0,0]>0))).any():
        raise ValueError('Polarity flip outside candidate support')
    return dict(strict_past_eligibility_exact=True,inactive_candidates_exact=True,
        mixture_max_absolute_error=float(np.max(np.abs(p-mixed),initial=0)),
        convexity_max_violation=violation)


def metrics(y,p):
    nz=y!=0
    return dict(ACC=100*accuracy_score(y[nz]>0,p[nz]>0) if nz.any() else None,
        WF1=100*f1_score(y[nz]>0,p[nz]>0,average='weighted',zero_division=0) if nz.any() else None,
        MSE=float(np.mean((p-y)**2)) if len(y) else None)


def analyze_cohort(labels,candidates,prediction,mask):
    y=np.asarray(labels,dtype=np.float64)[mask];c=np.asarray(candidates,dtype=np.float64)[mask]
    p=np.asarray(prediction,dtype=np.float64).reshape(-1)[mask];n=len(y)
    nz=y!=0;truth=y>0;good=(c>0)==truth[:,None]
    full_good=good[:,0];router_good=(p>0)==truth
    unanimous=(c>0).all(1)|(~(c>0)).all(1)
    correctable=nz & ~full_good & good.any(1)
    corrections=nz & ~full_good & router_good;harms=nz & full_good & ~router_good
    oracle=c[:,0].copy()
    if correctable.any():
        losses=np.where(good,(c-y[:,None])**2,np.inf)
        best=losses.argmin(1)
        oracle[correctable]=c[np.flatnonzero(correctable),best[correctable]]
    row=dict(N_regression=n,N_polarity=int(nz.sum()),unanimous=int((nz&unanimous).sum()),
        cross_threshold=int((nz&~unanimous).sum()),full_errors=int((nz&~full_good).sum()),
        candidate_correctable=int(correctable.sum()),corrections=int(corrections.sum()),harms=int(harms.sum()),
        missed_corrections=int((correctable&~router_good).sum()),
        candidate_exact_zero=int(((c==0).any(1)&nz).sum()))
    row['unanimous_pct']=100*row['unanimous']/row['N_polarity'] if row['N_polarity'] else None
    row['correctable_error_pct']=100*row['candidate_correctable']/row['full_errors'] if row['full_errors'] else None
    row['captured_correctable_pct']=100*row['corrections']/row['candidate_correctable'] if row['candidate_correctable'] else None
    for tag,v in dict(zip(TAGS,c.T),router=p,oracle=oracle).items():
        row.update({tag+'_'+key:value for key,value in metrics(y,v).items()})
    row['delta_MSE']=row['router_MSE']-row['full_MSE'] if n else None
    row['discrete_mse_floor']=float(np.mean(((c-y[:,None])**2).min(1))) if n else None
    row['convex_mse_floor']=float(np.mean((y-np.clip(y,c.min(1),c.max(1)))**2)) if n else None
    row['candidate_span_mean']=float(np.mean(np.ptp(c,axis=1))) if n else None
    row['candidate_span_p95']=float(np.percentile(np.ptp(c,axis=1),95)) if n else None
    groups={}
    for name,selected in dict(corrections=corrections,harms=harms,both_correct=nz&full_good&router_good,
        both_wrong=nz&~full_good&~router_good,unanimous=nz&unanimous,
        cross_threshold=nz&~unanimous,neutral=~nz).items():
        old=float(np.mean((c[selected,0]-y[selected])**2)) if selected.any() else None
        new=float(np.mean((p[selected]-y[selected])**2)) if selected.any() else None
        groups[name]=dict(N=int(selected.sum()),full_MSE=old,router_MSE=new,
                         delta_MSE=new-old if selected.any() else None)
    return row,groups


def load_case(label,path):
    path=Path(path);s=json.loads((path/'SUMMARY.json').read_text());cfg=s['config']
    if (s.get('status')!='complete' or s.get('mode')!='joint'
            or s['dataset'] not in ('CMUMOSI','CMUMOSEI')
            or cfg['missing_mask_protocol']!='gcnet-random-mask-v1'
            or cfg['checkpoint_selection']!='test-oracle-per-rate'
            or cfg['checkpoint_selection_metric']!='weighted_f1'):
        raise ValueError('Require completed current GCNet Joint Test-best scalar run')
    ledger=json.loads((path/'best_by_rate.json').read_text())
    if ledger!=s['best_by_rate'] or set(ledger)!=set(RATES):raise ValueError('Selection ledger differs')
    records=[];rows=[];group_rows=[];provenance=[];checks=[];first=None
    for rate in RATES:
        p=path/'evaluation'/('random_'+rate.replace('.','p')+'.npz')
        digest=sha(p);selection=ledger[rate];saved=s['evaluation'][rate]
        if digest!=selection['prediction_sha256'] or digest!=saved['output_sha256']:
            raise ValueError('Selected prediction SHA mismatch: '+str(p))
        if selection['epoch']!=s['selected_epoch_by_rate'][rate] or not selection['same_evaluation_call']:
            raise ValueError('Selected epoch/provenance differs')
        with np.load(p,allow_pickle=False) as f:a={k:f[k] for k in f.files}
        check=validate_predictions(a);checks.append(dict(rate=rate,**check))
        current={k:a[k] for k in ('labels','conversation_ids','utterance_indices','speakers')}
        if first is None:first=current
        else:
            for k in first:np.testing.assert_array_equal(current[k],first[k])
        if hashlib.sha256(a['availability'].tobytes()).hexdigest()!=saved['mask_sha256']:
            raise ValueError('Mask hash differs from selected evaluation')
        # Match original float32 MSE reduction as well as nonzero/strict>0 polarity.
        predictions=dict(full=a['candidates'][:,0,0],candidate_B=a['candidates'][:,1,0],
                         candidate_G=a['candidates'][:,2,0],router=a['prediction'][:,0])
        for tag,pred in predictions.items():
            actual=metrics(a['labels'],pred)
            for k,v in actual.items():
                if not np.isclose(v,saved[tag][k],atol=1e-8,rtol=0):
                    raise ValueError('Saved metric mismatch: '+tag+'/'+k)
        records.append(a)
        for cohort,mask in dict(all=np.ones(len(a['labels']),dtype=bool),eligible=a['eligible']).items():
            row,groups=analyze_cohort(a['labels'],a['candidates'][...,0],a['prediction'],mask)
            rows.append(dict(case=label,dataset=s['dataset'],seed=s['seed'],rate=rate,
                selected_epoch=selection['epoch'],cohort=cohort,**row))
            group_rows.extend(dict(case=label,rate=rate,cohort=cohort,group=name,**v) for name,v in groups.items())
        provenance.append(dict(rate=rate,selected_epoch=selection['epoch'],prediction_file=str(p),
            prediction_sha256=digest,checkpoint=selection['checkpoint'],
            checkpoint_sha256=selection['checkpoint_sha256'],mask_sha256=saved['mask_sha256']))
    aggregates={}
    for scope,indices in dict(mean=list(range(8)),high=[5,6,7]).items():
        combined={k:np.concatenate([records[i][k] for i in indices]) for k in records[0]}
        aggregates[scope]={}
        for cohort,mask in dict(all=np.ones(len(combined['labels']),dtype=bool),eligible=combined['eligible']).items():
            pooled,groups=analyze_cohort(combined['labels'],combined['candidates'][...,0],combined['prediction'],mask)
            selected=[r for r in rows if r['cohort']==cohort and r['rate'] in [RATES[i] for i in indices]]
            means={k:float(np.mean([r[k] for r in selected if r[k] is not None]))
                for k in pooled if k.endswith(('_ACC','_WF1','_MSE')) and any(r[k] is not None for r in selected)}
            aggregates[scope][cohort]=dict(occurrence_statistics=pooled,mean_per_rate_metrics=means,
                pooled_MSE_groups=groups,metric_rate_count=sum(r['N_polarity']>0 for r in selected))
    return dict(case=label,dataset=s['dataset'],seed=s['seed'],epochs=s['epoch'],backbone_lr=cfg['learning_rate'],
        source_commit=s['source_commit'],summary_sha256=sha(path/'SUMMARY.json'),
        ledger_sha256=sha(path/'best_by_rate.json'),config=cfg,artifacts=provenance,checks=checks,
        aggregate=aggregates),rows,group_rows


def write_csv(path,rows):
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',action='append',required=True,help='LABEL=read-only saved run directory')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();cases=[];rows=[];groups=[]
    labels=[value.split('=',1)[0] for value in args.case]
    if len(set(labels))!=len(labels):raise ValueError('Duplicate case label')
    if args.output.exists():raise FileExistsError('Fresh independent analysis output required')
    for value in args.case:
        label,path=value.split('=',1);s,r,g=load_case(label,path);cases.append(s);rows.extend(r);groups.extend(g)
    args.output.mkdir(parents=True)
    result=dict(label='INTERNAL POST-HOC ORACLE DIAGNOSTIC ONLY',new_training=False,new_inference=False,
        analysis_git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        analysis_file_sha256=sha(__file__),numpy=np.__version__,cases=cases)
    (args.output/'SUMMARY.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    write_csv(args.output/'PER_RATE.csv',rows);write_csv(args.output/'MSE_GROUPS.csv',groups)
    lines=['# Scalar Router candidate capacity audit','','INTERNAL POST-HOC ORACLE DIAGNOSTIC ONLY','',
        'No training, model loading, inference, mask generation, or weight edits. Saved selected NPZs only.',
        'Full means the full readout of the SAME Joint checkpoint, not separately-trained Flat.',
        'Candidates: Full; remove Gap-T←T (candidate_B); remove Base←T (candidate_G).',
        'Eligible: current Text absent and strictly-past Text observed. Outside eligible all candidates are identical.',
        'Polarity excludes gold0 and uses prediction>0, exactly matching original evaluation; MSE includes gold0.',
        'Rates retain their originally selected checkpoints. Eight-rate metrics are arithmetic means, not a deployment checkpoint.',
        'Counts summed over rates are repeated utterance-rate occurrences, not independent unique utterances.',
        'Unanimity is reported WITHIN eligible rows. Neutral labels and ineligible copies do not inflate it.','',
        '## Eight-rate candidate support and actual routing (eligible nonzero labels)','','| Case | N | Same polarity | Correctable / Full errors | Router corrections / harms | Missed corrections |',
        '|---|---:|---:|---:|---:|---:|']
    for s in cases:
        a=s['aggregate']['mean']['eligible']['occurrence_statistics']
        lines.append(f"| {s['case']} | {a['N_polarity']} | {a['unanimous_pct']:.3f}% | {a['candidate_correctable']} / {a['full_errors']} | {a['corrections']} / {a['harms']} | {a['missed_corrections']} |")
    lines+=['','## Whole-cohort W-F1, original arithmetic rate averaging','',
        '| Case | Scope | Joint Full | Actual Router | Correctable-only oracle | Oracle−Full pp |',
        '|---|---|---:|---:|---:|---:|']
    for s in cases:
        for scope in ('mean','high'):
            a=s['aggregate'][scope]['all']['mean_per_rate_metrics']
            lines.append(f"| {s['case']} | {scope} | {a['full_WF1']:.4f} | {a['router_WF1']:.4f} | {a['oracle_WF1']:.4f} | {a['oracle_WF1']-a['full_WF1']:+.4f} |")
    lines+=['','Oracle preserves Full-correct predictions and only repairs Full errors for which a correct candidate exists.',
        'Among correct candidates choose minimum squared error. This uses labels AFTER evaluation, not a trained/deployable Router.',
        'It gives attainable discrete-candidate classification accuracy with oracle selection, not a proven global maximum of W-F1.','',
        '## MSE on eligible rows, pooled across rates (gold0 included)','',
        '| Case | Full | Router | Router−Full | Discrete candidate floor | Convex hull floor |',
        '|---|---:|---:|---:|---:|---:|']
    for s in cases:
        a=s['aggregate']['mean']['eligible']['occurrence_statistics']
        lines.append(f"| {s['case']} | {a['full_MSE']:.6f} | {a['router_MSE']:.6f} | {a['delta_MSE']:+.6f} | {a['discrete_mse_floor']:.6f} | {a['convex_mse_floor']:.6f} |")
    lines+=['','## Paired MSE changes by actual polarity outcome (eligible nonzero labels)','',
        '| Case | Outcome | N | Full MSE | Router MSE | Router−Full |',
        '|---|---|---:|---:|---:|---:|']
    for s in cases:
        for name in ('corrections','harms','both_correct','both_wrong'):
            a=s['aggregate']['mean']['eligible']['pooled_MSE_groups'][name]
            values=' | '.join('N/A' if a[k] is None else f'{a[k]:.6f}' for k in ('full_MSE','router_MSE','delta_MSE'))
            lines.append(f"| {s['case']} | {name} | {a['N']} | {values} |")
    lines+=['','Discrete floor chooses the closest candidate by gold label. Convex floor projects the label onto [min candidate,max candidate].',
        'Both are post-hoc regression bounds, not deployed scores. Convex hull endpoints may require limiting one-hot weights.',
        'A same-polarity convex mixture cannot change polarity, but may still improve or harm MSE.',
        'Do not infer that Memory is useless or that training LR is optimal from this conditional readout audit.',
        'Per-rate all/eligible counts, oracle metrics and candidate metrics: PER_RATE.csv.',
        'MSE differences for corrections, harms, both-correct, both-wrong and unanimity: MSE_GROUPS.csv.',
        'All prediction/checkpoint/mask provenance and numerical checks: SUMMARY.json.']
    (args.output/'RESULT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':main()
