"""Offline label-assisted diagnostics. Never loads weights or runs inference."""
import csv
import hashlib
import json
from pathlib import Path
import pickle
import statistics
import sys

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gcnet_modality_jepa.protocol import EpochSeededSubsetSampler, SeedBundle
from gcnet_modality_jepa.splits import build_official_split

PATTERNS = {'A': (1,0,0), 'T': (0,1,0), 'V': (0,0,1),
            'AT': (1,1,0), 'AV': (1,0,1), 'TV': (0,1,1), 'ATV': (1,1,1)}
ALPHAS = (.8,.85,.9,.95,1.,1.05,1.1,1.15,1.2)


def wf1(y, pred):
    return 100 * float(f1_score(y, pred, average='weighted', zero_division=0)) if len(y) else None


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_save(path, rows):
    with path.open('x', newline='') as out:
        writer=csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    exp=Path(__file__).parent; source=exp/'results'; out=exp/'prediction_analysis'
    out.mkdir(exist_ok=False)
    files=sorted(source.glob('*.npz'))
    assert len(files)==216
    before={p.name:sha(p) for p in files}
    labels_path=ROOT/'dataset/CMUMOSI/CMUMOSI_features_raw_2way.pkl'
    assert sha(labels_path)=='931ee50f3a77b2eb18786840a880a439f3ae6542bf2dfb5e14ecc9690702af84'
    with labels_path.open('rb') as f:
        ids,labels,_,_,train,val,test=pickle.load(f,encoding='latin1')
    vids=sorted(train)+sorted(val)+sorted(test)
    split=build_official_split(vids,train,val,test)
    recorded=json.loads((source/'rows.json').read_text())
    recorded={(r['seed'],r['rate'],r['alpha']):r for r in recorded}
    transitions=[]; cells=[]; samples=[]; mapping=[]
    for seed in (66,67,68):
        order=list(EpochSeededSubsetSampler(split.test,SeedBundle(seed).derive('data_order:CMUMOSI:fold:1:test')))
        uids=[u for j in order for u in ids[vids[j]]]
        target=np.array([y for j in order for y in labels[vids[j]]],dtype=np.float32)
        assert len(uids)==len(set(uids))==686
        keep=target!=0; y=target[keep]>0; sample_ids=np.array(uids)[keep]
        for i,uid in enumerate(uids):
            mapping.append(dict(seed=seed,artifact_row=i,utterance_id=uid,label=float(target[i]),included=bool(keep[i])))
        for rate in [i/10 for i in range(8)]:
            arrays=[]; available=None
            for a in ALPHAS:
                with np.load(source/f'seed_{seed}_miss_{rate:.1f}_alpha_{a}.npz') as f:
                    assert np.array_equal(f['labels'],target)
                    assert np.isfinite(f['predictions']).all()
                    if available is None: available=f['availability'].copy()
                    assert np.array_equal(available,f['availability'])
                    arrays.append(f['predictions'][keep].copy())
                score=wf1(y,arrays[-1]>0)
                assert abs(score-recorded[seed,rate,a]['metrics']['weighted_f1']*100)<1e-10
                assert recorded[seed,rate,a]['upstream_sha256']==recorded[seed,rate,1.]['upstream_sha256']
                assert recorded[seed,rate,a]['metrics']['mask_sha256']==recorded[seed,rate,1.]['metrics']['mask_sha256']
            preds=np.stack(arrays); correct=(preds>0)==y[None,:]
            base=correct[ALPHAS.index(1.)]; basepred=preds[ALPHAS.index(1.)]
            recover=~base & correct.any(axis=0)
            oracle=basepred.copy()
            for i in np.flatnonzero(recover):
                oracle[i]=preds[np.flatnonzero(correct[:,i])[0],i]
            assert np.array_equal(oracle[base],basepred[base])
            assert np.array_equal(oracle[~recover],basepred[~recover])
            assert np.array_equal((oracle>0)==y,base|recover)
            pattern=np.array([''.join(c for c,b in zip('ATV',v) if b) for v in available[keep]])
            assert set(pattern)<=set(PATTERNS)
            lower=correct[np.array(ALPHAS)<1].any(axis=0)
            upper=correct[np.array(ALPHAS)>1].any(axis=0)
            for i,uid in enumerate(sample_ids):
                samples.append(dict(seed=seed,rate=rate,utterance_id=uid,pattern=pattern[i],
                    label=float(target[keep][i]),baseline_prediction=float(basepred[i]),
                    baseline_correct=bool(base[i]),recoverable=bool(recover[i]),
                    correct_alphas=';'.join(str(a) for j,a in enumerate(ALPHAS) if correct[j,i]),
                    oracle_prediction=float(oracle[i])))
            for group in ('ALL',*PATTERNS):
                select=np.ones(len(y),dtype=bool) if group=='ALL' else pattern==group
                n=int(select.sum()); errors=int((select & ~base).sum()); rescued=int((select & recover).sum())
                row=dict(seed=seed,rate=rate,pattern=group,n=n,baseline_errors=errors,
                    recoverable=rescued,recoverable_fraction=rescued/errors if errors else None,
                    baseline_wf1=wf1(y[select],basepred[select]>0),oracle_wf1=wf1(y[select],oracle[select]>0),
                    lower_only=int((select & recover & lower & ~upper).sum()),
                    upper_only=int((select & recover & upper & ~lower).sum()),
                    either_direction=int((select & recover & upper & lower).sum()))
                row['oracle_delta_pp']=row['oracle_wf1']-row['baseline_wf1'] if n else None
                # Fixed-alpha correct-sample coverage vs sample-wise oracle; diagnostic only.
                row['best_uniform_correct']=max(int((select & c).sum()) for c in correct)
                row['oracle_correct']=int((select & (base|recover)).sum())
                row['oracle_over_best_uniform_correct']=row['oracle_correct']-row['best_uniform_correct']
                cells.append(row)
                for j,a in enumerate(ALPHAS):
                    transitions.append(dict(seed=seed,rate=rate,pattern=group,alpha=a,n=n,
                        wrong_to_right=int((select & ~base & correct[j]).sum()),
                        right_to_wrong=int((select & base & ~correct[j]).sum()),
                        wf1=wf1(y[select],preds[j,select]>0)))
    summary={}
    for group in ('ALL',*PATTERNS):
        chosen=[r for r in cells if r['pattern']==group and r['n']]
        def aggregate(metric,high=False):
            perseed=[statistics.mean(r[metric] for r in chosen if r['seed']==s and
                (not high or r['rate']>=.5) and r[metric] is not None) for s in (66,67,68)]
            return dict(mean=statistics.mean(perseed),sample_sd=statistics.stdev(perseed),per_seed=perseed)
        errors=sum(r['baseline_errors'] for r in chosen);rescued=sum(r['recoverable'] for r in chosen)
        summary[group]=dict(nonempty_cells=len(chosen),sample_occurrences=sum(r['n'] for r in chosen),
            baseline_errors=errors,recoverable=rescued,pooled_recoverable_fraction=rescued/errors,
            macro_recoverable_fraction=aggregate('recoverable_fraction'),
            baseline_wf1=aggregate('baseline_wf1'),oracle_wf1=aggregate('oracle_wf1'),
            delta_pp=aggregate('oracle_delta_pp'),high_baseline=aggregate('baseline_wf1',True),
            high_oracle=aggregate('oracle_wf1',True),
            lower_only=sum(r['lower_only'] for r in chosen),upper_only=sum(r['upper_only'] for r in chosen),
            either_direction=sum(r['either_direction'] for r in chosen),
            opposite_direction_cells=sum(r['lower_only']>0 and r['upper_only']>0 for r in chosen),
            oracle_beats_best_uniform_cells=sum(r['oracle_over_best_uniform_correct']>0 for r in chosen),
            transitions={str(a):dict(wrong_to_right=sum(r['wrong_to_right'] for r in transitions if r['pattern']==group and r['alpha']==a),
                right_to_wrong=sum(r['right_to_wrong'] for r in transitions if r['pattern']==group and r['alpha']==a),
                simultaneous_gain_loss_cells=sum(r['wrong_to_right']>0 and r['right_to_wrong']>0 for r in transitions if r['pattern']==group and r['alpha']==a)) for a in ALPHAS})
    csv_save(out/'transitions_by_seed_rate_pattern_alpha.csv',transitions)
    csv_save(out/'oracle_by_seed_rate_pattern.csv',cells)
    csv_save(out/'samples_label_assisted.csv',samples)
    csv_save(out/'reconstructed_sample_ids.csv',mapping)
    (out/'SUMMARY.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    after={p.name:sha(p) for p in files};assert before==after
    (out/'AUDIT.json').write_text(json.dumps(dict(prediction_sha256=before,
        label_metadata_sha256=sha(labels_path),analysis_script_sha256=sha(Path(__file__)),
        inference_run=False,training_run=False,weights_loaded=False,predictions_unchanged=True,
        threshold='label > 0, prediction > 0',label_filter='exclude labels == 0',
        id_recovery='original sampler epoch=0, conversation-major flatten, original utterance order; exact labels checked for all 216 arrays',
        cells=len(cells),transition_rows=len(transitions),samples=len(samples)),indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
