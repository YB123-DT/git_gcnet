"""Frozen readout interventions by strictly-past target-modality exposure."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from experiments.osram_mosei_n10_scale_20261024.run import (
    FLAT, GPU, RATES, SOURCE, paired_counts, validate_checkpoint_identity)

ROOT=Path(__file__).resolve().parents[2]


def history_support(availability, ids, times):
    n=len(ids)
    if availability.shape!=(n,3) or len(times)!=n or not np.isin(availability,[0,1]).all() or not availability.any(-1).all():
        raise ValueError('Valid utterances require nonempty binary availability')
    counts=np.zeros((n,3),dtype=np.int64)
    state,last={},{}
    for i,(cid,t) in enumerate(zip(ids.tolist(),times.tolist())):
        if t!=last.get(cid,-1)+1:
            raise ValueError('Require contiguous ordered history per conversation')
        counts[i]=state.get(cid,np.zeros(3,dtype=np.int64))
        state[cid]=counts[i]+availability[i].astype(np.int64)
        last[cid]=t
    return counts>0,counts


def make_cohorts(availability, ids, times):
    seen,_=history_support(availability,ids,times)
    groups={'ALL':np.ones(len(ids),bool),'FIRST':times==0}
    for i,m in enumerate('ATV'):
        missing=availability[:,i]==0
        groups.update({m+':missing':missing,m+':never':missing&~seen[:,i],
            m+':never_with_history':missing&~seen[:,i]&(times>0),
            m+':seen':missing&seen[:,i],m+':first':missing&(times==0)})
    return groups


def readout_cases():
    cases={'Full':(True,(0,1,2)),'Local':(False,()),
        'LocalBase':(True,()),'LocalGap':(False,(0,1,2))}
    for i,m in enumerate('ATV'):
        others=tuple(j for j in range(3) if j!=i)
        cases.update({'NoGap-'+m:(True,others),
            'NoBaseNoGap-'+m:(False,others),'LocalGap-'+m:(False,(i,))})
    return cases


def select_readout(local,base,gap,case):
    keep_base,keep_gaps=readout_cases()[case]
    selected=gap.clone() if hasattr(gap,'clone') else gap.copy()
    for i in range(3):
        if i not in keep_gaps:selected[:,:,i]=0
    return local,base if keep_base else base*0,selected


def measure(y,pred,full,local,selected):
    valid=selected&(y!=0)
    p=paired_counts(y,full,pred,selected)
    l=paired_counts(y,local,pred,selected)
    result=dict(N_all=int(selected.sum()),N_binary=int(valid.sum()),
        N_positive=p['N_positive'],N_negative=p['N_negative'],
        ACC=None,WF1=None,MSE=None,pred_positive_ratio=None,
        positive_recall=None,negative_recall=None,
        corrections_vs_full=p['corrections'],harms_vs_full=p['harms'],
        corrections_vs_local=l['corrections'],harms_vs_local=l['harms'])
    if selected.any():result['MSE']=float(np.mean((pred[selected]-y[selected])**2))
    if valid.any():
        truth,decision=y[valid]>0,pred[valid]>0
        weighted_f1=0.
        for label in (False,True):
            tp=int(((truth==label)&(decision==label)).sum())
            denominator=int((truth==label).sum()+(decision==label).sum())
            f1=2*tp/denominator if denominator else 0.
            weighted_f1+=f1*float((truth==label).mean())
        result.update(ACC=100*float((truth==decision).mean()),WF1=100*weighted_f1,
            pred_positive_ratio=float((pred[valid]>0).mean()))
        for name,mask in [('positive_recall',valid&(y>0)),('negative_recall',valid&(y<0))]:
            if mask.any():result[name]=float(((pred[mask]>0)==(y[mask]>0)).mean())
    return result


def summarize_arrays(labels,predictions,availability,ids,times,norms):
    groups=make_cohorts(availability,ids,times)
    result={}
    for group,selected in groups.items():
        m=group.split(':')[0]
        cases=list(readout_cases()) if group in ('ALL','FIRST') else [
            'Full','Local','LocalBase','LocalGap','LocalGap-'+m,'NoGap-'+m,'NoBaseNoGap-'+m]
        stats={name:float(values[selected].mean()) if selected.any() else None for name,values in norms.items()}
        result[group]=dict(stats=stats,readouts={case:measure(labels,predictions[case],
            predictions['Full'],predictions['Local'],selected) for case in cases})
    return result


def run(args):
    os.environ.update(CUDA_VISIBLE_DEVICES=GPU,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    import torch
    from experiments.osram_nested_cmerc_tuning_20261023.run import verify_source_snapshot
    from experiments.osram_mosei_flat128_nested_20261024.worker import setup
    from experiments.osram_mosei_flat128_nested_20261024.config import verify_registry_extension
    from experiments.osram_meld_flat_source_combinations_20261013.evaluate import FlatReadoutCapture
    from experiments.osram_nested_delta_off_missing_text_20261010.readout import flat_prediction
    from experiments.osram_text_path_router_joint_mosi_20261019.run import metadata
    from experiments.osram_tasr_20261018 import runner_helpers as h
    if args.output.exists():raise FileExistsError('Preserve old outputs; fresh directory required')
    manifest=verify_source_snapshot(ROOT,args.source_commit)
    verify_registry_extension(ROOT)
    free=int(subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    if free<6144:raise RuntimeError('Insufficient GPU1 free memory; do not disturb training')
    args.output.mkdir(parents=True)
    tick=time.monotonic()
    summary=dict(status='preflight',label='INTERNAL DIAGNOSTIC ONLY',source_commit=args.source_commit,
        manifest_sha256=manifest['manifest_sha256'],gpu_uuid=GPU,training_updates=0,models={})
    h.write_json(args.output/'SUMMARY.json',summary)
    torch.set_num_threads(1)
    try:
        for name,reference,variant in [('Flat128',FLAT,'F66'),('N10',SOURCE,'N10')]:
            original=json.loads((reference/'SUMMARY.json').read_text())
            if original['status']!='complete' or original['epoch']!=60:raise ValueError('Need completed seed66 run')
            for relative,digest in original['identity']['source_hashes'].items():
                if relative.startswith(('gcnet_missing_m3/','gcnet_modality_jepa/')):
                    # Flat predates the approved21 registry entries; their exact
                    # extension has been validated above, all other bytes match.
                    if name=='Flat128' and relative=='gcnet_missing_m3/meaningful_new40_registry.py':continue
                    if h.sha(ROOT/relative)!=digest:raise ValueError('Numerical source mismatch:'+relative)
            bundle,plain=setup('CMUMOSEI',original['method'],GPU,args.output/name/'data',seed=66,fold=1,variant=variant)
            del plain
            model=bundle['model'].eval().requires_grad_(False)
            trainer,loader=bundle['trainer'],bundle['test_loader']
            cfg=json.loads(json.dumps(bundle['cfg'].__dict__))
            if cfg!=original['config'] or bundle['environment']!=original['environment']:
                raise ValueError('Configuration/environment differs from original run')
            record=dict(reference=str(reference),source_summary_sha256=h.sha(reference/'SUMMARY.json'),
                config=cfg,environment=bundle['environment'],gpu=bundle['gpu'],data=bundle['data'],
                mask_reference=bundle['mask_reference'],rates={})
            summary['models'][name]=record
            scans=[0]
            actual_scan=model.osram._scan
            def counted(*a,**kw):
                scans[0]+=1
                return actual_scan(*a,**kw)
            model.osram._scan=counted
            try:
                for rate in RATES:
                    selected=original['best_by_rate'][rate]
                    ckpt=reference/selected['checkpoint'];predfile=reference/selected['predictions']
                    if h.sha(ckpt)!=selected['checkpoint_sha256'] or h.sha(predfile)!=selected['prediction_sha256']:
                        raise ValueError('Protected checkpoint/prediction changed')
                    state=torch.load(ckpt,map_location='cpu',weights_only=False)
                    validate_checkpoint_identity(state,selected,original['config'])
                    model.load_state_dict(state['model'],strict=True);del state
                    before=h.backbone_digest(model)
                    with np.load(predfile,allow_pickle=False) as p:saved={k:p[k].copy() for k in p.files}
                    if name=='N10':
                        other=summary['models']['Flat128']['rates'][rate]['prediction_file']
                        with np.load(args.output/other,allow_pickle=False) as a:
                            for k in ('labels','availability','conversation_ids','utterance_indices'):
                                np.testing.assert_array_equal(saved[k],a[k])
                    loader.sampler.set_epoch(0)
                    pieces={k:[] for k in readout_cases()}
                    norms={k:[] for k in ('Base_norm','Gap-A_norm','Gap-T_norm','Gap-V_norm')}
                    cursor=0;start=scans[0];error=0.
                    with h.isolated_rng(66+99193),torch.no_grad():
                        for i,raw in enumerate(loader):
                            data=trainer._move_batch(raw,torch.device('cuda:0'));meta=metadata(data);n=len(meta['labels'])
                            bank=bundle['masks'][rate][i]
                            for k,v in meta.items():
                                np.testing.assert_array_equal(v,saved[k][cursor:cursor+n])
                                np.testing.assert_array_equal(v,bank[k])
                            av=bank['availability'].to(data[7].device)
                            valid=data[7].bool()
                            np.testing.assert_array_equal(av.transpose(0,1)[valid].cpu().numpy(),saved['availability'][cursor:cursor+n])
                            view=trainer._prepare_view_from_primary_masks(data,av,av,bundle['dimensions'])
                            initial=scans[0]
                            with FlatReadoutCapture(model.osram) as capture:direct=h.forward(model,view).squeeze(-1)
                            if scans[0]!=initial+1 or capture.local_calls!=1 or capture.adapter_calls!=1:
                                raise RuntimeError('Expected one scan and one original Flat')
                            ld,cd=model.osram.latent_dim,model.osram.context_dim
                            local=capture.flat_input[...,:ld]
                            contexts=capture.flat_input[...,ld:].reshape(*local.shape[:2],4,cd)
                            base,gap=contexts[:,:,0],contexts[:,:,1:]
                            if contexts[...,512:].ne(0).any() or gap[av.bool()].ne(0).any():
                                raise RuntimeError('Backward/inactive Gap nonzero')
                            for j,k in enumerate(norms):
                                norms[k].append(contexts[:,:,j,:512].norm(dim=-1).T[valid].cpu().numpy())
                            cpu_rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone()
                            for case in pieces:
                                readout=select_readout(local,base,gap,case)
                                p=flat_prediction(model,capture.local,base,gap,av,data[7],readout=readout)
                                values=p.T[valid].cpu().numpy()
                                if not np.isfinite(values).all():raise RuntimeError('Nonfinite prediction')
                                if case=='Full':
                                    expected=saved['prediction'][cursor:cursor+n].reshape(-1)
                                    np.testing.assert_allclose(values,expected,atol=1e-6,rtol=1e-6)
                                    np.testing.assert_array_equal(values>0,expected>0)
                                    np.testing.assert_allclose(values,direct.T[valid].cpu().numpy(),atol=1e-6,rtol=1e-6)
                                    error=max(error,float(abs(values-expected).max()))
                                pieces[case].append(values)
                            if scans[0]!=initial+1 or not torch.equal(cpu_rng,torch.get_rng_state()) or not torch.equal(cuda_rng,torch.cuda.get_rng_state()):
                                raise RuntimeError('Readout changed scan count/RNG')
                            cursor+=n
                    if cursor!=len(saved['labels']) or scans[0]-start!=len(loader):raise ValueError('Incomplete replay')
                    predictions={k:np.concatenate(v) for k,v in pieces.items()}
                    norm_values={k:np.concatenate(v) for k,v in norms.items()}
                    y=saved['labels'];times=saved['utterance_indices'];av=saved['availability'];ids=saved['conversation_ids']
                    for p in predictions.values():
                        np.testing.assert_allclose(p[times==0],predictions['Full'][times==0],atol=1e-6,rtol=1e-6)
                    for values in norm_values.values():np.testing.assert_array_equal(values[times==0],0.)
                    metrics=summarize_arrays(y,predictions,av,ids,times,norm_values)
                    for key,value in metrics['ALL']['readouts']['Full'].items():
                        if key in ('ACC','WF1','MSE') and abs(value-selected['metrics']['router'][key])>1e-5:
                            raise RuntimeError('Original full metrics differ')
                    file=args.output/name/('miss_'+rate.replace('.','p')+'.npz')
                    np.savez_compressed(file,labels=y,availability=av,conversation_ids=ids,utterance_indices=times,
                        **predictions,**norm_values)
                    if h.backbone_digest(model)!=before or h.sha(ckpt)!=selected['checkpoint_sha256']:
                        raise RuntimeError('Frozen state/checkpoint changed')
                    record['rates'][rate]=dict(selected_epoch=selected['epoch'],checkpoint_sha256=selected['checkpoint_sha256'],
                        original_prediction_sha256=selected['prediction_sha256'],prediction_file=str(file.relative_to(args.output)),
                        prediction_sha256=h.sha(file),memory_scans=scans[0]-start,max_full_error=error,groups=metrics)
                    summary.update(status='running',seconds=time.monotonic()-tick)
                    h.write_json(args.output/'SUMMARY.json',summary)
                    print(json.dumps(dict(model=name,rate=rate,seconds=summary['seconds'],full=metrics['ALL']['readouts']['Full'])),flush=True)
            finally:del model.osram._scan
            record['total_memory_scans']=scans[0]
            del model,bundle,trainer,loader
            torch.cuda.empty_cache()
        summary.update(status='complete',seconds=time.monotonic()-tick,
            checks=dict(no_training=True,original_predictions_reproduced=True,same_masks_between_models=True,
                first_turn_reads_zero=True,first_turn_readouts_equal=True,original_parameters_unchanged=True,
                one_scan_per_batch=True,readout_rng_unchanged=True))
        if verify_source_snapshot(ROOT,args.source_commit)!=manifest:raise RuntimeError('Source changed')
    except BaseException as error:
        summary.update(status='failed',error=repr(error),seconds=time.monotonic()-tick)
        raise
    finally:
        h.write_json(args.output/'SUMMARY.json',summary)
        with (args.output/'METRICS.csv').open('w',newline='') as stream:
            fields=['model','rate','group','readout',*measure(np.array([]),np.array([]),np.array([]),np.array([]),np.array([],bool))]
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for model,r in summary['models'].items():
                for rate,item in r['rates'].items():
                    for group,g in item['groups'].items():
                        for case,v in g['readouts'].items():writer.writerow(dict(model=model,rate=rate,group=group,readout=case,**v))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-commit',required=True)
    run(parser.parse_args())
