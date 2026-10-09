"""Replay stored Text deletions and split raw read changes along a frozen probe tangent."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback

import numpy as np

from experiments.osram_frozen_memory_audit_20261009.run import GPUS, REPO, now, write

LABEL='INTERNAL DIAGNOSTIC ONLY; frozen Test-oracle backbone/probes; local probe direction, not pure emotion semantics'


def stored_cases(rows,rate):
    result=[]
    seen=set()
    for row in rows:
        if row['family']!='recent_vs_earlier_T':
            continue
        if row['rate']!=rate or row['split']!='test':
            raise ValueError('stored rate/split mismatch')
        cid,t=str(row['conversation_id']),int(row['utterance_index'])
        if (cid,t) in seen:
            raise ValueError('duplicate stored target')
        seen.add((cid,t))
        for arm in ('delete','control'):
            pos=int(row['position_'+arm])
            if not 0<=pos<t or row['lag_'+arm]!=t-pos:
                raise ValueError('stored intervention must be strictly past')
            if row['modality_'+arm]!='T' or row['deleted_bits']!=1 or row['control_deleted_bits']!=1:
                raise ValueError('requires stored single-Text deletion')
            result.append(dict(conversation_id=cid,utterance_index=t,arm=arm,position=pos,lag=t-pos,
                pattern=int(row['pattern']),y=float(row['y']),pred_original=float(row['pred_real']),
                pred_expected=float(row['pred_'+arm]),rate=rate))
    return result


def check_memory_slots(memory,availability):
    if memory.shape!=(len(availability),4,512) or not np.isfinite(memory).all():
        raise ValueError('bad memory shape/finiteness')
    if np.count_nonzero(memory[:,1:][availability.astype(bool)]):
        raise ValueError('inactive Gap leakage')


def replay(model,views,data,cases,dimensions,device):
    """Each unique stored deletion gets an independent full causal scan."""
    import torch
    from experiments.osram_history_drift_20261002.diagnostic import capture
    by_id={(str(c),int(t)):i for i,(c,t) in enumerate(zip(data['conversation_ids'],data['utterance_indices']))}
    row_indices=np.array([by_id[c['conversation_id'],c['utterance_index']] for c in cases])
    for case,index in zip(cases,row_indices):
        if float(data['labels'][index])!=case['y'] or int(np.dot(data['availability'][index],[1,2,4]))!=case['pattern']:
            raise AssertionError('stored current label/pattern mismatch')
    output={k:data[k][row_indices].copy() for k in ('local','memory','availability','labels','utterance_indices','conversation_ids')}
    output.update(memory_deleted=np.zeros_like(output['memory']),
                  prediction_deleted=np.zeros(len(cases)),local_error=np.zeros(len(cases)),
                  arm=np.array([c['arm'] for c in cases]),position=np.array([c['position'] for c in cases]))
    filled=np.zeros(len(cases),dtype=bool)
    scans=0
    for cpu_view,_ in views:
        mapping={str(c):b for b,c in enumerate(cpu_view['conversation_ids'])}
        jobs={}
        for i,c in enumerate(cases):
            if c['conversation_id'] in mapping:
                jobs.setdefault((mapping[c['conversation_id']],c['position']),[]).append(i)
        keys=list(jobs)
        for start in range(0,len(keys),16):
            chunk=keys[start:start+16]
            columns=torch.tensor([b for b,pos in chunk])
            features=cpu_view['incomplete'].index_select(1,columns).to(device)
            availability=cpu_view['availability'].index_select(1,columns).to(device)
            qmask=cpu_view['qmask'].index_select(0,columns).to(device)
            umask=cpu_view['umask'].index_select(0,columns).to(device)
            lengths=[int(cpu_view['lengths'][b]) for b,pos in chunk]
            for j,(b,pos) in enumerate(chunk):
                if availability[pos,j,1]!=1 or availability[pos,j].sum()<=1:
                    raise ValueError('stored Text deletion no longer valid')
                availability[pos,j,1]=0
                features[pos,j,dimensions[0]:dimensions[0]+dimensions[1]]=0
                for i in jobs[b,pos]:
                    t=cases[i]['utterance_index']
                    if not torch.equal(features[t:,j].cpu(),cpu_view['incomplete'][t:,b]):
                        raise AssertionError('current/future input changed')
                    if not torch.equal(availability[t:,j].cpu(),cpu_view['availability'][t:,b]):
                        raise AssertionError('current/future mask changed')
            captured=capture(model,lambda:model([features],availability,qmask,umask,lengths,predict_missing=False))
            scans+=1
            for j,key in enumerate(chunk):
                for i in jobs[key]:
                    t=cases[i]['utterance_index']
                    local=captured['local'][t,j].cpu().numpy()
                    error=float(np.max(np.abs(local-output['local'][i])))
                    relative=float(np.linalg.norm(local-output['local'][i])/(np.linalg.norm(output['local'][i])+1e-8))
                    pred=float(captured['prediction'][t,j].reshape(-1)[0])
                    if error>1e-5 or relative>1e-6 or abs(pred-cases[i]['pred_expected'])>1e-5:
                        raise AssertionError(f'stored deletion/local replay failed {cases[i]}: {error}, {relative}, {pred}')
                    output['local_error'][i]=error
                    output['prediction_deleted'][i]=pred
                    output['memory_deleted'][i]=torch.cat((captured['base'][t,j][None],captured['gap'][t,j]),dim=0).cpu().numpy()
                    filled[i]=True
    if not filled.all():
        raise AssertionError('missing stored targets')
    check_memory_slots(output['memory'],output['availability'])
    check_memory_slots(output['memory_deleted'],output['availability'])
    return output,dict(scans=scans,cases=len(cases),max_local_error=float(output['local_error'].max(initial=0)))


def evaluate_directions(model,cache,cases,source,output,device,probe_seeds):
    import torch
    from experiments.osram_probe_direction_20261009.direction import FrozenHistoryProbe,probe_direction,split_delta,flat_predict
    from experiments.osram_readout_paths_20260930.run import state_hash,sha
    from experiments.osram_frozen_memory_audit_20261009.probe import make_targets
    old=np.load(source/'test_features.npz',allow_pickle=False)
    targets=make_targets(dict(old))['preceding3_mean']
    lookup={(str(c),int(t)):i for i,(c,t) in enumerate(zip(old['conversation_ids'],old['utterance_indices']))}
    selected=np.array([lookup[c['conversation_id'],c['utterance_index']] for c in cases])
    s_true=targets[selected]
    if not np.isfinite(s_true).all():
        raise AssertionError('historical target missing')
    all_rows=[]
    checks=[]
    source_hashes={}
    for seed in probe_seeds:
        folder=source/'probes'/f'seed{seed}'/'preceding3_mean/B'
        weights=folder/'best.pt'
        preprocessing=source/'probes/preprocessing.npz'
        for path in (weights,preprocessing,folder/'predictions.npz'):
            source_hashes[str(path)]=sha(path)
        probe=FrozenHistoryProbe(weights,preprocessing,device=device)
        probe_hash=state_hash(probe)
        prior=np.load(folder/'predictions.npz',allow_pickle=False)
        pred_lookup={int(r):float(p) for r,p in zip(prior['test_row'],prior['test_prediction'])}
        units=[]
        maxima=dict(full_replay_error=0.,base_replay_error=0.,probe_parity_error=0.,reconstruction_error=0.,orthogonality_error=0.)
        parity_count=0
        for start in range(0,len(cases),128):
            end=min(start+128,len(cases))
            def tensor(key):
                return torch.as_tensor(cache[key][start:end],device=device)
            local,r,a,indices=tensor('local'),tensor('memory'),tensor('availability'),tensor('utterance_indices')
            rprime=tensor('memory_deleted')
            tangent=probe_direction(probe,local,r,a,indices)
            split=split_delta(rprime.double()-r.double(),tangent['u'],tangent['valid_gradient'])
            memories=dict(base=r,full=rprime,hist=(r.double()+split['history']).to(r.dtype),
                          other=(r.double()+split['other']).to(r.dtype))
            yp,sp={},{}
            with torch.no_grad():
                for key,memory in memories.items():
                    check_memory_slots(memory.cpu().numpy(),a.cpu().numpy())
                    yp[key]=flat_predict(model,local,memory,a,indices).detach().cpu().numpy()
                    sp[key]=probe(local,memory,a,indices).detach().cpu().numpy().reshape(-1)
            for k,value in {**{'y_'+k:v for k,v in yp.items()},**{'s_'+k:v for k,v in sp.items()}}.items():
                if not np.isfinite(value).all():
                    raise AssertionError(f'nonfinite prediction {k}')
            for j,i in enumerate(range(start,end)):
                source_case=cases[i]
                full_error=abs(float(yp['full'][j])-cache['prediction_deleted'][i])
                base_error=abs(float(yp['base'][j])-source_case['pred_original'])
                parity_error=abs(float(sp['base'][j])-pred_lookup[selected[i]]) if selected[i] in pred_lookup else None
                parity_count+=int(parity_error is not None)
                if max(full_error,base_error,parity_error or 0)>1e-5:
                    raise AssertionError(f'readout/probe replay failed: {full_error}, {base_error}, {parity_error}')
                reconstruction=float(split['reconstruction_error'][j])
                orthogonality=float(split['orthogonality_error'][j])
                if reconstruction>1e-8 or orthogonality>1e-8:
                    raise AssertionError('float64 decomposition failed')
                row={k:source_case[k] for k in ('rate','conversation_id','utterance_index','arm','lag','pattern','y')}
                row.update(probe_seed=seed,case_index=i,s_true=float(s_true[i]),
                    delta_r_norm=float(split['full_norm'][j]),hist_norm=float(split['history_norm'][j]),
                    other_norm=float(split['other_norm'][j]),gradient_norm=float(tangent['gradient_norm'][j]),
                    projection_coefficient=float(split['projection_coefficient'][j]),
                    probe_linear_delta=float(tangent['gradient_norm'][j]*split['projection_coefficient'][j]),
                    valid_gradient=bool(tangent['valid_gradient'][j]),local_error=float(cache['local_error'][i]),
                    full_replay_error=full_error,base_replay_error=base_error,probe_parity_error=parity_error,
                    reconstruction_error=reconstruction,orthogonality_error=orthogonality,
                    **{'y_'+k:float(v[j]) for k,v in yp.items()},**{'s_'+k:float(v[j]) for k,v in sp.items()})
                all_rows.append(row)
                for name in maxima:
                    maxima[name]=max(maxima[name],row.get(name) or 0)
            units.append(tangent['u'].detach().cpu().float().numpy())
        np.savez_compressed(output/f'directions_seed{seed}.npz',u=np.concatenate(units))
        if state_hash(probe)!=probe_hash or any(p.requires_grad for p in probe.parameters()):
            raise AssertionError('frozen probe changed')
        checks.append(dict(probe_seed=seed,probe_state_sha256=probe_hash,probe_unchanged=True,
                           original_probe_prediction_pairs_checked=parity_count,**maxima))
        print(json.dumps(dict(phase='directions_complete',rate=cases[0]['rate'],probe_seed=seed,rows=len(cases))),flush=True)
    for path,digest in source_hashes.items():
        if sha(Path(path))!=digest:
            raise AssertionError('source probe artifact modified')
    with (output/'rows.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(all_rows[0]))
        writer.writeheader(); writer.writerows(all_rows)
    return dict(checks=checks,source_sha256=source_hashes,rows=len(all_rows))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-audit',type=Path,default=Path('/data2/yb/remote_experiments/osram_frozen_memory_audit_20261009'))
    p.add_argument('--reference',type=Path,default=Path('/data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full/seed_66'))
    p.add_argument('--dataset-root',type=Path,default=Path('/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset'))
    p.add_argument('--rates',nargs='+',type=float,default=[i/10 for i in range(8)])
    p.add_argument('--gpu',type=int,choices=tuple(GPUS),default=2)
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args()
    if len(set(args.rates))!=len(args.rates) or any(r not in [i/10 for i in range(8)] for r in args.rates):
        p.error('invalid rates')
    uuid=subprocess.check_output(['nvidia-smi',f'--id={args.gpu}','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    if uuid!=GPUS[args.gpu] or os.environ.get('CUDA_VISIBLE_DEVICES')!=uuid:
        raise ValueError('explicit healthy GPU UUID required')
    args.output.mkdir(parents=True,exist_ok=False)
    status=dict(label=LABEL,status='running',started_utc=now(),pid=os.getpid(),gpu=args.gpu,gpu_uuid=uuid,
                rates=args.rates,completed=[],records=[],check_only=args.check_only,command=sys.argv)
    write(args.output/'STATUS.json',status)
    try:
        import torch
        from gcnet_modality_jepa.train_gcnet import get_loaders
        from gcnet_missing_m3.train_gcnet import TrainConfig,_schedules
        from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model
        from experiments.osram_readout_paths_20260930.run import state_hash,sha,split_ids,configuration_dict
        from experiments.osram_frozen_memory_audit_20261009.run import extract
        from experiments.osram_probe_direction_20261009.analyze import analyze
        os.environ['GCNET_DATASET_ROOT']=str(args.dataset_root)
        torch.set_num_threads(2); torch.manual_seed(66)
        configuration_dict(66,args.reference.parent)
        cfg=TrainConfig(**json.loads((args.reference/'config.json').read_text()))
        if cfg.task_regression_loss!='mse' or cfg.mosi_task_mode!='regression':
            raise ValueError('original regression model required')
        features=[str(args.dataset_root/'CMUMOSI/features'/n) for n in ('wav2vec-large-c-UTT','deberta-large-4-UTT','manet_UTT')]
        train,val,test,*dimensions=get_loaders(audio_root=features[0],text_root=features[1],video_root=features[2],
            num_folder=1,dataset='CMUMOSI',batch_size=cfg.batch_size,num_workers=0,seed=66,
            validation_fraction=cfg.validation_fraction,evaluation_protocol=cfg.evaluation_protocol)
        ids=split_ids(dict(train=train[0],validation=val[0],test=test[0]))['test']
        device=torch.device('cuda:0')
        model=_build_model(cfg,tuple(dimensions)).to(device).requires_grad_(False).eval()
        provenance=dict(label=LABEL,host=platform.node(),torch=torch.__version__,numpy=np.__version__,
            source_audit=str(args.source_audit),training=False,primary_probe_seed=66,probe_seeds=[66,67,68],
            code_sha256={str(f.relative_to(REPO)):sha(f) for f in Path(__file__).parent.glob('*.py')},
            snapshot_commit=json.loads((REPO/'SNAPSHOT.json').read_text())['code_commit'] if (REPO/'SNAPSHOT.json').exists() else None)
        write(args.output/'PROVENANCE.json',provenance)
        for rate in args.rates:
            tag=f'rate_{rate:.1f}'.replace('.','p')
            source=args.source_audit/'runs'/('low' if rate<.4 else 'high')/tag
            old_result=json.loads((source/'RESULT.json').read_text())
            for name in ('test_features.npz','interventions.json'):
                if sha(source/name)!=old_result['artifacts'][name]:
                    raise AssertionError('source cache hash mismatch')
            checkpoint=args.reference/f'best_miss_{rate:.1f}.pt'.replace('.','p',1)
            checkpoint_hash=sha(checkpoint)
            if checkpoint_hash!=old_result['checkpoint_sha256']:
                raise AssertionError('not the original parent checkpoint')
            weights=torch.load(checkpoint,map_location='cpu',weights_only=False)
            model.load_state_dict(weights['model'],strict=True); del weights
            frozen_hash=state_hash(model)
            if frozen_hash!=old_result['frozen_state_sha256']:
                raise AssertionError('original model state differs')
            data,views,maskhash=extract(model,test[0],_schedules(cfg,'test')[rate],tuple(dimensions),device,ids)
            if maskhash!=old_result['mask_sha256']['test']:
                raise AssertionError('original mask/order changed')
            cached=np.load(source/'test_features.npz',allow_pickle=False)
            for key,value in data.items():
                if key in ('conversation_ids','utterance_indices','availability','labels','speaker'):
                    if not np.array_equal(value,cached[key]):
                        raise AssertionError('cached sample/mask mismatch '+key)
                elif not np.allclose(value,cached[key],atol=1e-5,rtol=1e-6):
                    raise AssertionError('cached representation mismatch '+key)
            cases=stored_cases(json.loads((source/'interventions.json').read_text())['rows'],rate)
            full_count=len(cases)
            if args.check_only: cases=cases[:16]
            if not cases: raise ValueError('no eligible stored cases')
            destination=args.output/tag; destination.mkdir()
            cache,replay_check=replay(model,views,data,cases,tuple(dimensions),device)
            np.savez_compressed(destination/'reads.npz',**cache)
            write(destination/'cases.json',cases)
            print(json.dumps(dict(rate=rate,phase='history_replayed',**replay_check)),flush=True)
            diagnostic=evaluate_directions(model,cache,cases,source,destination,device,(66,) if args.check_only else (66,67,68))
            if state_hash(model)!=frozen_hash or sha(checkpoint)!=checkpoint_hash:
                raise AssertionError('main model changed')
            for path,digest in diagnostic['source_sha256'].items():
                relative=str(Path(path).relative_to(source))
                if old_result['artifacts'][relative]!=digest:
                    raise AssertionError('probe source differs from completed audit')
            record=dict(rate=rate,source=str(source),source_result_sha256=sha(source/'RESULT.json'),
                source_intervention_sha256=sha(source/'interventions.json'),checkpoint_sha256=checkpoint_hash,
                original_eligible_cases=full_count,model_state_sha256=frozen_hash,model_unchanged=True,
                mask_sha256=maskhash,replay=replay_check,**diagnostic,
                artifacts={str(f.relative_to(destination)):sha(f) for f in destination.iterdir() if f.is_file()})
            write(destination/'RESULT.json',record)
            status['completed'].append(rate); status['records'].append(record)
            write(args.output/'STATUS.json',status)
            print(json.dumps(dict(rate=rate,phase='complete',rows=diagnostic['rows'])),flush=True)
        status.update(status='complete',finished_utc=now())
        write(args.output/'STATUS.json',status)
        analyze(args.output,args.output/'summary',bootstrap=0 if args.check_only else 500)
    except BaseException:
        status.update(status='failed',error=traceback.format_exc(),finished_utc=now())
        raise
    finally:
        write(args.output/'STATUS.json',status)


if __name__=='__main__':
    main()
