"""Finite frozen-Flat three-probability Router training; never changes OSRAM."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
ROUTER_KEYS=('local','base_text','gap_text','text_trajectory','candidates','eligible')
PROTOCOL=dict(seed=66,epochs=30,learning_rate=3e-4,weight_decay=1e-5,batch_size=32,
    optimizer='adam',clip=1.,missing='gcnet-random-mask-v1',training='cyclic-random-only',
    candidate_order=['BG','B','G'],selection='validation-eight-rate-ACC-then-lower-CE',
    loss='sample-mean-mixture-NLL-all-valid',initialization='uniform-on-eligible-full-only-elsewhere')


def optimize_record(router,optimizer,item,clip):
    import torch
    from torch.nn import functional as F
    optimizer.zero_grad(set_to_none=True)
    router.train()
    logp,_=router(**{k:item[k] for k in ROUTER_KEYS})
    loss=F.nll_loss(logp,item['labels'].long())
    if not torch.isfinite(loss):raise RuntimeError('Nonfinite mixture NLL')
    eligible=int(item['eligible'].sum());n=len(item['labels'])
    result=dict(loss=float(loss.detach()),N=n,eligible=eligible,optimizer_steps=0,
                gradient_norm=0.,nonzero_gradient_parameters=0)
    if not eligible:return result
    loss.backward()
    gradients=[p.grad for p in router.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise RuntimeError('Router has missing or nonfinite gradients')
    norm=torch.nn.utils.clip_grad_norm_(router.parameters(),clip)
    if not torch.isfinite(norm):raise RuntimeError('Nonfinite Router gradient norm')
    result.update(optimizer_steps=1,gradient_norm=float(norm),
                  nonzero_gradient_parameters=sum(bool(g.ne(0).any()) for g in gradients))
    optimizer.step()
    return result


def predict_records(router,records):
    import torch
    from torch.nn import functional as F
    router.eval();parts={}
    with torch.no_grad():
        for item in records:
            logp,weights=router(**{k:item[k] for k in ROUTER_KEYS})
            uniform=torch.logsumexp(F.log_softmax(item['candidates'],dim=-1),dim=1)-np.log(3.)
            uniform=torch.where(item['eligible'][:,None],uniform,F.log_softmax(item['candidates'][:,0],dim=-1))
            baseline=F.log_softmax(item['candidates'][:,0],dim=-1)
            torch.testing.assert_close(logp[~item['eligible']],baseline[~item['eligible']],rtol=0,atol=0)
            values={k:item[k].detach().cpu().numpy() for k in ('candidates','eligible','labels','availability')}
            values.update(log_probs_router=logp.cpu().numpy(),log_probs_uniform=uniform.cpu().numpy(),
                          weights=weights.cpu().numpy(),**item['meta'])
            for k,v in values.items():parts.setdefault(k,[]).append(v)
    if not parts:raise RuntimeError('Empty prediction records')
    return {k:np.concatenate(v) for k,v in parts.items()}


def summarize_predictions(arrays,classes):
    from experiments.osram_12_source_evidence_audit_20261019.analysis import _metrics,_paired
    y=arrays['labels'];candidate=arrays['candidates'];chosen=arrays['eligible'].astype(bool)
    names=dict(full=candidate[:,0],candidate_B=candidate[:,1],candidate_G=candidate[:,2],
               uniform=arrays['log_probs_uniform'],router=arrays['log_probs_router'])
    ids=np.arange(classes)
    out={k:_metrics(y,v,ids) for k,v in names.items()}
    out.update(eligible_N=int(chosen.sum()),eligible={k:_metrics(y[chosen],v[chosen],ids) for k,v in names.items()},
        paired_vs_full=_paired(y,names['full'],names['router'],ids),
        paired_eligible=_paired(y[chosen],names['full'][chosen],names['router'][chosen],ids))
    weights=arrays['weights'][chosen]
    out['weights']=dict(order=['BG','B','G'],mean=weights.mean(0).tolist() if len(weights) else None,
        q10=np.quantile(weights,.1,axis=0).tolist() if len(weights) else None,
        q90=np.quantile(weights,.9,axis=0).tolist() if len(weights) else None)
    return out


def selection_key(metrics):
    return (sum(v['router']['ACC'] for v in metrics.values())/len(metrics),
            -sum(v['router']['CE'] for v in metrics.values())/len(metrics))


def cache_records(bundle,loader,*,saved=None,batches=None):
    """Explicit fixed masks only. Does not sample fresh evaluation availability."""
    import torch
    from .candidates import pack_view
    from experiments.osram_tasr_20261018 import runner_helpers as h
    records=[];cursor=0;loader.sampler.set_epoch(0)
    with h.isolated_rng(66+88191):
        for index,raw in enumerate(loader):
            data=bundle['trainer']._move_batch(raw,next(bundle['model'].parameters()).device)
            meta=h.metadata(data);n=len(meta['labels'])
            if saved is not None:
                for k,v in meta.items():np.testing.assert_array_equal(v,saved[k][cursor:cursor+n])
                valid=data[7].bool();batch=torch.zeros((*valid.shape,3),device=valid.device)
                batch[valid]=torch.as_tensor(saved['availability'][cursor:cursor+n],device=valid.device,dtype=batch.dtype)
                av=batch.transpose(0,1)
            elif batches is not None:
                row=batches[index]
                for k,v in meta.items():np.testing.assert_array_equal(v,row[k])
                av=row['availability'].to(data[7].device)
            else:raise ValueError('Saved masks required')
            view=bundle['trainer']._prepare_view_from_primary_masks(data,av,av,bundle['dimensions'])
            item=pack_view(bundle,view);item['meta']={k:v for k,v in meta.items() if k!='labels'}
            records.append(item);cursor+=n
    if saved is not None and cursor!=len(saved['labels']):raise RuntimeError('Incomplete mask cohort')
    return records


def train_epoch(bundle,router,optimizer,epoch,max_batches=None):
    from .candidates import pack_view
    from gcnet_missing_m3.mixed_rate import BalancedBatchRateSchedule
    from experiments.osram_tasr_20261018 import runner_helpers as h
    model,trainer,loader=bundle['model'],bundle['trainer'],bundle['train_loader']
    model.eval();loader.sampler.set_epoch(epoch)
    schedules=trainer._schedules(bundle['cfg'],'train')
    cycle=BalancedBatchRateSchedule(rates=bundle['cfg'].train_missing_rates)
    rows=[];digest=hashlib.sha256();observed=slots=0
    for index,raw in enumerate(loader):
        if max_batches is not None and index>=max_batches:break
        data=trainer._move_batch(raw,next(model.parameters()).device)
        rate=cycle.rate_for(epoch,index)
        view=trainer._prepare_view(data,schedules[rate],epoch,bundle['dimensions'])
        item=pack_view(bundle,view)
        rows.append(optimize_record(router,optimizer,item,PROTOCOL['clip']))
        av=view['availability'].detach().cpu().numpy()
        digest.update(json.dumps([str(v) for v in data[-1]],separators=(',',':')).encode());digest.update(av.tobytes())
        observed+=int(av.sum());slots+=int(data[7].sum())*3
        if any(p.requires_grad or p.grad is not None for p in model.parameters()):
            raise RuntimeError('Original model must remain frozen without gradients')
    if not rows:raise RuntimeError('Empty training epoch')
    return dict(epoch=epoch+1,batches=len(rows),memory_scans=len(rows),
        optimizer_steps=sum(r['optimizer_steps'] for r in rows),
        N=sum(r['N'] for r in rows),eligible_N=sum(r['eligible'] for r in rows),
        task_NLL=sum(r['loss']*r['N'] for r in rows)/sum(r['N'] for r in rows),
        nonzero_gradient_parameters_max=max(r['nonzero_gradient_parameters'] for r in rows),
        gradient_norm_max=max(r['gradient_norm'] for r in rows),
        mask_trace_sha256=digest.hexdigest(),actual_missing_fraction=1-observed/max(1,slots))


def verify_test_candidates(arrays,dataset,rate):
    from experiments.osram_tasr_20261018 import runner_helpers as h
    root=Path('/home/qinjinghui/remote_experiments/osram_12_source_evidence_audit_20261019/runs_v1')/dataset
    name='random_'+rate.replace('.','p');manifest=json.loads((root/(name+'.json')).read_text())
    p=root/(name+'.npz')
    if h.sha(p)!=manifest['checks']['output_sha256']:raise RuntimeError('Prior audit NPZ hash changed')
    with np.load(p,allow_pickle=False) as z:
        for k in ('labels','speakers','conversation_ids','utterance_indices','availability'):
            np.testing.assert_array_equal(arrays[k],z[k])
        np.testing.assert_array_equal(arrays['candidates'][:,0],z['logits_full'])
        chosen=arrays['eligible']
        np.testing.assert_array_equal(chosen,(z['availability'][:,1]==0)&z['trajectory'][:,1,3].astype(bool))
        np.testing.assert_array_equal(arrays['candidates'][chosen,1],z['logits_removed'][chosen,2,1])
        np.testing.assert_array_equal(arrays['candidates'][chosen,2],z['logits_removed'][chosen,0,1])
    return dict(all_full_logits_exact=True,eligible_single_removal_logits_exact=True,
                identity_and_masks_exact=True,previous_npz_sha256=manifest['checks']['output_sha256'])


def checkpoint(router,optimizer,epoch,identity,history,best_key):
    from experiments.osram_tasr_20261018 import runner_helpers as h
    return dict(version=1,router={k:v.detach().cpu().clone() for k,v in router.state_dict().items()},
        optimizer=optimizer.state_dict(),rng=h.rng_state(),epoch=epoch,identity=identity,history=history,best_key=best_key)


def restore(router,optimizer,state,identity):
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_12_source_evidence_audit_20261019.run import require_same_rng
    if state['identity']!=identity:raise ValueError('Recovery identity differs from immutable run')
    router.load_state_dict(state['router'],strict=True)
    optimizer.load_state_dict(state['optimizer'])
    h.restore_rng(state['rng'])
    require_same_rng(state['rng'],h.rng_state())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('smoke','train'),required=True)
    parser.add_argument('--dataset',choices=('MELD','IEMOCAPSix'),required=True)
    parser.add_argument('--gpu-uuid',required=True)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    import fcntl
    import torch
    from .router import TextPathRouter
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_tasr_20261018.run import build_validation_masks
    from experiments.osram_tasr_20261018.protocol import RATES
    if args.output.exists() and not args.resume:raise FileExistsError('Use independent output; resume only exact same run')
    if args.resume and args.phase!='train':raise ValueError('Resume only training')
    args.output.mkdir(parents=True,exist_ok=args.resume)
    lock=(args.output/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tick=time.monotonic();path=args.output/'SUMMARY.json'
    summary=dict(status='preflight',dataset=args.dataset,phase=args.phase,source_commit=args.source_commit,pid=os.getpid(),
        protocol=PROTOCOL,claim='INTERNAL FROZEN TEST-ORACLE ANCHOR; Router chosen only on validation')
    try:
        bundle=h.setup_case(args.dataset,args.gpu_uuid,args.output)
        from gcnet_modality_jepa.train_gcnet import reset_training_stochasticity
        model=bundle['model'];model.osram.osram_source_tracking=True;model.osram.osram_source_audit=True
        if model.osram.osram_tasr:raise RuntimeError('TASR must remain off')
        original_digest=h.backbone_digest(model)
        with h.isolated_rng(66+88192):router=TextPathRouter().to(next(model.parameters()).device)
        optimizer=torch.optim.Adam(router.parameters(),lr=PROTOCOL['learning_rate'],weight_decay=PROTOCOL['weight_decay'])
        files=['router.py','candidates.py','run.py','report.py']
        source_hashes={name:h.sha(Path(__file__).with_name(name)) for name in files}
        identity=dict(case=bundle['case'],split=bundle['split'],protocol=PROTOCOL,source_commit=args.source_commit,
                      source_hashes=source_hashes)
        summary.update(identity=identity,environment=bundle['environment'],gpu=bundle['gpu'],
            anchor_parity=bundle['anchor_parity'],router_parameters=sum(p.numel() for p in router.parameters()),
            backbone_parameters=sum(p.numel() for p in model.parameters()),backbone_sha256=original_digest,status='running')
        h.write_json(path,summary);torch.cuda.reset_peak_memory_stats()
        reset_training_stochasticity(66,bundle['case']['fold'])
        if args.phase=='smoke':
            row=train_epoch(bundle,router,optimizer,0,max_batches=3)
            if row['nonzero_gradient_parameters_max']==0:raise RuntimeError('Smoke has no nonzero Router update')
            state=checkpoint(router,optimizer,1,identity,[row],None)
            h.atomic_torch_save(args.output/'smoke.pt',state)
            prior={k:v.detach().clone() for k,v in router.state_dict().items()}
            stored=torch.load(args.output/'smoke.pt',map_location='cpu',weights_only=False)
            restore(router,optimizer,stored,identity)
            if not all(torch.equal(v,router.state_dict()[k]) for k,v in prior.items()):raise RuntimeError('Router recovery mismatch')
            records=cache_records(bundle,bundle['test_loader'],saved=bundle['selected']['0.3'])
            arrays=predict_records(router,records)
            summary.update(smoke=row,smoke_only=True,optimizer_rng_reloaded=True,
                candidate_checks=verify_test_candidates(arrays,args.dataset,'0.3'))
        else:
            masks=build_validation_masks(bundle)
            val_records={rate:cache_records(bundle,bundle['validation_loader'],batches=masks[rate]) for rate in RATES}
            val_hash={rate:hashlib.sha256(b''.join(r['availability'].numpy().tobytes() for r in masks[rate])).hexdigest() for rate in RATES}
            identity['validation_masks']=val_hash;summary['validation_mask_hashes']=val_hash
            start,history,best_key=0,[],None
            reset_training_stochasticity(66,bundle['case']['fold'])
            if args.resume:
                state=torch.load(args.output/'last.pt',map_location='cpu',weights_only=False)
                restore(router,optimizer,state,identity)
                start,history,best_key=state['epoch'],state['history'],state['best_key']
            for epoch in range(start,PROTOCOL['epochs']):
                row=train_epoch(bundle,router,optimizer,epoch)
                with h.isolated_rng(66+88193):
                    metrics={rate:summarize_predictions(predict_records(router,records),bundle['case']['classes'])
                             for rate,records in val_records.items()}
                key=selection_key(metrics);improved=best_key is None or tuple(key)>tuple(best_key)
                if improved:best_key=key
                row.update(validation=metrics,validation_key=key);history.append(row)
                state=checkpoint(router,optimizer,epoch+1,identity,history,best_key)
                if improved:h.atomic_torch_save(args.output/'best_router.pt',state)
                h.atomic_torch_save(args.output/'last.pt',state);h.write_json(args.output/'history.json',history)
                summary.update(epoch=epoch+1,best_validation_key=best_key,seconds=time.monotonic()-tick,
                    peak_memory_mib=torch.cuda.max_memory_allocated()/2**20)
                h.write_json(path,summary)
                print(json.dumps(dict(event='epoch',epoch=epoch+1,loss=row['task_NLL'],eligible=row['eligible_N'],
                    validation=key,best=best_key)),flush=True)
            state=torch.load(args.output/'best_router.pt',map_location='cpu',weights_only=False)
            router.load_state_dict(state['router'],strict=True)
            summary.update(selected_epoch=state['epoch'],evaluation={},candidate_checks={})
            evaluation=args.output/'evaluation';evaluation.mkdir(exist_ok=True)
            for rate in RATES:
                records=cache_records(bundle,bundle['test_loader'],saved=bundle['selected'][rate])
                arrays=predict_records(router,records)
                summary['candidate_checks'][rate]=verify_test_candidates(arrays,args.dataset,rate)
                summary['evaluation'][rate]=summarize_predictions(arrays,bundle['case']['classes'])
                name='random_'+rate.replace('.','p')+'.npz';np.savez_compressed(evaluation/name,**arrays)
                summary['evaluation'][rate]['output_sha256']=h.sha(evaluation/name)
                h.write_json(path,summary)
            summary['aggregate']={tag:h.aggregate_rates({r:v[tag] for r,v in summary['evaluation'].items()})
                                  for tag in ('full','router','uniform','candidate_B','candidate_G')}
            summary['persistent_scope']='All persistent patterns bypass by construction: Text present or never historically seen; no claimed improvement.'
        if h.backbone_digest(model)!=original_digest:raise RuntimeError('Original model parameters changed')
        for name,digest in source_hashes.items():
            if h.sha(Path(__file__).with_name(name))!=digest:raise RuntimeError('Live source changed')
        summary.update(status='complete',backbone_unchanged=True,seconds=time.monotonic()-tick,
            peak_memory_mib=torch.cuda.max_memory_allocated()/2**20)
        if args.phase=='train':
            from .report import write_report
            write_report(args.output,summary)
    except BaseException as error:
        summary.update(status='failed',error=repr(error),seconds=time.monotonic()-tick);raise
    finally:h.write_json(path,summary)


if __name__=='__main__':main()
