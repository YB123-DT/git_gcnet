"""One fixed-backbone TASR adapter job: seed66 MELD or IEMOCAP6 fold5 only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--phase',choices=('smoke','train','evaluate'),required=True)
    result.add_argument('--dataset',choices=('MELD','IEMOCAPSix'),required=True)
    result.add_argument('--mode',choices=('real','neutral'),required=True)
    result.add_argument('--output',type=Path,required=True)
    result.add_argument('--gpu-uuid',required=True)
    result.add_argument('--resume',action='store_true')
    result.add_argument('--adapter-checkpoint',type=Path)
    result.add_argument('--adapter-learning-rate',type=float,default=None,
                        help='Only adapter Adam LR; omitted preserves the original3e-4 protocol')
    return result


def training_protocol(args):
    from experiments.osram_tasr_20261018.protocol import TRAINING_PROTOCOL
    protocol = dict(TRAINING_PROTOCOL)
    if args.adapter_learning_rate is not None:
        rate = args.adapter_learning_rate
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError('Adapter learning rate must be positive and finite')
        protocol['learning_rate'] = rate
    return protocol


def build_validation_masks(bundle):
    """One fixed original GCNet mask bank; isolated from training RNG."""
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_tasr_20261018.protocol import RATES
    trainer,cfg,loader = bundle['trainer'],bundle['cfg'],bundle['validation_loader']
    loader.sampler.set_epoch(0)
    masks = {}
    with h.isolated_rng(66+76192):
        schedules = trainer._schedules(cfg,'validation')
        for rate in RATES:
            rows = []
            for raw in loader:
                data = trainer._move_batch(raw,next(bundle['model'].parameters()).device)
                view = trainer._prepare_view(data,schedules[float(rate)],0,bundle['dimensions'])
                rows.append(dict(h.metadata(data),availability=view['availability'].detach().cpu()))
            masks[rate] = rows
    return masks


def validation_scores(bundle,masks):
    from experiments.osram_tasr_20261018 import runner_helpers as h
    with h.isolated_rng(66+76193):
        return {rate:h.collect(bundle['model'],bundle['validation_loader'],bundle['trainer'],
            bundle['dimensions'],bundle['case']['classes'],batches=rows)[0] for rate,rows in masks.items()}


def train_epoch(bundle,optimizer,epoch,*,max_batches=None):
    """Unchanged random sampler + independent full-input persistent view."""
    import numpy as np
    import torch
    from gcnet_missing_m3.mixed_rate import BalancedBatchRateSchedule
    from experiments.osram_tasr_20261018.protocol import persistent_availability,PATTERNS,TRAINING_PROTOCOL
    from experiments.osram_tasr_20261018 import runner_helpers as h
    model,trainer,loader = bundle['model'],bundle['trainer'],bundle['train_loader']
    model.eval()  # Frozen backbone including ALL dropout remains deterministic.
    loader.sampler.set_epoch(epoch)
    schedules = trainer._schedules(bundle['cfg'],'train')
    cycle = BalancedBatchRateSchedule(rates=bundle['cfg'].train_missing_rates)
    reports, patterns, digest = [],np.zeros(6,dtype=np.int64),hashlib.sha256()
    observed = valid_slots = persistent_observed = 0
    device = next(model.parameters()).device
    for index,raw in enumerate(loader):
        if max_batches is not None and index >= max_batches: break
        data = trainer._move_batch(raw,device)
        rate = cycle.rate_for(epoch,index)
        view1 = trainer._prepare_view(data,schedules[rate],epoch,bundle['dimensions'])
        availability,ids = persistent_availability(data[7],seed=66,epoch=epoch,batch_index=index)
        # Critically, use RAW complete features, not View1's incomplete tensor.
        view2 = trainer._prepare_view_from_primary_masks(data,availability,availability,bundle['dimensions'])
        optimizer.zero_grad(set_to_none=True)
        loss1 = trainer._task_loss(bundle['case']['dataset'],h.forward(model,view1),view1['labels'],view1['umask'])
        # Backpropagate first view before the second to bound live graph memory;
        # gradients add before the SAME clipping and single optimizer update.
        (TRAINING_PROTOCOL['random_weight']*loss1).backward()
        loss2 = trainer._task_loss(bundle['case']['dataset'],h.forward(model,view2),view2['labels'],view2['umask'])
        (TRAINING_PROTOCOL['persistent_weight']*loss2).backward()
        if not torch.isfinite(loss1) or not torch.isfinite(loss2): raise RuntimeError('Nonfinite task loss')
        gradient = h.check_adapter_gradients(model)
        norm = float(torch.nn.utils.clip_grad_norm_(model.osram.tasr.parameters(),bundle['cfg'].gradient_clip_norm))
        if not np.isfinite(norm): raise RuntimeError('Nonfinite gradient norm')
        optimizer.step()
        random_mask = view1['availability'].detach().cpu().numpy()
        digest.update(json.dumps([str(v) for v in data[-1]],separators=(',',':')).encode())
        digest.update(random_mask.tobytes())
        counts = np.bincount(ids.cpu().numpy(),minlength=6); patterns += counts
        observed += int(random_mask.sum()); valid_slots += int(data[7].sum())*3
        persistent_observed += int(availability.sum())
        reports.append(dict(random_CE=float(loss1.detach()),persistent_CE=float(loss2.detach()),
            loss=.8*float(loss1.detach())+.2*float(loss2.detach()),gradient_norm=norm,**gradient))
    if not reports: raise RuntimeError('No training minibatch')
    return dict(epoch=epoch+1,batches=len(reports),optimizer_steps=len(reports),
                **{k:sum(row[k] for row in reports)/len(reports) for k in ('loss','random_CE','persistent_CE','gradient_norm')},
                nonzero_gradient_parameters_max=max(r['nonzero_parameters'] for r in reports),
                original_random_mask_trace_sha256=digest.hexdigest(),
                persistent_pattern_conversations={name:int(patterns[i]) for i,name in enumerate(PATTERNS)},
                persistent_pattern_fraction={name:float(patterns[i]/patterns.sum()) for i,name in enumerate(PATTERNS)},
                random_actual_missing_fraction=1-observed/max(1,valid_slots),
                persistent_actual_missing_fraction=1-persistent_observed/max(1,valid_slots))


def evaluate_all(bundle,output):
    import numpy as np
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_tasr_20261018.protocol import RATES
    from experiments.osram_persistent_patterns_20261018.evaluate import PATTERNS
    output=Path(output); output.mkdir()
    model=bundle['model']; original_flags=(model.osram.osram_tasr,model.osram.osram_source_tracking)
    groups={'random':RATES,'persistent':tuple(PATTERNS)}
    result={}
    try:
        for group,names in groups.items():
            scores={}
            for name in names:
                saved=bundle['selected'][name if group=='random' else '0.3']
                kwargs=dict(saved=saved,pattern=None if group=='random' else name)
                model.osram.osram_tasr=False; model.osram.osram_source_tracking=False
                base,base_arrays,_=h.collect(model,bundle['test_loader'],bundle['trainer'],bundle['dimensions'],bundle['case']['classes'],**kwargs)
                model.osram.osram_tasr=True; model.osram.osram_source_tracking=True
                full,full_arrays,diagnostics=h.collect(model,bundle['test_loader'],bundle['trainer'],bundle['dimensions'],bundle['case']['classes'],**kwargs)
                filename=name.replace('.','p')
                for tag,arrays in (('F0',base_arrays),('adapter',full_arrays)):
                    np.savez_compressed(output/f'{group}_{filename}_{tag}.npz',**arrays)
                scores[name]=dict(F0=base,adapter=full,paired=h.paired_counts(base_arrays,full_arrays),diagnostics=diagnostics,
                    delta={m:full[m]-base[m] for m in ('ACC','UA','WF1','CE')})
            result[group]=dict(conditions=scores)
            if group=='random':
                for tag in ('F0','adapter'):
                    result[group][tag]=h.aggregate_rates({key:row[tag] for key,row in scores.items()})
            else:
                for scope,chosen in (('six',tuple(PATTERNS)[:-1]),('seven',tuple(PATTERNS))):
                    result[group][scope]={tag:{m:sum(scores[p][tag][m] for p in chosen)/len(chosen)
                        for m in ('ACC','UA','WF1','CE')} for tag in ('F0','adapter')}
    finally:
        model.osram.osram_tasr,model.osram.osram_source_tracking=original_flags
    h.write_json(output/'RESULTS.json',result)
    return result


def main():
    args=parser().parse_args()
    protocol=training_protocol(args)
    if args.output.exists() and not args.resume: raise FileExistsError('Independent new output required; --resume only same-source training')
    if args.resume and args.phase!='train': raise ValueError('--resume is only for adapter training')
    args.output.mkdir(parents=True,exist_ok=args.resume)
    # Set UUID before importing Torch; an internal cuda:0 is never a host index.
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    import torch
    from experiments.osram_tasr_20261018.protocol import TRAINING_PROTOCOL,validation_key
    from experiments.osram_tasr_20261018 import runner_helpers as h
    import fcntl
    lock=(args.output/'RUN.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tick=time.monotonic()
    summary=dict(status='preflight',phase=args.phase,dataset=args.dataset,mode=args.mode,
                 protocol='FROZEN ANCHOR INTERNAL DIAGNOSTIC; adapter selected only on validation',pid=os.getpid())
    path=args.output/'SUMMARY.json'
    try:
        # config imports must occur AFTER setup_case sets dataset environment.
        bundle=h.setup_case(args.dataset,args.gpu_uuid,args.output)
        from gcnet_modality_jepa.train_gcnet import reset_training_stochasticity
        model=bundle['model']; backbone_before=h.backbone_digest(model)
        names=h.attach_tasr(model,args.mode)
        optimizer=torch.optim.Adam(model.osram.tasr.parameters(),lr=protocol['learning_rate'],
                                   weight_decay=protocol['weight_decay'])
        source_files=('gcnet_missing_m3/osram.py','gcnet_missing_m3/tasr.py','gcnet_missing_m3/model.py',
            'gcnet_missing_m3/train_gcnet.py','gcnet_modality_jepa/gcnet_legacy_random_mask.py',
            'experiments/osram_tasr_20261018/run.py','experiments/osram_tasr_20261018/runner_helpers.py',
            'experiments/osram_tasr_20261018/protocol.py')
        identity=dict(case=bundle['case'],training=protocol,mode=args.mode,
                      source_hashes={name:h.sha(ROOT/name) for name in source_files},split=bundle['split'])
        with h.isolated_rng(66+76194):
            _,zero_arrays,_=h.collect(model,bundle['test_loader'],bundle['trainer'],bundle['dimensions'],
                bundle['case']['classes'],saved=bundle['selected']['0.3'])
        import numpy as np
        np.testing.assert_array_equal(zero_arrays['logits'],bundle['anchor_predictions']['logits'])
        summary.update(identity=identity,environment=bundle['environment'],gpu=bundle['gpu'],
            anchor_parity=bundle['anchor_parity'],zero_adapter_logits_bit_exact=True,
            numerical_strategy=bundle['numeric_strategy'],trainable_parameters=names,
            adapter_parameters=sum(p.numel() for p in model.osram.tasr.parameters()),
            total_parameters=sum(p.numel() for p in model.parameters()),
            checkpoint_references=bundle['references'],status='running')
        h.write_json(path,summary)
        # Source adapter/output setup cannot advance the original training stream.
        reset_training_stochasticity(66,bundle['case']['fold'])
        if args.phase=='smoke':
            torch.cuda.reset_peak_memory_stats()
            report=train_epoch(bundle,optimizer,0,max_batches=3)
            if report['nonzero_gradient_parameters_max']==0: raise RuntimeError('No real finite adapter update')
            state=h.checkpoint_payload(model,optimizer,epoch=1,identity=identity,history=[report],best_key=None)
            h.atomic_torch_save(args.output/'smoke.pt',state)
            old={k:v.detach().clone() for k,v in model.osram.tasr.state_dict().items()}
            h.restore_checkpoint(model,optimizer,torch.load(args.output/'smoke.pt',map_location='cpu',weights_only=False),identity=identity)
            if not all(torch.equal(v,model.osram.tasr.state_dict()[k]) for k,v in old.items()):
                raise RuntimeError('Recovery did not exactly restore adapter')
            summary.update(smoke=report,smoke_only=True,optimizer_restore_exact=True)
        elif args.phase=='train':
            masks=build_validation_masks(bundle)
            val_hash={rate:hashlib.sha256(b''.join(row['availability'].numpy().tobytes() for row in rows)).hexdigest() for rate,rows in masks.items()}
            summary['validation_mask_hashes']=val_hash
            start,history,best_key=0,[],None
            if args.resume:
                state=torch.load(args.output/'last.pt',map_location='cpu',weights_only=False)
                h.restore_checkpoint(model,optimizer,state,identity=identity)
                start,history,best_key=state['epoch'],state['history'],state['best_key']
            torch.cuda.reset_peak_memory_stats()
            for epoch in range(start,protocol['epochs']):
                report=train_epoch(bundle,optimizer,epoch)
                metrics=validation_scores(bundle,masks)
                key=validation_key(metrics)
                report.update(validation=metrics,validation_key=key)
                history.append(report)
                improved=best_key is None or tuple(key)>tuple(best_key)
                if improved: best_key=key
                state=h.checkpoint_payload(model,optimizer,epoch=epoch+1,identity=identity,history=history,best_key=best_key)
                # Commit BEST before LAST: a crash can repeat the deterministic
                # epoch, but LAST can never reference an unsaved newer BEST.
                if improved: h.atomic_torch_save(args.output/'best_adapter.pt',state)
                h.atomic_torch_save(args.output/'last.pt',state)
                h.write_json(args.output/'history.json',history)
                summary.update(epoch=epoch+1,validation_best_key=best_key,peak_memory_mib=torch.cuda.max_memory_allocated()/2**20,
                               seconds=time.monotonic()-tick)
                h.write_json(path,summary)
                print(json.dumps(dict(event='epoch_complete',epoch=epoch+1,loss=report['loss'],validation_key=key,best=best_key)),flush=True)
            best=torch.load(args.output/'best_adapter.pt',map_location='cpu',weights_only=False)
            model.osram.tasr.load_state_dict(best['adapter'],strict=True)
            summary['selected_epoch']=best['epoch']
            summary['evaluation']=evaluate_all(bundle,args.output/'evaluation')
        else:
            if args.adapter_checkpoint is None: raise ValueError('--adapter-checkpoint required for evaluate')
            state=torch.load(args.adapter_checkpoint,map_location='cpu',weights_only=False)
            if state['identity']!=identity: raise ValueError('Evaluation checkpoint identity mismatch')
            model.osram.tasr.load_state_dict(state['adapter'],strict=True)
            summary['selected_epoch']=state['epoch']
            summary['evaluation']=evaluate_all(bundle,args.output/'evaluation')
        if h.backbone_digest(model)!=backbone_before: raise RuntimeError('Original backbone parameters changed')
        for name,digest in identity['source_hashes'].items():
            if h.sha(ROOT/name)!=digest: raise RuntimeError('Live experiment source changed: '+name)
        summary.update(status='complete',backbone_unchanged=True,seconds=time.monotonic()-tick,
                       peak_memory_mib=torch.cuda.max_memory_allocated()/2**20)
    except BaseException as error:
        summary.update(status='failed',error=repr(error),seconds=time.monotonic()-tick)
        raise
    finally:
        # Do not claim success from optimizer completion alone.
        h.write_json(path,summary)
        lock.close()


if __name__=='__main__': main()
