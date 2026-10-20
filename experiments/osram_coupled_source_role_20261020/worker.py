"""One-stage shared source-role read calibration; no Router or extra loss."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
RATES=tuple(f'{i/10:.1f}' for i in range(8))
REFERENCES={
    'CMUMOSI':Path('/data2/yb/remote_experiments/osram_mosi_gcnet_testbest_20261020/execution_v2/runs/MOSI/seed_66/flat'),
    'CMUMOSEI':Path('/data2/yb/remote_experiments/osram_mosei_gcnet_router_20261020/execution_v1/runs/CMUMOSEI/seed_66/flat')}


def source_role_config(cfg, *, source_role_mode='residual'):
    if source_role_mode not in ('residual','direct'):raise ValueError('Unknown source-role mode')
    if (cfg.backbone_type!='osram' or cfg.osram_readout_fusion!='flat'
            or cfg.osram_bidirectional or cfg.osram_tasr or cfg.osram_source_role
            or cfg.missing_mask_protocol!='gcnet-random-mask-v1'
            or cfg.training_objective!='emotion-only'
            or cfg.checkpoint_selection!='test-oracle-per-rate'
            or cfg.checkpoint_selection_metric!='weighted_f1'):
        raise ValueError('Require original no-JEPA GCNet Flat, no overlapping block')
    return replace(cfg,osram_source_role=True,osram_source_role_mode=source_role_mode)


def set_learning_rates(optimizer,dataset,epoch):
    epochs={'CMUMOSI':150,'CMUMOSEI':60}[dataset]
    if not 0<=epoch<epochs or any(g['name']!='backbone' for g in optimizer.param_groups):
        raise ValueError('Only the original backbone group and approved epoch budget')
    for group in optimizer.param_groups:
        group['lr']=1e-4 if dataset=='CMUMOSI' and epoch>=100 else 1e-3


def original_digest(model):
    """Same baseline digest algorithm, excluding ONLY added source-role tensors."""
    digest=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        if name.startswith('osram.source_role.'):continue
        digest.update(name.encode());digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def validate_reference(identity,prior,initial,epochs, *, source_role_mode='residual'):
    if source_role_mode not in ('residual','direct'):raise ValueError('Unknown source-role mode')
    if prior.get('status')!='complete' or prior.get('epoch')!=epochs:
        raise ValueError('Require completed original Flat reference')
    if prior['initial_backbone_sha256']!=initial:raise ValueError('Original initialization differs')
    expected=dict(prior['identity']['config'],osram_source_role=True)
    if 'osram_source_role_mode' in identity['config']:
        expected['osram_source_role_mode']=source_role_mode
    elif source_role_mode!='residual':
        raise ValueError('Direct config must explicitly name its source-role mode')
    if expected!=json.loads(json.dumps(identity['config'])):raise ValueError('Other training config changed')
    for key in ('split','seed_protocol','mask_hashes'):
        if identity[key]!=prior['identity'][key]:raise ValueError('Reference '+key+' differs')
    return dict(initialization_exact=True,only_source_role_switch_changed=source_role_mode=='residual',
        only_source_role_settings_changed=True,
        evaluation_masks_exact=True,split_exact=True,seed_protocol_exact=True,
        old_weights_loaded=False,baseline_core_source_changed_by_opt_in_integration=True)


def serial(value):
    import torch
    if torch.is_tensor(value):return value.detach().cpu().tolist()
    if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serial(v) for v in value]
    return value


def progress_fields(epoch,best,peak_memory_mib):
    """Also publish valid metadata when final-step recovery skips training."""
    return dict(epoch=epoch,best_by_rate=best,
        selected_epoch_by_rate={r:v['epoch'] for r,v in best.items()},
        peak_memory_mib=peak_memory_mib)


def check_zero_init_arrays(original,current,source_role_mode):
    if source_role_mode not in ('residual','direct'):raise ValueError('Unknown source-role mode')
    if set(original)!=set(current):raise ValueError('Zero-init arrays have different fields')
    exact=True;maximum=0.
    for name in original:
        if name in ('prediction','candidates'):
            exact=exact and np.array_equal(original[name],current[name])
            maximum=max(maximum,float(np.max(np.abs(original[name]-current[name]),initial=0)))
            if source_role_mode=='direct':
                np.testing.assert_allclose(current[name],original[name],atol=1e-5,rtol=1e-5)
            else:np.testing.assert_array_equal(original[name],current[name])
        else:np.testing.assert_array_equal(original[name],current[name])
    return dict(zero_init_predictions_exact=exact,zero_init_predictions_within_tolerance=True,
        zero_init_prediction_max_abs_error=maximum,
        zero_init_atol=1e-5 if source_role_mode=='direct' else 0.,
        zero_init_rtol=1e-5 if source_role_mode=='direct' else 0.)


def main(dataset,argv=None, *, source_role_mode='residual',extra_source_files=()):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('smoke','train'),required=True)
    parser.add_argument('--mode',choices=('joint',),required=True,help='Queue compatibility label; no Router is instantiated')
    parser.add_argument('--seed',type=int,choices=(66,),required=True)
    parser.add_argument('--gpu-uuid',required=True)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    import fcntl
    import torch
    from experiments.osram_text_path_router_joint_mosi_20261019 import run as scalar
    from experiments.osram_mosi_gcnet_testbest_20261020 import worker as selected
    from experiments.osram_mosi_gcnet_router_3seed_20261019 import worker as mosi
    from experiments.osram_mosei_gcnet_router_20261020 import worker as mosei
    from experiments.osram_text_path_router_joint_20261019.run import restore
    from experiments.osram_tasr_20261018 import runner_helpers as h
    setup=mosi if dataset=='CMUMOSI' else mosei
    epochs=150 if dataset=='CMUMOSI' else 60
    if args.output.exists() and not args.resume:raise FileExistsError('Fresh output required')
    if args.resume and args.phase!='train':raise ValueError('Only formal full-state resume')
    args.output.mkdir(parents=True,exist_ok=args.resume)
    lock=(args.output/'RUN.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=args.output/'SUMMARY.json';tick=time.monotonic()
    previous=json.loads(path.read_text()) if args.resume and path.exists() else {}
    if previous.get('status')=='complete':raise ValueError('Completed run must not be repeated')
    summary=dict(status='preflight',dataset=dataset,seed=66,mode=args.mode,phase=args.phase,
        pid=os.getpid(),source_commit=args.source_commit,block='coupled-source-role-'+source_role_mode,router_parameters=0,
        experiment_label='INTERNAL TEST-ORACLE SCREEN')
    hooks=[]
    try:
        torch.set_num_threads(1)
        bundle=setup.setup(args.gpu_uuid,args.output,66)
        # Dataset-root variables must be established before trainer/config imports.
        from gcnet_modality_jepa.train_gcnet import reset_training_stochasticity
        cfg=selected.testbest_config(bundle['cfg']) if dataset=='CMUMOSI' else bundle['cfg']
        bundle['cfg']=source_role_config(cfg,source_role_mode=source_role_mode)
        baseline=bundle['model']
        model=setup.fresh_backbone(bundle['cfg'],bundle['dimensions'],next(baseline.parameters()).device)
        model.requires_grad_(True)
        if original_digest(model)!=bundle['initial_backbone_sha256']:
            raise RuntimeError('Added module changed original parameter initialization')
        # Fresh backbone construction resets the same seed; all evaluation
        # calls below isolate RNG and training explicitly resets it again.
        bundle['model']=model
        splits=('validation','test') if dataset=='CMUMOSI' else ('test',)
        masks={split:(mosi.build_masks(bundle,split) if dataset=='CMUMOSI'
                      else scalar.build_masks(bundle,split)) for split in splits}
        hashes={split:{rate:hashlib.sha256(b''.join(row['availability'].numpy().tobytes()
            for row in bank)).hexdigest() for rate,bank in rates.items()} for split,rates in masks.items()}
        prior=json.loads((REFERENCES[dataset]/'SUMMARY.json').read_text())
        sources=tuple(prior['identity']['source_hashes'])+('gcnet_missing_m3/source_role.py',)+tuple(
            str(Path(__file__).with_name(n).relative_to(ROOT)) for n in ('worker.py','run.py','report.py','mosi.py','mosei.py'))+tuple(extra_source_files)
        identity=dict(source_commit=args.source_commit,source_hashes={n:h.sha(ROOT/n) for n in sources},
            mode=args.mode,block='coupled-source-role-'+source_role_mode,seed=66,config=asdict(bundle['cfg']),
            mask_hashes=hashes,split=bundle['split'],seed_protocol=bundle['seed_protocol'],
            data_manifest_sha256=bundle['data_manifest_sha256'],data_manifest=bundle['data_manifest'],
            data_files=bundle['data_files'],training='one stage, all original parameters and source_role trainable',
            loss='original scalar MSE only',metrics_key='router is legacy scalar-runner alias, NOT a Router model')
        proof=validate_reference(identity,prior,bundle['initial_backbone_sha256'],epochs,source_role_mode=source_role_mode)
        proof.update(path=str(REFERENCES[dataset]),summary_sha256=h.sha(REFERENCES[dataset]/'SUMMARY.json'))
        identity['flat_reference']=proof
        _,base_arrays=scalar.evaluate(bundle,baseline,'flat','test',masks['test']['0.3'],max_batches=1)
        _,new_arrays=scalar.evaluate(bundle,model,'flat','test',masks['test']['0.3'],max_batches=1)
        zero_init=check_zero_init_arrays(base_arrays,new_arrays,source_role_mode)
        del baseline,base_arrays,new_arrays
        initial_decomposition=serial(model.osram.last_source_trace['diagnostics'])
        initial_diagnostics=serial(model.osram.source_role.last_diagnostics)
        ref_history=json.loads((REFERENCES[dataset]/'history.json').read_text())
        optimizer=scalar.make_optimizer(model,bundle['cfg'],'flat')
        observed={};scans=[0]
        def gradient_hook(name):
            def capture(g):
                if not torch.isfinite(g).all():raise RuntimeError('Nonfinite source-role gradient: '+name)
                observed[name]=observed.get(name,False) or bool(g.ne(0).any())
            return capture
        for name,p in model.osram.source_role.named_parameters():hooks.append(p.register_hook(gradient_hook(name)))
        hooks.append(model.osram.register_forward_hook(lambda *unused:scans.__setitem__(0,scans[0]+1)))
        torch.save(masks,args.output/'fixed_masks.pt')
        summary.update(status='running',identity=identity,config=identity['config'],environment=bundle['environment'],
            gpu=bundle['gpu'],initial_backbone_sha256=bundle['initial_backbone_sha256'],flat_reference=proof,
            **zero_init,initial_decomposition=initial_decomposition,
            initial_source_role_diagnostics=initial_diagnostics,
            total_parameters=sum(p.numel() for p in model.parameters()),
            source_role_parameters=sum(p.numel() for p in model.osram.source_role.parameters()))
        h.write_json(args.output/'config.json',identity['config']);h.write_json(path,summary)
        reset_training_stochasticity(66,bundle['cfg'].fold)
        torch.cuda.reset_peak_memory_stats()
        best,history,start={},[],0
        previous_peak=float(previous.get('peak_memory_mib',0.))
        if args.resume:
            state=torch.load(args.output/'last.pt',map_location='cpu',weights_only=False)
            restore(model,optimizer,state,identity);start=state['epoch'];history=state['history'];best=state['best_by_rate']
            selected.validate_selected(args.output,best)
            summary.update(progress_fields(start,best,max(previous_peak,torch.cuda.max_memory_allocated()/2**20)))
        initial_added=h.backbone_digest(model.osram.source_role)
        for epoch in (range(1) if args.phase=='smoke' else range(start,epochs)):
            set_learning_rates(optimizer,dataset,epoch);observed.clear();scans[0]=0
            row=scalar.train_epoch(bundle,model,optimizer,epoch,'flat',max_batches=3 if args.phase=='smoke' else None)
            if scans[0]!=row['optimizer_steps']:raise RuntimeError('Training must scan memory once per batch')
            row['actual_memory_scans']=scans[0]
            row['source_role_gradients']=dict(connected=len(observed),nonzero=sum(observed.values()))
            row['last_training_batch_source_role']=serial(model.osram.source_role.last_diagnostics)
            row['last_training_batch_decomposition']=serial(model.osram.last_source_trace['diagnostics'])
            if not any(observed.values()):raise RuntimeError('No nonzero new-module task gradient in epoch')
            if args.phase=='train':
                for key in ('epoch','mask_trace_sha256','optimizer_steps','learning_rates'):
                    if row[key]!=ref_history[epoch][key]:raise RuntimeError('Reference training '+key+' differs')
                row['original_training_mask_and_update_parity']=True
            evaluated={r:scalar.evaluate(bundle,model,'flat','test',masks['test'][r],
                max_batches=1 if args.phase=='smoke' else None) for r in RATES}
            row['test']={r:v[0] for r,v in evaluated.items()};history.append(row)
            improved=selected.persist_epoch(args.output,model,optimizer,epoch=epoch+1,
                identity=identity,history=history,best=best,evaluated=evaluated)
            summary.update(progress_fields(epoch+1,best,max(previous_peak,torch.cuda.max_memory_allocated()/2**20)),
                seconds=time.monotonic()-tick)
            h.write_json(path,summary)
            print(json.dumps(dict(event='epoch',epoch=epoch+1,task_MSE=row['task_MSE'],
                learning_rates=row['learning_rates'],source_role_gradients=row['source_role_gradients'],
                improved_rates=improved,test_WF1={r:v[0]['router']['WF1'] for r,v in evaluated.items()})),flush=True)
        if args.phase=='smoke':
            if original_digest(model)==bundle['initial_backbone_sha256'] or h.backbone_digest(model.osram.source_role)==initial_added:
                raise RuntimeError('Smoke must update original backbone AND source-role parameters')
            state=torch.load(args.output/'last.pt',map_location='cpu',weights_only=False);restore(model,optimizer,state,identity)
            if any(not torch.equal(v.cpu(),state['model'][n]) for n,v in model.state_dict().items()):raise RuntimeError('Reload mismatch')
            selected.validate_selected(args.output,best)
            _,arrays=scalar.evaluate(bundle,model,'flat','test',masks['test']['0.3'],max_batches=1)
            with np.load(args.output/best['0.3']['predictions'],allow_pickle=False) as saved:
                for n in arrays:np.testing.assert_array_equal(arrays[n],saved[n])
            summary.update(smoke_only=True,smoke=history[0],full_checkpoint_reloaded=True,selected_artifacts_exact_reload=True)
        else:
            summary['evaluation']=selected.publish_selected(args.output,best)
            summary['aggregate']=scalar.aggregate(summary['evaluation'])
        for n,digest in identity['source_hashes'].items():
            if h.sha(ROOT/n)!=digest:raise RuntimeError('Immutable source changed: '+n)
        summary.update(status='complete',seconds=time.monotonic()-tick)
    except BaseException as error:
        summary.update(status='failed',error=repr(error),seconds=time.monotonic()-tick);raise
    finally:
        for hook in hooks:hook.remove()
        h.write_json(path,summary)
