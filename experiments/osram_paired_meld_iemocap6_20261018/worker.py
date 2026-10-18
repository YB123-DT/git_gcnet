"""Run existing paired task-only numerics, then frozen Persistent7; no new model."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
HELPERS=Path('/home/qinjinghui/remote_experiments/osram_meld_related_history_diagnostic_20261018/source_470e653')
GPUS={0:'GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45',
      3:'GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a'}


def backbone_identity(model):
    digest=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        if name.startswith('history_contrast_projector.'):
            continue
        digest.update(name.encode());digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return dict(sha256=digest.hexdigest(),parameters=sum(p.numel() for n,p in model.named_parameters()
        if not n.startswith('history_contrast_projector.')))


def check_epoch(metrics):
    paired=metrics['paired_history']
    expected=.5*(paired['task_view1']+paired['task_view2'])
    if (paired['task_view2_weight']!=.5 or not math.isfinite(metrics['loss'])
            or not math.isclose(metrics['loss'],expected,rel_tol=1e-5,abs_tol=1e-5)):
        raise ValueError('Actual task loss is not equal-weight paired task-only')
    return dict(anchor_count=paired['anchor_count'],observed_before=paired['observed_before'],
        observed_dropped=paired['observed_dropped'],
        actual_drop_ratio=paired['observed_dropped']/max(1,paired['observed_before']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',required=True);parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--gpu-uuid',required=True)
    parser.add_argument('--commit',required=True)
    args=parser.parse_args()
    from experiments.osram_paired_meld_iemocap6_20261018.spec import build_jobs,RATES
    from experiments.osram_persistent_patterns_20261018.run import sha,write,check_source
    manifest=check_source()
    jobs={j['id']:j for j in build_jobs(args.output_root)}
    job=jobs[args.job];output=Path(job['output']);base=Path(job['baseline_output'])
    if json.loads(args.config.read_text())!=job['config']:
        raise ValueError('Configuration differs from the finite approved transfer')
    if args.gpu_uuid not in GPUS.values():
        raise ValueError('Only healthy physical GPU0/3; GPU4 prohibited')
    sys.path.append(str(HELPERS))
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        GCNET_CACHE_ROOT=str(args.output_root/'feature_cache'/job['dataset']))
    meld=job['dataset']=='MELD'
    if meld:
        from experiments.osram_meld_related_history_diagnostic_20261018.run import check_sources
        from experiments.osram_meld_optimizer_screen_20261018 import run as screen
        check_sources()
        os.environ.update(GCNET_MELD_PICKLE=str(screen.original.DATASET),
            GCNET_DATASET_ROOT=str(screen.MODEL_SOURCE/'dataset'))
        if sha(screen.original.DATASET)!=screen.original.DATA_SHA:
            raise ValueError('Accepted MELD feature file changed')
        trainer=screen.original.load_historical(screen.MODEL_SOURCE)
        numerical=dict(root=str(screen.MODEL_SOURCE),git_sha=screen.original.HISTORICAL,
                       dataset_sha256=screen.original.DATA_SHA)
        roots=('MELD-pickle',)*3
    else:
        from experiments.osram_persistent_patterns_20261018 import iemocap_source as source
        os.environ['GCNET_DATASET_ROOT']=str(source.DATA_ROOT)
        os.environ.pop('GCNET_MELD_PICKLE',None)
        code_hashes=source.activate_numeric_source()
        from gcnet_missing_m3 import train_gcnet as trainer
        label=source.DATA_ROOT/'IEMOCAP/IEMOCAP_features_raw_6way.pkl'
        if sha(label)!=source.LABEL_HASHES[6]:
            raise ValueError('Original IEMOCAP6 labels changed')
        numerical=dict(root=str(source.SOURCE_ROOT),git_sha=source.TRAINING_SHA,
                       dataset_sha256=source.LABEL_HASHES[6],code_hashes=code_hashes)
        roots=tuple(str(source.DATA_ROOT/'IEMOCAP/features'/n) for n in
                    ('wav2vec-large-c-UTT','deberta-large-4-UTT','manet_UTT'))
        if not all(Path(p).is_dir() for p in roots):
            raise FileNotFoundError('Original IEMOCAP features unavailable')
    import numpy as np
    import torch
    from gcnet_missing_m3.training_resume import TrainingState
    from experiments.osram_meld_write_retention_20261014.run import guard_gpu
    from experiments.osram_iemocap_b6_transfer_20261015.worker import require_replication_environment,repair_uuid_rng
    from experiments.osram_persistent_patterns_20261018.evaluate import classification_metrics
    environment=require_replication_environment(torch)
    repair=repair_uuid_rng(torch);gpu=guard_gpu(args.gpu_uuid)
    if GPUS.get(gpu['host_index'])!=gpu['uuid']:
        raise ValueError('Physical GPU mapping changed')
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    config=trainer.TrainConfig(**job['config'])
    output.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(output/'RUN.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    identity=dict(driver_commit=args.commit,manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),
        config_sha256=sha(args.config),numerical=numerical,baseline_output=str(base))
    path=output/'SUMMARY.json'
    if path.exists():
        previous=json.loads(path.read_text())
        if previous['identity']!=identity:
            raise ValueError('Cannot resume changed source/config/data')
        if previous['status']=='complete':
            return
    summary=dict(status='running',id=job['id'],dataset=job['dataset'],classes=job['classes'],
        seed=66,fold=job['fold'],config=job['config'],identity=identity,manifest=manifest,
        environment=environment,gpu=gpu,pid=os.getpid(),from_scratch=True,
        protocol='INTERNAL TEST-ORACLE; per-rate ACC best with same-forward UA/WF1',
        inference='single view; unchanged CHM/Flat/Local Skip/task head',uuid_rng=repair)
    write(path,summary)
    capture=None
    if meld:
        from experiments.osram_meld_write_tuning_20261017.capture import TrainingCapture
        capture=TrainingCapture(trainer,output)
        parent=screen.original.state_class(TrainingState,capture,output,'W0',base)
        reference_trace=json.loads((base/'MASK_TRACE.json').read_text())
    else:
        from experiments.osram_meld_alternate_features_20261017.worker import compact_state_class
        parent=compact_state_class(TrainingState,torch)

    class AuditedState(parent):
        def bind(self,model,optimizer,**kwargs):
            super().bind(model,optimizer,**kwargs)
            initial=backbone_identity(model)
            if meld and initial!=json.loads((base/'INITIAL_STATE.json').read_text()):
                raise ValueError('Paired training changed original Flat initialization')
            write(output/'BACKBONE_INITIAL_STATE.json',dict(initial,
                matched_to_accepted_baseline=meld,total_parameters=sum(p.numel() for p in model.parameters()),
                inert_projector_parameters=sum(p.numel() for p in model.history_contrast_projector.parameters())))
            return self

        def commit_epoch(self,**kwargs):
            epoch=kwargs['next_epoch'];metrics=kwargs['history'][-1]['train']
            paired=check_epoch(metrics)
            grads=[p.grad for p in self.model.parameters() if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads):
                raise ValueError('Expected finite real classification gradients')
            if any(p.grad is not None for p in self.model.history_contrast_projector.parameters()):
                raise ValueError('Task-only projector unexpectedly received gradient')
            if meld:
                screen.assert_epoch_masks(capture.trace,reference_trace,epoch)
            super().commit_epoch(**kwargs)
            if epoch==1:
                # Read ALREADY saved selected logits, no extra random inference.
                if not meld:
                    from experiments.osram_iemocap_b6_transfer_20261015.worker import verify_first_epoch_retention
                    verify_first_epoch_retention(output,torch,6,cuda_rng_states=len(torch.cuda.get_rng_state_all()))
                write(output/'FIRST_EPOCH_INTEGRATION.json',dict(status='complete',epoch=1,
                    finite_task_gradients=True,inert_projector_has_no_gradient=True,
                    original_mask_trace_equal=meld,peak_memory_mib=torch.cuda.max_memory_allocated()/2**20,
                    **paired))
            best={}
            for rate,ref in self.references.items():
                cp=json.loads(self._checked(ref['selection']).read_text())['checkpoint']
                best[rate]=dict(epoch=cp['epoch'],ACC=100*cp['selection_accuracy'])
                if 'selected_metrics' in cp:
                    best[rate].update(cp['selected_metrics'])
            write(output/'EPOCH_STATUS.json',dict(status='training',epoch=epoch,epochs=config.epochs,
                loss=metrics['loss'],paired=paired,best=best,
                actual_optimizer_lrs=[g['lr'] for g in self.optimizer.param_groups],
                peak_memory_mib=torch.cuda.max_memory_allocated()/2**20))

    state=AuditedState(output,identity=identity)
    tick=time.monotonic();torch.cuda.reset_peak_memory_stats()
    try:
        with capture if capture is not None else nullcontext():
            trainer.run_experiment(config,*roots,str(output),training_state=state)
        history=json.loads((output/'history.json').read_text())
        if len(history)!=config.epochs or len(torch.cuda.get_rng_state_all())!=1:
            raise ValueError('Training or recovery state incomplete')
        totals=dict(anchors=0,observed_before=0,observed_dropped=0)
        for row in history:
            p=check_epoch(row['train']);totals['anchors']+=p['anchor_count']
            for k in ('observed_before','observed_dropped'):totals[k]+=p[k]
        totals['actual_drop_ratio']=totals['observed_dropped']/max(1,totals['observed_before'])
        rates={}
        for rate in RATES:
            checkpoint=output/f'best_miss_{rate.replace(".","p")}.pt'
            cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
            ref=cp['selection_artifact'];npz=(output/ref['path']).resolve()
            if not npz.is_relative_to(output.resolve()) or sha(npz)!=ref['sha256']:
                raise ValueError('Selected prediction artifact mismatch')
            with np.load(npz,allow_pickle=False) as a:
                scores=classification_metrics(a['labels'],a['logits'],job['classes'])
                if (int(a['selected_epoch'])!=cp['epoch']
                        or not np.array_equal(a['predictions'],a['logits'].argmax(1))
                        or abs(scores['ACC']/100-cp['selection_accuracy'])>1e-8
                        or hashlib.sha256(a['availability'].tobytes()).hexdigest()!=ref['availability_sha256']):
                    raise ValueError('UA/ACC must come from the same exact selected predictions')
            rates[rate]=dict(scores,epoch=cp['epoch'],checkpoint_sha256=sha(checkpoint),
                             selection_sha256=ref['sha256'],mask_sha256=ref['availability_sha256'])
            del cp
        summary['testbest']=dict(rates=rates,**{scope:{m:sum(rates[r][m] for r in chosen)/len(chosen)
            for m in ('ACC','UA','WF1','CE')} for scope,chosen in
            (('mean',RATES),('high',('0.5','0.6','0.7')))})
        summary.update(training_status='complete',paired_training=totals,
                       train_seconds=time.monotonic()-tick,peak_memory_mib=torch.cuda.max_memory_allocated()/2**20)
        write(path,summary)
        # One ordinary single-view model; the same retained weights are tested
        # with the pre-fixed singleton=.7 / pair=.3 / full=.0 mapping.
        from gcnet_modality_jepa.train_gcnet import get_loaders
        if meld:
            from experiments.osram_meld_flat_source_combinations_20261013.evaluate import _build_model
        else:
            # Historical helpers contain a seven-class MELD-only builder too;
            # choose the actual IEMOCAP training checkout explicitly.
            import importlib.util
            spec=importlib.util.spec_from_file_location('_paired_original_iemocap_builder',source.SOURCE_ROOT/source.BUILDER)
            builder=importlib.util.module_from_spec(spec)
            previous_path=list(sys.path)
            try:spec.loader.exec_module(builder)
            finally:sys.path[:]=previous_path
            _build_model=builder._build_model
        from experiments.osram_persistent_patterns_20261018.evaluate import evaluate_condition,CHECKPOINT_RATE,PATTERNS
        _,_,loaders,*dimensions=get_loaders(*roots,dataset=config.dataset,
            num_folder=1 if meld else 5,batch_size=32,num_workers=0,seed=66,
            validation_fraction=config.validation_fraction,evaluation_protocol=config.evaluation_protocol)
        model=_build_model(config,dimensions,torch.device('cuda:0'))
        trainer._attach_history_projector(model,config)
        model.eval().requires_grad_(False)
        def prohibit_projector(*_):
            raise RuntimeError('Training-only projector called during inference')
        hook=model.history_contrast_projector.register_forward_pre_hook(prohibit_projector)
        persistent=dict(patterns={},selected_replays={})
        for rate in ('0.7','0.3','0.0'):
            cp=torch.load(output/f'best_miss_{rate.replace(".","p")}.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(cp['model'],strict=True)
            with np.load(output/cp['selection_artifact']['path'],allow_pickle=False) as a:
                saved={k:a[k].copy() for k in a.files}
            replay,full=evaluate_condition(model,loaders[job['fold']-1],dimensions,saved,trainer,None,job['classes'])
            persistent['selected_replays'][rate]=replay
            for pattern in (p for p in PATTERNS if CHECKPOINT_RATE[p]==rate):
                if pattern=='ATV':
                    if not np.all(saved['availability']==1):raise ValueError('Full reference mask changed')
                    result,artifacts=replay,full
                else:
                    result,artifacts=evaluate_condition(model,loaders[job['fold']-1],dimensions,saved,trainer,pattern,job['classes'])
                file=output/f'persistent_{pattern}.npz';np.savez_compressed(file,**artifacts)
                persistent['patterns'][pattern]=dict(result,checkpoint_rate=rate,epoch=cp['epoch'],
                    artifact=str(file),artifact_sha256=sha(file))
        hook.remove()
        summary.update(status='complete',persistent=persistent,total_seconds=time.monotonic()-tick)
    except BaseException as error:
        summary.update(status='failed',error=repr(error),total_seconds=time.monotonic()-tick)
        raise
    finally:
        write(path,summary)
        lock.close()


if __name__=='__main__':
    main()
