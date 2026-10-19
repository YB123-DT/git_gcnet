"""One from-scratch MELD seed66 joint TASR run, original random task loss only."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import inspect
import itertools
import json
import os
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[2]


def make_config():
    from gcnet_missing_m3.train_gcnet import TrainConfig
    from experiments.osram_tasr_20261018.protocol import build_cases
    base = build_cases()['MELD']['expected_config']
    return TrainConfig(**dict(base, osram_source_tracking=True, osram_tasr=True,
        osram_tasr_source_mode='real', checkpoint_selection='validation',
        preserve_selection_artifacts=False))


def fresh_model(cfg, dimensions, device):
    """Mirror the current trainer's model-init seed, including its fixed fold5 key.

    No checkpoint is ever loaded here. The inference-only builder cannot be
    reused: it deliberately freezes everything and seeds using another key.
    """
    from gcnet_missing_m3.model import MissingM3GraphModel
    from gcnet_missing_m3.train_gcnet import _dataset_shape
    from gcnet_modality_jepa.protocol import SeedBundle
    from gcnet_modality_jepa.train_gcnet import set_random_seed
    shape = _dataset_shape(cfg.dataset)
    set_random_seed(SeedBundle(cfg.seed).derive('missing_m3_model_init:fold:5'))
    keywords = {name:getattr(cfg,name) for name in inspect.signature(MissingM3GraphModel).parameters
                if hasattr(cfg,name)}
    for name in ('base_model','adim','tdim','vdim','D_e','graph_hidden_size'):
        keywords.pop(name,None)
    keywords.update(n_speakers=int(shape['num_speakers']), n_classes=int(shape['num_classes']),
        time_attn=cfg.time_attention, no_cuda=device.type!='cuda',
        complete_state_jepa=False, write_state_completion=False, future_state_jepa=False)
    return MissingM3GraphModel(cfg.base_model,*dimensions,cfg.hidden,cfg.hidden//2,**keywords).to(device)


def make_optimizer(model,cfg):
    import torch
    from gcnet_missing_m3.train_gcnet import _optimizer_parameter_groups
    groups, provenance = _optimizer_parameter_groups(model,cfg)
    optimizer = torch.optim.Adam(groups,weight_decay=cfg.weight_decay)
    actual = [id(p) for group in groups for p in group['params']]
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if len(actual)!=len(set(actual)) or set(actual)!=expected:
        raise RuntimeError('Joint optimizer must cover every original trainable parameter exactly once')
    return optimizer,provenance


def gradient_report(model):
    import torch
    prefixes = dict(encoder='observed_set.', writer_key='osram.key_projectors.',
        writer_value='osram.value_projectors.',query='osram.query_projector.',
        flat='osram.emotion_adapter.',classifier='smax_fc.',tasr='osram.tasr.')
    result = dict(finite=True, groups={})
    for group,prefix in prefixes.items():
        gradients=[p.grad for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
        result['groups'][group]=dict(connected=len(gradients),
            nonzero=sum(bool(g.ne(0).any()) for g in gradients))
    gradients=[p.grad for p in model.parameters() if p.grad is not None]
    result['finite']=all(bool(torch.isfinite(g).all()) for g in gradients)
    if not result['finite']:raise RuntimeError('Nonfinite joint training gradient')
    return result


def checkpoint_payload(model,optimizer,*,epoch,identity,history,best_key):
    from experiments.osram_tasr_20261018.runner_helpers import rng_state
    return dict(version=1,model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
        optimizer=copy.deepcopy(optimizer.state_dict()), rng=rng_state(), epoch=epoch,
        identity=identity, history=history, best_key=best_key,
        recovery='Full joint model/optimizer/epoch/RNG/config; no pretrained anchor')


def restore_checkpoint(model,optimizer,checkpoint,*,identity):
    from experiments.osram_tasr_20261018.runner_helpers import restore_rng
    if checkpoint['identity']!=identity:raise ValueError('Joint resume identity mismatch')
    model.load_state_dict(checkpoint['model'],strict=True)
    optimizer.load_state_dict(checkpoint['optimizer'])
    restore_rng(checkpoint['rng'])


def setup(gpu_uuid,output):
    """Use the accepted data/config, but NEVER load a pretrained model."""
    import torch
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_tasr_20261018.protocol import build_cases
    case=build_cases()['MELD']
    feature=Path('/data2/yb/paper/remote_runs_20261012/data/meld/meld_emotion_semantic_features_roberta.pkl')
    if h.sha(feature)!=case['dataset_sha256']:raise ValueError('Accepted MELD feature hash changed')
    os.environ.update(GCNET_MELD_PICKLE=str(feature),
        GCNET_DATASET_ROOT=str(Path(case['numerical_source']['source'])/'dataset'),
        GCNET_CACHE_ROOT=str(output.parent/'feature_cache/MELD'))
    # Environment must precede imports of config/data loaders.
    from experiments.osram_iemocap_b6_transfer_20261015.worker import require_replication_environment,repair_uuid_rng
    from experiments.osram_meld_write_retention_20261014.run import guard_gpu
    repair_uuid_rng(torch)
    environment=require_replication_environment(torch)
    gpu=guard_gpu(gpu_uuid)
    from gcnet_missing_m3 import train_gcnet as trainer
    from gcnet_modality_jepa.train_gcnet import get_loaders,set_random_seed
    cfg=make_config();set_random_seed(cfg.seed);torch.set_num_threads(2)
    train,validation,test,*dimensions=get_loaders('MELD-pickle','MELD-pickle','MELD-pickle',
        num_folder=1,dataset='MELD',batch_size=cfg.batch_size,num_workers=0,
        seed=cfg.seed,validation_fraction=cfg.validation_fraction,evaluation_protocol=cfg.evaluation_protocol)
    loaders=(train[0],validation[0],test[0])
    ids=[set(loader.sampler.indices) for loader in loaders]
    if ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2]:raise ValueError('Official splits overlap')
    model=fresh_model(cfg,tuple(dimensions),torch.device('cuda:0'))
    # Protected files supply exact evaluation masks/metadata ONLY. Their model
    # weights are never passed to model.load_state_dict or used for training.
    selected,references=h.read_selected_artifacts(case)
    return dict(model=model,trainer=trainer,cfg=cfg,dimensions=tuple(dimensions),
        train_loader=loaders[0],validation_loader=loaders[1],test_loader=loaders[2],
        selected=selected,references=references,case=case,environment=environment,gpu=gpu,
        split=dict(method='official-train-dev-test',train_indices=sorted(ids[0]),
            validation_indices=sorted(ids[1]),test_indices=sorted(ids[2])))


def run_epoch(bundle,optimizer,epoch,*,smoke=False):
    loader=bundle['train_loader'];loader.sampler.set_epoch(epoch)
    if smoke:
        original_loader=loader
        class FewBatches:
            sampler=original_loader.sampler
            dataset=original_loader.dataset
            def __iter__(self):return itertools.islice(iter(original_loader),5)
        loader=FewBatches()
    trainer,cfg=bundle['trainer'],bundle['cfg']
    # Audit exact actual training masks without generating another view or
    # consuming any RNG. The scoped wrapper runs only in this isolated worker.
    prepare=trainer._prepare_view;digest=hashlib.sha256();counts=[0]
    def traced(*args,**kwargs):
        view=prepare(*args,**kwargs)
        digest.update(json.dumps([str(v) for v in args[0][-1]],separators=(',',':')).encode())
        digest.update(view['availability'].detach().cpu().numpy().tobytes())
        counts[0]+=1
        return view
    trainer._prepare_view=traced
    try:
        report=trainer.train_epoch(bundle['model'],loader,optimizer,cfg,trainer._schedules(cfg,'train'),
            epoch,bundle['dimensions'],next(bundle['model'].parameters()).device)
    finally:trainer._prepare_view=prepare
    if (counts[0]!=report['optimizer_steps'] or counts[0]!=report['model_forward_count']
            or cfg.paired_history_views or 'paired_history' in report):
        raise RuntimeError('Joint run requires exactly one original random forward per optimizer step')
    report.update(original_random_mask_trace_sha256=digest.hexdigest(),random_batches=counts[0])
    return report


def evaluate(bundle,output):
    from experiments.osram_tasr_20261018.run import evaluate_all
    from experiments.osram_tasr_20261018.runner_helpers import write_json
    result=evaluate_all(bundle,output)
    result['interpretation']={'F0':'Same jointly trained checkpoint, TASR branch disabled; NOT original trained Flat',
        'adapter':'Jointly trained checkpoint with TASR enabled'}
    write_json(output/'RESULTS.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('smoke','train'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu-uuid',required=True)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if len(args.source_commit)!=40 or any(c not in '0123456789abcdef' for c in args.source_commit):
        raise ValueError('An immutable source SHA is required')
    if args.resume and args.phase!='train':raise ValueError('Only formal same-source training can resume')
    if args.output.exists() and not args.resume:raise FileExistsError('Independent new output required')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(args.output.parent).free<5*2**30:raise RuntimeError('Need5GiB free for complete joint checkpoints')
    args.output.mkdir(exist_ok=args.resume)
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid,CUDA_DEVICE_ORDER='PCI_BUS_ID',
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',GCNET_CPU_THREADS='2')
    import torch
    import fcntl
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_tasr_20261018.run import build_validation_masks,validation_scores
    from experiments.osram_tasr_20261018.protocol import validation_key
    lock=(args.output/'RUN.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tick=time.monotonic();summary=dict(status='preflight',pid=os.getpid(),phase=args.phase,
        protocol='INTERNAL DIAGNOSTIC ONLY; from-scratch joint/random-only; validation selection',
        pretrained_model_loaded=False,source_commit=args.source_commit)
    try:
        bundle=setup(args.gpu_uuid,args.output);model=bundle['model'];cfg=bundle['cfg']
        h.write_json(args.output/'config.json',asdict(cfg))
        optimizer,provenance=make_optimizer(model,cfg)
        files=('gcnet_missing_m3/osram.py','gcnet_missing_m3/tasr.py','gcnet_missing_m3/model.py',
            'gcnet_missing_m3/train_gcnet.py','gcnet_modality_jepa/gcnet_legacy_random_mask.py',
            'experiments/osram_tasr_joint_20261019/run.py','experiments/osram_tasr_20261018/run.py',
            'experiments/osram_tasr_20261018/runner_helpers.py')
        identity=dict(config=asdict(cfg),split=bundle['split'],source_commit=args.source_commit,
            source_hashes={name:h.sha(ROOT/name) for name in files},dataset_sha256=bundle['case']['dataset_sha256'],
            pretrained_model_loaded=False,evaluation_references=bundle['references'],
            checkpoint_selection='validation-random8-ACC-then-CE')
        before=h.backbone_digest(model)
        # Train-data-only numerical parity gate; this is NOT a pretrained-model replay.
        with h.isolated_rng(66+88101),torch.no_grad():
            bundle['train_loader'].sampler.set_epoch(0)
            data=bundle['trainer']._move_batch(next(iter(bundle['train_loader'])),torch.device('cuda:0'))
            view=bundle['trainer']._prepare_view(data,bundle['trainer']._schedules(cfg,'train')[.3],0,bundle['dimensions'])
            model.eval();full=h.forward(model,view)
            model.osram.osram_tasr=False;model.osram.osram_source_tracking=False
            baseline=h.forward(model,view)
            torch.testing.assert_close(full,baseline,rtol=0,atol=0)
            model.osram.osram_tasr=True;model.osram.osram_source_tracking=True
        audit=dict(steps=0,first=None,last=None)
        original_step=optimizer.step
        def audited_step(*a,**kw):
            report=gradient_report(model)
            audit['steps']+=1
            if audit['first'] is None:audit['first']=report
            audit['last']=report
            return original_step(*a,**kw)
        optimizer.step=audited_step
        summary.update(identity=identity,environment=bundle['environment'],gpu=bundle['gpu'],
            parameters=sum(p.numel() for p in model.parameters()),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            adapter_parameters=sum(p.numel() for p in model.osram.tasr.parameters()),
            optimizer_groups=provenance,zero_init_logits_bit_exact=True,
            initial_backbone_sha256=before,status='running')
        h.write_json(args.output/'SUMMARY.json',summary)
        torch.cuda.reset_peak_memory_stats()
        if args.phase=='smoke':
            metrics=run_epoch(bundle,optimizer,0,smoke=True)
            state=checkpoint_payload(model,optimizer,epoch=1,identity=identity,history=[metrics],best_key=None)
            h.atomic_torch_save(args.output/'smoke.pt',state)
            restore_checkpoint(model,optimizer,torch.load(args.output/'smoke.pt',map_location='cpu',weights_only=False),identity=identity)
            summary.update(smoke=metrics,original_backbone_changed=h.backbone_digest(model)!=before)
            if not summary['original_backbone_changed']:raise RuntimeError('Original backbone did not update')
            for name in ('encoder','writer_key','writer_value','query','flat','classifier','tasr'):
                if not audit['last']['groups'][name]['nonzero']:raise RuntimeError('No real joint gradient: '+name)
        else:
            masks=build_validation_masks(bundle)
            summary['validation_mask_hashes']={r:hashlib.sha256(b''.join(x['availability'].numpy().tobytes() for x in rows)).hexdigest()
                                               for r,rows in masks.items()}
            start,history,best_key=0,[],None
            if args.resume:
                state=torch.load(args.output/'last.pt',map_location='cpu',weights_only=False)
                restore_checkpoint(model,optimizer,state,identity=identity)
                start,history,best_key=state['epoch'],state['history'],state['best_key']
            for epoch in range(start,cfg.epochs):
                metrics=run_epoch(bundle,optimizer,epoch)
                validation=validation_scores(bundle,masks);key=validation_key(validation)
                metrics.update(epoch=epoch+1,validation=validation,validation_key=key,gradient_audit=copy.deepcopy(audit))
                history.append(metrics);improved=best_key is None or tuple(key)>tuple(best_key)
                if improved:best_key=key
                state=checkpoint_payload(model,optimizer,epoch=epoch+1,identity=identity,history=history,best_key=best_key)
                if improved:h.atomic_torch_save(args.output/'best.pt',state)
                h.atomic_torch_save(args.output/'last.pt',state)
                h.write_json(args.output/'history.json',history)
                summary.update(epoch=epoch+1,validation_best_key=best_key,gradient_audit=audit,
                    seconds=time.monotonic()-tick,peak_memory_mib=torch.cuda.max_memory_allocated()/2**20)
                h.write_json(args.output/'SUMMARY.json',summary)
                print(json.dumps(dict(event='joint_epoch_complete',epoch=epoch+1,task_CE=metrics['classification_loss'],
                    validation=key,best=best_key)),flush=True)
            state=torch.load(args.output/'best.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(state['model'],strict=True)
            summary.update(selected_epoch=state['epoch'],evaluation=evaluate(bundle,args.output/'evaluation'))
        if any(h.sha(ROOT/name)!=digest for name,digest in identity['source_hashes'].items()):
            raise RuntimeError('Immutable joint source changed during execution')
        summary.update(status='complete',seconds=time.monotonic()-tick,gradient_audit=audit,
            peak_memory_mib=torch.cuda.max_memory_allocated()/2**20)
    except BaseException as error:
        summary.update(status='failed',error=repr(error),seconds=time.monotonic()-tick)
        raise
    finally:
        h.write_json(args.output/'SUMMARY.json',summary);lock.close()


if __name__=='__main__':main()
