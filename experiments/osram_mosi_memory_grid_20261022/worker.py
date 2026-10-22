"""Original Flat only: finite write-strength / per-head K/V dimension sweep."""
import argparse
from dataclasses import asdict,replace
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from .config import MODES,RATES,candidate_config
from experiments.osram_source_relation_attention_20261021.worker import (
    REFERENCES,check_training_parity,set_learning_rates,settings)

ROOT=Path(__file__).resolve().parents[2]


def align_fresh_initialization(original,target,*,seed=66):
    """Match common tensors to the same UNTRAINED same-seed Flat initialization.

    Changing K/V shapes otherwise shifts constructor RNG for unrelated layers.
    Shape-changed tensors retain their fresh seeded initialization. No trained
    checkpoint weights enter either model.
    """
    import torch
    source,destination=original.state_dict(),target.state_dict()
    if source.keys()!=destination.keys():
        raise ValueError('Memory sweep must not add or remove model parameter paths')
    copied,changed=[],[]
    for name,value in destination.items():
        if source[name].shape==value.shape:
            destination[name]=source[name].detach().clone()
            copied.append(name)
        else:
            changed.append(name)
    target.load_state_dict(destination,strict=True)
    actual=target.state_dict()
    if not all(torch.equal(actual[name],source[name]) for name in copied):
        raise RuntimeError('Common original initialization changed')
    return dict(same_shape_parameters_exact=True,copied=copied,shape_changed=changed,
                source=f'fresh untrained original seed{seed} Flat; not checkpoint weights')


def source_catalog(prior):
    paths=set(prior)
    for package in ('gcnet_missing_m3','gcnet_modality_jepa'):
        paths.update(str(p.relative_to(ROOT)) for p in (ROOT/package).rglob('*.py'))
    paths.update('experiments/osram_mosi_memory_grid_20261022/'+n
                 for n in ('config.py','worker.py','run.py','report.py'))
    paths.update(('experiments/osram_mosi_gcnet_router_3seed_20261019/worker.py',
        'experiments/osram_mosi_gcnet_testbest_20261020/worker.py',
        'experiments/osram_mosi_v3_capacity_20261022/worker.py',
        'experiments/osram_mosi_v3_capacity_20261022/run.py',
        'experiments/osram_text_path_router_3seed_20261019/resource_gate.py',
        'experiments/osram_source_relation_attention_20261021/worker.py',
        'experiments/osram_source_relation_attention_20261021/run.py',
        'experiments/osram_text_path_router_joint_mosi_20261019/run.py',
        'experiments/osram_text_path_router_joint_20261019/run.py',
        'experiments/osram_tasr_20261018/runner_helpers.py'))
    return sorted(paths)


def default_reference(seed):
    if seed!=66:
        raise ValueError('The original grid only admits seed66')
    return REFERENCES['CMUMOSI']


def main(argv=None,*,allowed_seeds=(66,),allowed_modes=MODES,
         config_factory=candidate_config,reference_factory=default_reference,
         source_catalog_factory=source_catalog):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=allowed_modes,required=True)
    p.add_argument('--phase',choices=('smoke','train'),required=True)
    p.add_argument('--seed',type=int,choices=allowed_seeds,
                   default=66 if 66 in allowed_seeds else None,required=66 not in allowed_seeds)
    p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--source-commit',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args(argv)
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    import fcntl
    import torch
    from gcnet_missing_m3.train_gcnet import TrainConfig
    from gcnet_modality_jepa.train_gcnet import reset_training_stochasticity
    from experiments.osram_mosi_gcnet_router_3seed_20261019 import worker as original
    from experiments.osram_mosi_gcnet_testbest_20261020 import worker as selected
    from experiments.osram_text_path_router_joint_mosi_20261019 import run as scalar
    from experiments.osram_text_path_router_joint_20261019.run import restore
    from experiments.osram_mosei_v3_capacity_20261022.worker import same_state
    from experiments.osram_tasr_20261018 import runner_helpers as h
    if args.output.exists() and not args.resume:
        raise FileExistsError('Fresh independent output required')
    if args.resume and args.phase!='train':
        raise ValueError('Only train supports complete-state resume')
    args.output.mkdir(parents=True,exist_ok=args.resume)
    lock=(args.output/'RUN.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=args.output/'SUMMARY.json'
    previous=json.loads(path.read_text()) if path.exists() else {}
    if previous.get('status')=='complete':
        raise ValueError('Do not repeat a completed run')
    start_clock=time.monotonic()
    summary=dict(status='preflight',dataset='CMUMOSI',seed=args.seed,mode=args.mode,phase=args.phase,
        source_commit=args.source_commit,pid=os.getpid(),label='INTERNAL TEST-ORACLE SCREEN')
    handle=None
    try:
        torch.set_num_threads(1)
        h.write_json(path,summary)
        bundle=original.setup(args.gpu_uuid,args.output,args.seed)
        reference=reference_factory(args.seed)
        prior=json.loads((reference/'SUMMARY.json').read_text())
        base_cfg=replace(selected.testbest_config(bundle['cfg']),learning_rate=3e-4)
        canonical=lambda cfg: json.loads(json.dumps(asdict(cfg)))
        if prior['status']!='complete' or prior['epoch']!=150 or canonical(base_cfg)!=canonical(TrainConfig(**prior['identity']['config'])):
            raise ValueError('Accepted original MOSI150 reference/config required')
        if bundle['initial_backbone_sha256']!=prior['initial_backbone_sha256']:
            raise ValueError('Current original fresh initialization differs')
        for k in ('split','seed_protocol','data_files'):
            if bundle[k]!=prior['identity'][k]:
                raise ValueError('Accepted MOSI reference differs: '+k)
        cfg=config_factory(base_cfg,args.mode)
        fresh=bundle.pop('model')
        model=original.fresh_backbone(cfg,bundle['dimensions'],torch.device('cuda:0'))
        proof=align_fresh_initialization(fresh,model,seed=args.seed)
        if cfg.osram_key_dim==64 and h.backbone_digest(model)!=bundle['initial_backbone_sha256']:
            raise ValueError('Write-strength sweep changed original initialization')
        del fresh
        bundle.update(cfg=cfg,model=model)
        masks={'test':original.build_masks(bundle,'test')}
        hashes={'test':{r:hashlib.sha256(b''.join(x['availability'].numpy().tobytes() for x in bank)).hexdigest()
                         for r,bank in masks['test'].items()}}
        if hashes['test']!=prior['identity']['mask_hashes'].get('test',prior['identity']['mask_hashes']):
            raise ValueError('Original exact GCNet test masks differ')
        reference_history=json.loads((reference/'history.json').read_text())
        if len(reference_history)!=150:
            raise ValueError('Require original complete training Mask/LR ledger')
        identity=dict(source_commit=args.source_commit,
            source_hashes={name:h.sha(ROOT/name) for name in source_catalog_factory(prior['identity']['source_hashes'])},
            config=asdict(cfg),mode=args.mode,seed=args.seed,dataset='CMUMOSI',split=bundle['split'],
            seed_protocol=bundle['seed_protocol'],data_files=bundle['data_files'],
            data_manifest=bundle['data_manifest'],environment=bundle['environment'],mask_hashes=hashes,
            initialization_proof=proof,initial_model_sha256=h.backbone_digest(model),
            selection='per-rate TEST weighted_f1; strict improvement; earliest tie',
            reference=dict(path=str(reference),summary_sha256=h.sha(reference/'SUMMARY.json'),
                           history_sha256=h.sha(reference/'history.json')),learning_rates=settings('CMUMOSI'))
        optimizer=scalar.make_optimizer(model,cfg,'flat')
        if args.resume:
            stored=torch.load(args.output/'fixed_masks.pt',map_location='cpu')
            for r,bank in masks['test'].items():
                if len(bank)!=len(stored['test'][r]):
                    raise ValueError('Resume Mask batch count differs')
                for x,y in zip(bank,stored['test'][r]):
                    for k in x:
                        np.testing.assert_array_equal(x[k],y[k])
        else:
            torch.save(masks,args.output/'fixed_masks.pt')
        counts=dict(total=sum(p.numel() for p in model.parameters()),
            flat_adapter=sum(p.numel() for p in model.osram.emotion_adapter.parameters()),
            memory_state_elements_per_conversation=cfg.osram_num_heads*cfg.osram_key_dim*cfg.osram_value_dim,
            forward_read_dim=cfg.osram_num_heads*cfg.osram_value_dim,
            context_dim=model.osram.context_dim,flat_input_dim=model.osram.emotion_adapter[1].in_features)
        summary.update(status='running',identity=identity,config=asdict(cfg),parameters=counts,
            gpu=bundle['gpu'],environment=bundle['environment'],initialization_proof=proof)
        h.write_json(args.output/'config.json',asdict(cfg));h.write_json(path,summary)
        scans=[0]
        handle=model.osram.register_forward_hook(lambda *unused:scans.__setitem__(0,scans[0]+1))
        reset_training_stochasticity(args.seed,cfg.fold)
        torch.cuda.reset_peak_memory_stats()
        best,history,first={},[],0
        if args.resume:
            state=torch.load(args.output/'last.pt',map_location='cpu')
            restore(model,optimizer,state,identity)
            first,history,best=state['epoch'],state['history'],state['best_by_rate']
            if len(history)!=first or not 0<=first<=150:
                raise ValueError('Invalid full-state recovery progress')
            selected.validate_selected(args.output,best)
            for i,row in enumerate(history):
                check_training_parity(row,reference_history[i])
        initial=h.backbone_digest(model)
        limit=2 if args.phase=='smoke' else cfg.epochs
        for epoch in range(first,limit):
            scans[0]=0
            set_learning_rates(optimizer,'CMUMOSI',epoch)
            row=scalar.train_epoch(bundle,model,optimizer,epoch,'flat',max_batches=3 if args.phase=='smoke' else None)
            if scans[0]!=row['optimizer_steps']:
                raise ValueError('Require one original causal Memory scan per training batch')
            if args.phase=='train':
                check_training_parity(row,reference_history[epoch])
            train_scans=scans[0]
            evaluated={r:scalar.evaluate(bundle,model,'flat','test',masks['test'][r],
                max_batches=1 if args.phase=='smoke' else None) for r in RATES}
            row.update(actual_memory_scans=train_scans,test={r:v[0] for r,v in evaluated.items()})
            history.append(row)
            improved=selected.persist_epoch(args.output,model,optimizer,epoch=epoch+1,
                identity=identity,history=history,best=best,evaluated=evaluated)
            summary.update(epoch=epoch+1,best_by_rate=best,seconds=time.monotonic()-start_clock,
                peak_memory_mib=max(float(previous.get('peak_memory_mib',0)),torch.cuda.max_memory_allocated()/2**20))
            h.write_json(path,summary)
            print(json.dumps(dict(event='epoch',mode=args.mode,epoch=epoch+1,task_MSE=row['task_MSE'],
                improved_rates=improved,test_WF1={r:v[0]['router']['WF1'] for r,v in evaluated.items()})),flush=True)
        if args.phase=='smoke':
            if initial==h.backbone_digest(model):
                raise ValueError('Real optimizer smoke did not update model')
            state=torch.load(args.output/'last.pt',map_location='cpu')
            before_metrics,before=scalar.evaluate(bundle,model,'flat','test',masks['test']['0.3'],max_batches=1)
            restore(model,optimizer,state,identity)
            same_state(model.state_dict(),state['model']);same_state(optimizer.state_dict(),state['optimizer'])
            from experiments.osram_12_source_evidence_audit_20261019.run import require_same_rng
            require_same_rng(state['rng'],h.rng_state())
            after_metrics,after=scalar.evaluate(bundle,model,'flat','test',masks['test']['0.3'],max_batches=1)
            if before_metrics!=after_metrics:
                raise ValueError('Restored smoke metrics differ')
            for k in before:
                np.testing.assert_array_equal(before[k],after[k])
            selected.validate_selected(args.output,best)
            summary.update(smoke_only=True,smoke_optimizer_steps=sum(r['optimizer_steps'] for r in history),
                           optimizer_rng_prediction_reload_exact=True)
        else:
            summary['evaluation']=selected.publish_selected(args.output,best)
            summary['aggregate']=scalar.aggregate(summary['evaluation'])
        for name,digest in identity['source_hashes'].items():
            if h.sha(ROOT/name)!=digest:
                raise ValueError('Immutable source changed: '+name)
        summary.update(status='complete',seconds=time.monotonic()-start_clock)
    except BaseException as error:
        summary.update(status='failed',error=repr(error),seconds=time.monotonic()-start_clock)
        raise
    finally:
        if handle is not None:
            handle.remove()
        h.write_json(path,summary)


if __name__=='__main__':
    main()
