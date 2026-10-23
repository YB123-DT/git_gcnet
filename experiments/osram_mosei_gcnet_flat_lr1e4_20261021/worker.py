"""MOSEI Flat constant1e-4,GCNet-v1; seed66 completion is separately launched."""
import argparse
from dataclasses import asdict, replace, fields, MISSING
import hashlib
import json
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[2]
RATES=tuple(f'{i/10:.1f}' for i in range(8))
REFERENCE=Path('/data2/yb/remote_experiments/osram_source_role_direct_cross_20261020/execution_v1/runs/CMUMOSEI/fold_1')


def low_lr_config(cfg,seed):
    if seed not in (66,67,68):raise ValueError('Only approved seeds66/67/68')
    required=dict(dataset='CMUMOSEI',fold=1,seed=66,epochs=60,batch_size=32,
        learning_rate=.001,weight_decay=1e-5,optimizer='adam',lr_schedule='constant',
        backbone_type='osram',osram_readout_fusion='flat',osram_bidirectional=False,
        osram_output_dim=1600,osram_write_step=.6,osram_num_heads=8,osram_key_dim=64,osram_value_dim=64,
        training_objective='emotion-only',checkpoint_selection='test-oracle-per-rate',
        checkpoint_selection_metric='weighted_f1',missing_mask_protocol='gcnet-random-mask-v1',
        osram_source_role=False,osram_tasr=False,osram_meaningful_block='none',
        osram_disable_local_skip=False,paired_history_views=False,completion_path='none')
    for key,value in required.items():
        if getattr(cfg,key)!=value:raise ValueError('Accepted original Flat config changed: '+key)
    return replace(cfg,seed=seed,learning_rate=1e-4)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('train',),required=True)
    parser.add_argument('--mode',choices=('flat',),required=True)
    parser.add_argument('--seed',type=int,choices=(66,67,68),required=True)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--gpu-uuid',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    import fcntl
    import numpy as np
    import torch
    from experiments.osram_mosei_gcnet_router_20261020 import worker as original
    from experiments.osram_text_path_router_joint_mosi_20261019 import run as scalar
    from experiments.osram_text_path_router_joint_20261019.run import seed_training_loader,restore
    from experiments.osram_mosi_gcnet_testbest_20261020.worker import persist_epoch,publish_selected,validate_selected
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_exemplar_mosi_20261023.worker import validate_reference_config
    tick=time.monotonic();path=args.output/'SUMMARY.json'
    if args.output.exists() and not args.resume:raise FileExistsError('Fresh output required')
    args.output.mkdir(parents=True,exist_ok=args.resume)
    lock=(args.output/'RUN.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if path.exists() and json.loads(path.read_text()).get('status')=='complete':raise ValueError('Completed run must not repeat')
    summary=dict(status='preflight',dataset='CMUMOSEI',seed=args.seed,mode='flat',phase='train',
        source_commit=args.source_commit,pid=os.getpid(),experiment_label='INTERNAL TEST-ORACLE SCREEN')
    try:
        bundle=original.setup(args.gpu_uuid,args.output,66)
        from gcnet_modality_jepa.train_gcnet import reset_training_stochasticity
        bundle['cfg']=low_lr_config(bundle['cfg'],args.seed)
        bundle['seed_protocol']=seed_training_loader(bundle,args.seed)
        del bundle['model']
        model=original.fresh_backbone(bundle['cfg'],bundle['dimensions'],torch.device('cuda:0'))
        bundle['model']=model
        masks=scalar.build_masks(bundle,'test')
        hashes={r:hashlib.sha256(b''.join(row['availability'].numpy().tobytes() for row in bank)).hexdigest() for r,bank in masks.items()}
        prior_path=REFERENCE/f'seed_{args.seed}'/'direct'
        prior=json.loads((prior_path/'SUMMARY.json').read_text())
        if prior['status']!='complete' or prior['epoch']!=60:raise ValueError('Require complete same-seed current-protocol reference')
        expected=dict(prior['config'],osram_source_role=False,osram_source_role_mode='residual',learning_rate=1e-4)
        validate_reference_config(asdict(bundle['cfg']),expected,
            {f.name:f.default for f in fields(bundle['cfg']) if f.default is not MISSING})
        initial=h.backbone_digest(model)
        if prior['initial_backbone_sha256']!=initial:raise ValueError('Same-seed original initialization differs')
        if prior['identity']['mask_hashes']!=hashes or prior['identity']['split']!=bundle['split']:
            raise ValueError('Current-protocol masks or split differ')
        if prior['identity']['seed_protocol']!=bundle['seed_protocol']:raise ValueError('Training seed protocol differs')
        reference_history=json.loads((prior_path/'history.json').read_text())
        torch.save(masks,args.output/'fixed_masks.pt')
        sources=tuple(prior['identity']['source_hashes'])+(
            'experiments/osram_mosei_gcnet_flat_lr1e4_20261021/worker.py',
            'experiments/osram_exemplar_mosi_20261023/worker.py',
            'experiments/osram_mosei_gcnet_flat_lr1e4_20261021/run.py')
        identity=dict(source_commit=args.source_commit,source_hashes={p:h.sha(ROOT/p) for p in sources},
            config=asdict(bundle['cfg']),seed=args.seed,mode='flat',split=bundle['split'],
            seed_protocol=bundle['seed_protocol'],mask_hashes=hashes,data_manifest=bundle['data_manifest'],
            dataset_manifest_sha256=bundle['data_manifest_sha256'],data_files=bundle['data_files'],
            reference=dict(path=str(prior_path),summary_sha256=h.sha(prior_path/'SUMMARY.json'),
                initialization_exact=True,evaluation_masks_exact=True,not_same_lr_as_direct=True),
            selection='per-rate TEST weighted_f1; earliest strict-best ties',
            initialization='fresh original factory; no trained checkpoint weights loaded')
        optimizer=scalar.make_optimizer(model,bundle['cfg'],'flat')
        summary.update(status='running',config=identity['config'],identity=identity,
            environment=bundle['environment'],gpu=bundle['gpu'],initial_backbone_sha256=initial,
            total_parameters=sum(p.numel() for p in model.parameters()),router_parameters=0)
        h.write_json(args.output/'config.json',identity['config']);h.write_json(path,summary)
        reset_training_stochasticity(args.seed,1);torch.cuda.reset_peak_memory_stats()
        best,history,start={},[],0
        if args.resume:
            state=torch.load(args.output/'last.pt',map_location='cpu',weights_only=False)
            restore(model,optimizer,state,identity);start,history,best=state['epoch'],state['history'],state['best_by_rate']
            validate_selected(args.output,best)
        for epoch in range(start,60):
            if [g['lr'] for g in optimizer.param_groups]!=[1e-4]:raise ValueError('Constant1e-4 required')
            row=scalar.train_epoch(bundle,model,optimizer,epoch,'flat')
            for key in ('epoch','mask_trace_sha256','optimizer_steps'):
                if row[key]!=reference_history[epoch][key]:raise ValueError('Same-seed training '+key+' differs')
            evaluated={r:scalar.evaluate(bundle,model,'flat','test',masks[r]) for r in RATES}
            row['test']={r:v[0] for r,v in evaluated.items()};history.append(row)
            improved=persist_epoch(args.output,model,optimizer,epoch=epoch+1,identity=identity,history=history,best=best,evaluated=evaluated)
            summary.update(epoch=epoch+1,best_by_rate=best,selected_epoch_by_rate={r:v['epoch'] for r,v in best.items()},
                seconds=time.monotonic()-tick,peak_memory_mib=torch.cuda.max_memory_allocated()/2**20)
            h.write_json(path,summary)
            print(json.dumps(dict(event='epoch',seed=args.seed,epoch=epoch+1,task_MSE=row['task_MSE'],
                learning_rates=row['learning_rates'],improved_rates=improved,test_WF1={r:v[0]['router']['WF1'] for r,v in evaluated.items()})),flush=True)
        summary['evaluation']=publish_selected(args.output,best)
        summary['aggregate']=scalar.aggregate(summary['evaluation'])
        for source,digest in identity['source_hashes'].items():
            if h.sha(ROOT/source)!=digest:raise RuntimeError('Immutable source changed: '+source)
        summary.update(status='complete',seconds=time.monotonic()-tick)
    except BaseException as error:
        summary.update(status='failed',error=repr(error),seconds=time.monotonic()-tick);raise
    finally:h.write_json(path,summary)


if __name__=='__main__':main()
