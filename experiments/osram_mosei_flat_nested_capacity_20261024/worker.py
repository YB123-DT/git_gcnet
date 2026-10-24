"""Finite small-Flat / existing Nested capacity comparison, unchanged s0 task."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import time

from experiments.osram_mosei_v3_capacity_20261022 import worker as base

MODES = {'f128':(128,'none'),'f384':(384,'none'),'f768':(768,'none'),
    'f256_n64':(256,'nested_gnn_rooted_evidence'),'f256_n128':(256,'nested_dim128'),
    'f512_n128':(512,'nested_dim128'),'f512_n256':(512,'mosei_capacity_nested256')}
RATES = base.RATES
ROOT = Path(__file__).resolve().parents[2]


def candidate_config(cfg, mode):
    if mode not in MODES:
        raise ValueError('Only seven new approved arms; Flat256 already runs separately')
    checked=base.candidate_config(cfg,'small_flat')
    width,method=MODES[mode]
    return replace(checked,osram_adapter_hidden_dim=width,osram_meaningful_block=method)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=tuple(MODES),required=True)
    p.add_argument('--phase',choices=('train',),required=True)
    p.add_argument('--seed',choices=(66,),type=int,default=66)
    p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--source-commit',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid,OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    import fcntl
    import numpy as np
    import torch
    from experiments.osram_mosei_gcnet_router_20261020 import worker as old
    from experiments.osram_text_path_router_joint_mosi_20261019 import run as task
    from experiments.osram_text_path_router_joint_20261019.run import restore
    from experiments.osram_mosi_gcnet_testbest_20261020.worker import persist_epoch,validate_selected,publish_selected
    from experiments.osram_source_relation_attention_20261021.worker import check_training_parity
    from experiments.osram_mosei_nested_gcnet_3seed_20261021.worker import validate_nested_gradients
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from gcnet_modality_jepa.train_gcnet import reset_training_stochasticity
    if args.output.exists() and not args.resume:
        raise FileExistsError('Fresh independent output required')
    args.output.mkdir(parents=True,exist_ok=args.resume)
    lock=(args.output/'RUN.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=args.output/'SUMMARY.json';tick=time.monotonic();hooks=[]
    if path.exists() and json.loads(path.read_text()).get('status')=='complete':
        raise ValueError('Completed run cannot repeat')
    summary=dict(status='preflight',dataset='CMUMOSEI',mode=args.mode,seed=66,phase='train',
        source_commit=args.source_commit,pid=os.getpid(),label='INTERNAL TEST-ORACLE SCREEN')
    try:
        torch.set_num_threads(1);h.write_json(path,summary)
        bundle=old.setup(args.gpu_uuid,args.output,66)
        large=bundle.pop('model');cfg=candidate_config(bundle['cfg'],args.mode)
        plain_cfg=replace(cfg,osram_meaningful_block='none')
        plain=old.fresh_backbone(plain_cfg,bundle['dimensions'],torch.device('cuda:0'))
        nested=cfg.osram_meaningful_block!='none'
        model=old.fresh_backbone(cfg,bundle['dimensions'],torch.device('cuda:0')) if nested else plain
        excluded=('osram.emotion_adapter.','osram.meaningful_block.')
        if base.digest_without(large,excluded)!=base.digest_without(model,excluded):
            raise RuntimeError('Width change altered untouched original initialization')
        plain_initial=h.backbone_digest(plain)
        if base.digest_without(model,('osram.meaningful_block.',))!=plain_initial:
            raise RuntimeError('Nested altered same-width Flat initialization')
        del large
        bundle.update(cfg=cfg,model=model)
        prior=json.loads((base.REFERENCE/'SUMMARY.json').read_text())
        masks=task.build_masks(bundle,'test')
        hashes={r:hashlib.sha256(b''.join(v['availability'].numpy().tobytes() for v in bank)).hexdigest()
                for r,bank in masks.items()}
        if hashes!=prior['identity']['mask_hashes'].get('test',prior['identity']['mask_hashes']):
            raise ValueError('Exact accepted test masks differ')
        for key in ('split','seed_protocol','data_files'):
            if bundle[key]!=prior['identity'][key]:raise ValueError('Original reference differs: '+key)
        if nested:
            _,a=task.evaluate(bundle,plain.eval(),'flat','test',masks['0.3'],max_batches=1)
            _,b=task.evaluate(bundle,model.eval(),'flat','test',masks['0.3'],max_batches=1)
            for key in ('prediction','labels','availability'):np.testing.assert_array_equal(a[key],b[key])
            del plain,a,b
        sources={str(f.relative_to(ROOT)):h.sha(f) for folder in ('gcnet_missing_m3','gcnet_modality_jepa')
                 for f in (ROOT/folder).rglob('*.py')}
        for relative in ('experiments/osram_mosei_flat_nested_capacity_20261024/worker.py',
            'experiments/osram_text_path_router_joint_mosi_20261019/run.py',
            'experiments/osram_text_path_router_joint_20261019/run.py',
            'experiments/osram_mosi_gcnet_testbest_20261020/worker.py',
            'experiments/osram_mosei_gcnet_router_20261020/worker.py',
            'experiments/osram_tasr_20261018/runner_helpers.py'):
            sources[relative]=h.sha(ROOT/relative)
        identity=dict(source_commit=args.source_commit,source_hashes=sources,config=asdict(cfg),
            dataset='CMUMOSEI',seed=66,mode=args.mode,split=bundle['split'],seed_protocol=bundle['seed_protocol'],
            data_files=bundle['data_files'],mask_hashes={'test':hashes},environment=bundle['environment'],
            initialization='fresh; no trained weights loaded',same_width_initial_sha256=plain_initial,
            reference=dict(path=str(base.REFERENCE),summary_sha256=h.sha(base.REFERENCE/'SUMMARY.json')),
            selection='per-rate TEST-best W-F1; same-call ACC/MSE; strict improvement/earliest tie')
        optimizer=task.make_optimizer(model,cfg,'flat')
        block=model.osram.meaningful_block if nested else None
        counts=dict(total=sum(p.numel() for p in model.parameters()),
            flat_adapter=sum(p.numel() for p in model.osram.emotion_adapter.parameters()),
            nested=sum(p.numel() for p in block.parameters()) if nested else 0)
        summary.update(status='running',config=asdict(cfg),identity=identity,environment=bundle['environment'],
            gpu=bundle['gpu'],parameters=counts,initial_backbone_sha256=plain_initial,
            zero_init_predictions_exact=True if nested else None)
        h.write_json(args.output/'config.json',asdict(cfg));torch.save({'test':masks},args.output/'fixed_masks.pt')
        h.write_json(path,summary)
        observed={};scans=[0]
        if nested:
            def capture(name):
                def hook(grad):
                    if not torch.isfinite(grad).all():raise RuntimeError('Nonfinite Nested gradient: '+name)
                    observed[name]=observed.get(name,False) or bool(grad.ne(0).any())
                return hook
            for name,param in block.core.named_parameters():hooks.append(param.register_hook(capture(name)))
        hooks.append(model.osram.register_forward_hook(lambda *unused:scans.__setitem__(0,scans[0]+1)))
        reset_training_stochasticity(66,1);torch.cuda.reset_peak_memory_stats()
        best,history,start={},[],0
        if args.resume:
            state=torch.load(args.output/'last.pt',map_location='cpu',weights_only=False)
            if len(state['rng']['cuda'])!=1:raise ValueError('Missing CUDA RNG for full resume')
            restore(model,optimizer,state,identity)
            start,history,best=state['epoch'],state['history'],state['best_by_rate']
            validate_selected(args.output,best)
        reference_history=json.loads((base.REFERENCE/'history.json').read_text())
        for epoch in range(start,60):
            scans[0]=0;observed.clear()
            row=task.train_epoch(bundle,model,optimizer,epoch,'flat')
            check_training_parity(row,reference_history[epoch])
            if scans[0]!=row['optimizer_steps']:raise RuntimeError('One CHM scan per batch required')
            if nested and epoch==0:validate_nested_gradients(observed)
            row.update(actual_memory_scans=scans[0],nested_gradient_paths=dict(observed))
            evaluated={r:task.evaluate(bundle,model,'flat','test',masks[r]) for r in RATES}
            row['test']={r:x[0] for r,x in evaluated.items()};history.append(row)
            improved=persist_epoch(args.output,model,optimizer,epoch=epoch+1,identity=identity,
                history=history,best=best,evaluated=evaluated)
            summary.update(epoch=epoch+1,best_by_rate=best,seconds=time.monotonic()-tick,
                peak_memory_mib=torch.cuda.max_memory_allocated()/2**20)
            h.write_json(path,summary)
            print(json.dumps(dict(event='epoch',mode=args.mode,epoch=epoch+1,task_MSE=row['task_MSE'],
                improved_rates=improved)),flush=True)
        summary['evaluation']=publish_selected(args.output,best)
        summary['aggregate']=task.aggregate(summary['evaluation'])
        for relative,digest in sources.items():
            if h.sha(ROOT/relative)!=digest:raise RuntimeError('Immutable source changed: '+relative)
        summary.update(status='complete',seconds=time.monotonic()-tick)
    except BaseException as error:
        summary.update(status='failed',error=repr(error),seconds=time.monotonic()-tick);raise
    finally:
        for hook in hooks:hook.remove()
        h.write_json(path,summary)


if __name__=='__main__':main()
