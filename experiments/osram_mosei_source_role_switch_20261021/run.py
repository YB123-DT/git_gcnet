"""MOSEI66 same-checkpoint learned-vs-gamma1 replay; no training or new masks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
ANCHOR=Path('/data2/yb/remote_experiments/osram_source_role_direct_cross_20261020/execution_v1/runs/CMUMOSEI/fold_1/seed_66/direct')
RATES=tuple(f'{i/10:.1f}' for i in range(8))


def validate_anchor(s):
    cfg=s.get('config',{})
    if (s.get('status')!='complete' or s.get('epoch')!=60 or s.get('seed')!=66
            or cfg.get('dataset')!='CMUMOSEI' or not cfg.get('osram_source_role')
            or cfg.get('osram_source_role_mode')!='direct' or cfg.get('missing_mask_protocol')!='gcnet-random-mask-v1'
            or cfg.get('osram_readout_fusion')!='flat' or cfg.get('osram_bidirectional')
            or cfg.get('osram_tasr') or cfg.get('osram_disable_local_skip')
            or set(s.get('best_by_rate',{}))!=set(RATES)):
        raise ValueError('Require completed original MOSEI66 Direct eight-rate anchor')


def collect(model,trainer,loader,dimensions,saved):
    import torch
    from experiments.osram_12_source_evidence_audit_20261019.run import capture_flat,require_same_rng
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_text_path_router_joint_mosi_20261019.run import metadata
    from experiments.osram_mosi_source_role_neutral_audit_20261020.readout import replay_neutral
    parts={};cursor=0;scans=0;errors={}
    loader.sampler.set_epoch(0)
    with torch.no_grad():
        for raw in loader:
            data=trainer._move_batch(raw,next(model.parameters()).device);valid=data[7].bool()
            meta=metadata(data);n=len(meta['labels'])
            for k,v in meta.items():np.testing.assert_array_equal(v,saved[k][cursor:cursor+n])
            av=torch.zeros((*valid.shape,3),device=valid.device)
            av[valid]=torch.as_tensor(saved['availability'][cursor:cursor+n],device=av.device,dtype=av.dtype)
            av=av.transpose(0,1)
            view=trainer._prepare_view_from_primary_masks(data,av,av,dimensions)
            if not torch.equal(view['availability'],av):raise ValueError('Exact availability changed')
            captures=[];hook=model.osram.source_role.register_forward_hook(lambda m,x,y:captures.append(y[1].detach().clone()))
            rng=h.rng_state()
            try:
                with capture_flat(model) as cap:prediction=h.forward(model,view)
            finally:hook.remove()
            if len(captures)!=1:raise ValueError('Expected exactly one source-role call')
            scans+=1;trace=model.osram.last_source_trace
            sources=torch.cat((trace['source_base'].unsqueeze(2),trace['source_gap']),2)
            result=replay_neutral(model.osram,model.smax_fc,cap['local'],cap['emotion_input'],sources,
                captures[0],trace['trajectory'],av,data[7],prediction,include_slot_resets=False)
            require_same_rng(rng,h.rng_state())
            flatten=lambda v:v.transpose(0,1)[valid].detach().cpu().numpy()
            safe=torch.where(result['active'][...,None],sources,torch.zeros_like(sources))
            unweighted=safe.sum(-2);weighted=(captures[0][...,None]*safe).sum(-2)
            relative=(weighted-unweighted).norm(dim=-1)/unweighted.norm(dim=-1).clamp_min(1e-12)
            values=dict(meta,availability=flatten(av),active=flatten(result['active']),gamma=flatten(captures[0]),
                trajectory=flatten(trace['trajectory']),read_relative_change=flatten(relative),
                prediction_learned=flatten(prediction).reshape(-1),
                prediction_neutral=flatten(result['prediction_neutral']).reshape(-1))
            np.testing.assert_array_equal(values['prediction_learned'],saved['prediction'][cursor:cursor+n].reshape(-1))
            for k,v in values.items():parts.setdefault(k,[]).append(v)
            for k,v in trace['diagnostics'].items():errors[k]=max(errors.get(k,0.),float(v))
            cursor+=n
    if cursor!=len(saved['labels']):raise ValueError('Incomplete test cohort')
    return {k:np.concatenate(v) for k,v in parts.items()},dict(
        memory_scans=scans,test_batches=scans,selected_predictions_exact=True,
        exact_labels_order_masks=True,no_rng_change=True,no_second_memory_scan=True,source_decomposition_errors=errors)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu-uuid',required=True)
    parser.add_argument('--source-commit',required=True)
    args=parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    import fcntl
    import shutil
    import torch
    import sklearn
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_meld_write_retention_20261014.run import guard_gpu
    from experiments.osram_text_path_router_3seed_20261019.resource_gate import available_gib
    from experiments.osram_mosi_source_role_neutral_audit_20261020.run import validate_checkpoint_identity
    from .analysis import analyze
    from .report import summarize
    if (torch.__version__,np.__version__,sklearn.__version__)!=('2.2.2+cu121','1.24.3','1.3.2'):
        raise ValueError('Use original s0 runtime')
    gpu=guard_gpu(args.gpu_uuid)
    if available_gib()<32 or shutil.disk_usage(args.output.parent).free<4*2**30:
        raise RuntimeError('Insufficient RAM/disk')
    args.output.mkdir(parents=True,exist_ok=False)
    lock=(args.output/'RUN.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tick=time.monotonic();path=args.output/'SUMMARY.json'
    sources=tuple(str(Path(__file__).with_name(n).relative_to(ROOT)) for n in ('run.py','analysis.py','report.py'))+(
        'experiments/osram_mosi_source_role_neutral_audit_20261020/readout.py',
        'experiments/osram_mosi_source_role_neutral_audit_20261020/analysis.py')
    summary=dict(status='preflight',label='INTERNAL DIAGNOSTIC ONLY',no_training=True,pid=os.getpid(),
        source_commit=args.source_commit,source_hashes={n:h.sha(ROOT/n) for n in sources},gpu=gpu,
        environment=dict(python=sys.executable,torch=torch.__version__,numpy=np.__version__,sklearn=sklearn.__version__),conditions={})
    try:
        torch.set_num_threads(1);h.write_json(path,summary)
        original=json.loads((ANCHOR/'SUMMARY.json').read_text());validate_anchor(original)
        for name,digest in original['identity']['source_hashes'].items():
            if name.startswith(('gcnet_missing_m3/','gcnet_modality_jepa/')) and h.sha(ROOT/name)!=digest:
                raise ValueError('Numerical source changed: '+name)
        from experiments.osram_mosei_gcnet_router_20261020.worker import load_manifest,build_loaders
        manifest,proof=load_manifest()
        os.environ['GCNET_DATASET_ROOT']=manifest['dataset_root'];os.environ['GCNET_CACHE_ROOT']=str(args.output/'feature_cache')
        from gcnet_missing_m3 import train_gcnet as trainer
        from experiments.osram_meld_flat_source_combinations_20261013.evaluate import _build_model
        cfg=trainer.TrainConfig(**original['config'])
        _,_,tests,*dims=build_loaders(manifest,cfg,trainer);loader=tests[0];dimensions=tuple(dims)
        if json.loads(json.dumps(loader.protocol_metadata))!=original['identity']['split']['test']:
            raise ValueError('MOSEI test split/order differs')
        model=_build_model(cfg,dimensions,torch.device('cuda:0')).eval().requires_grad_(False)
        summary.update(status='running',anchor=str(ANCHOR),anchor_summary_sha256=h.sha(ANCHOR/'SUMMARY.json'),
            anchor_source_commit=original['source_commit'],data_manifest=proof)
        torch.cuda.reset_peak_memory_stats();records={};all_arrays=[]
        for rate in RATES:
            best=original['best_by_rate'][rate];checkpoint=ANCHOR/best['checkpoint']
            if h.sha(checkpoint)!=best['checkpoint_sha256']:raise ValueError('Checkpoint hash differs')
            state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            validate_checkpoint_identity(state['identity'],original['identity'])
            model.load_state_dict(state['model'],strict=True);del state
            digest=h.backbone_digest(model)
            pred=ANCHOR/'evaluation'/('random_'+rate.replace('.','p')+'.npz')
            if h.sha(pred)!=original['evaluation'][rate]['output_sha256']:raise ValueError('Selected prediction hash differs')
            with np.load(pred,allow_pickle=False) as saved:data={k:saved[k] for k in saved.files}
            arrays,checks=collect(model,trainer,loader,dimensions,data)
            if h.backbone_digest(model)!=digest or any(p.requires_grad or p.grad is not None for p in model.parameters()):
                raise ValueError('Frozen state or gradient changed')
            stats=analyze(arrays)
            for metric in ('ACC','WF1','MSE'):
                if not np.isclose(stats['groups']['all']['learned'][metric],best['metrics']['router'][metric],rtol=1e-7,atol=1e-6):
                    raise ValueError('Selected metric mismatch: '+metric)
            key='rate_'+rate.replace('.','p');target=args.output/(key+'.npz');np.savez_compressed(target,**arrays)
            checks.update(frozen_state_exact=True,selected_epoch=best['epoch'],checkpoint=str(checkpoint),
                checkpoint_sha256=best['checkpoint_sha256'],prediction_reference=str(pred),prediction_sha256=h.sha(pred),
                saved_mask_sha256=hashlib.sha256(data['availability'].tobytes()).hexdigest(),artifact_sha256=h.sha(target))
            record=dict(rate=rate,checks=checks,statistics=stats);h.write_json(args.output/(key+'.json'),record)
            records[rate]=record;all_arrays.append(arrays)
            summary['conditions'][rate]=dict(path=key+'.json',checks=checks)
            summary.update(seconds=time.monotonic()-tick,peak_memory_mib=torch.cuda.max_memory_allocated()/2**20)
            h.write_json(path,summary);print(json.dumps(dict(rate=rate,statistics=stats['groups']['all'],checks=checks)),flush=True)
        summarize(args.output,records,all_arrays)
        for name,digest in summary['source_hashes'].items():
            if h.sha(ROOT/name)!=digest:raise ValueError('Audit source changed')
        summary.update(status='complete',seconds=time.monotonic()-tick)
    except BaseException as error:
        summary.update(status='failed',error=repr(error),seconds=time.monotonic()-tick);raise
    finally:h.write_json(path,summary)


if __name__=='__main__':main()
