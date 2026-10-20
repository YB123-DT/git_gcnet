"""One frozen24-checkpoint audit; each batch scans CHM once, no training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
RATES=tuple(f'{i/10:.1f}' for i in range(8))
RUNS={66:Path('/data2/yb/remote_experiments/osram_mosi_source_role_direct_20261020/execution_v1/runs/CMUMOSI/seed_66/joint'),
      **{seed:Path(f'/data2/yb/remote_experiments/osram_mosi_source_role_direct_3seed_20261020/execution_v1/runs/CMUMOSI/seed_{seed}/joint') for seed in (67,68)}}


def collect(model,trainer,loader,dimensions,saved,flat):
    import torch
    from experiments.osram_12_source_evidence_audit_20261019.run import capture_flat,require_same_rng
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_text_path_router_joint_mosi_20261019.run import metadata
    from .readout import replay_neutral
    parts={};cursor=0;scans=0;max_reconstruction=0.
    loader.sampler.set_epoch(0)
    with torch.no_grad():
        for raw in loader:
            data=trainer._move_batch(raw,next(model.parameters()).device)
            valid=data[7].bool();meta=metadata(data);n=len(meta['labels'])
            for k,v in meta.items():np.testing.assert_array_equal(v,saved[k][cursor:cursor+n])
            av=torch.zeros((*valid.shape,3),device=valid.device)
            av[valid]=torch.as_tensor(saved['availability'][cursor:cursor+n],device=av.device,dtype=av.dtype)
            av=av.transpose(0,1)
            view=trainer._prepare_view_from_primary_masks(data,av,av,dimensions)
            if not torch.equal(view['availability'],av):raise ValueError('Exact GCNet availability changed')
            captured_gamma=[]
            hook=model.osram.source_role.register_forward_hook(lambda m,x,y:captured_gamma.append(y[1].detach().clone()))
            rng=h.rng_state()
            try:
                with capture_flat(model) as cap:logits=h.forward(model,view)
            finally:hook.remove()
            scans+=1
            if len(captured_gamma)!=1:raise ValueError('Expected exactly one source-role invocation')
            trace=model.osram.last_source_trace
            sources=torch.cat((trace['source_base'].unsqueeze(2),trace['source_gap']),2)
            result=replay_neutral(model.osram,model.smax_fc,cap['local'],cap['emotion_input'],
                sources,captured_gamma[0],trace['trajectory'],av,data[7],logits)
            require_same_rng(rng,h.rng_state())
            def flatten(v):return v.transpose(0,1)[valid].detach().cpu().numpy()
            values=dict(meta,availability=flatten(av),active=flatten(result['active']),gamma=flatten(captured_gamma[0]),
                prediction_learned=flatten(logits).reshape(-1),prediction_neutral=flatten(result['prediction_neutral']).reshape(-1),
                prediction_reset=flatten(result['prediction_reset']).squeeze(-1))
            np.testing.assert_array_equal(values['prediction_learned'],saved['prediction'][cursor:cursor+n].reshape(-1))
            for k,v in values.items():parts.setdefault(k,[]).append(v)
            for k,v in trace['diagnostics'].items():
                if k.endswith('sum_max_abs_error'):max_reconstruction=max(max_reconstruction,float(v))
            cursor+=n
    if cursor!=len(saved['labels']):raise ValueError('Incomplete test cohort')
    arrays={k:np.concatenate(v) for k,v in parts.items()}
    for k in ('labels','availability','conversation_ids','utterance_indices','speakers'):
        np.testing.assert_array_equal(arrays[k],flat[k])
    arrays['prediction_flat']=flat['prediction'].reshape(-1)
    return arrays,dict(memory_scans=scans,test_batches=scans,selected_predictions_exact=True,
        exact_labels_order_masks=True,flat_cohort_exact=True,no_rng_change=True,
        no_second_memory_scan=True,source_reconstruction_max_abs_error=max_reconstruction)


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
    from .analysis import analyze_arrays
    from .report import summarize_run
    if (torch.__version__,np.__version__,sklearn.__version__)!=('2.2.2+cu121','1.24.3','1.3.2'):
        raise ValueError('Use original s0 environment')
    gpu=guard_gpu(args.gpu_uuid)
    if available_gib()<32 or shutil.disk_usage(args.output.parent).free<4*2**30:
        raise RuntimeError('Insufficient RAM/disk; do not modify experiment sizes')
    args.output.mkdir(parents=True,exist_ok=False)
    lock=(args.output/'RUN.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(1);tick=time.monotonic();path=args.output/'SUMMARY.json'
    source_files=('run.py','readout.py','analysis.py','report.py')
    source_hashes={n:h.sha(Path(__file__).with_name(n)) for n in source_files}
    summary=dict(status='running',label='INTERNAL DIAGNOSTIC ONLY',no_training=True,pid=os.getpid(),
        source_commit=args.source_commit,source_hashes=source_hashes,gpu=gpu,seeds=[66,67,68],conditions={},anchors={},
        intervention='Learned versus all-active-gamma1 and12individual-gamma1; NOT gamma0 deletion',
        environment=dict(python=sys.executable,torch=torch.__version__,numpy=np.__version__,sklearn=sklearn.__version__))
    try:
        h.write_json(path,summary)
        from experiments.osram_mosi_gcnet_router_3seed_20261019.worker import load_manifest
        from experiments.osram_mosi_recent_history_20261018 import run as original
        manifest,proof=load_manifest(original.DATA,original.DATA_SHA)
        for name,digest in manifest['files'].items():
            if h.sha(name)!=digest:raise ValueError('MOSI feature/label data changed: '+name)
        os.environ['GCNET_DATASET_ROOT']=manifest['dataset_root'];os.environ['GCNET_CACHE_ROOT']=str(args.output/'feature_cache')
        from gcnet_missing_m3 import train_gcnet as trainer
        from experiments.osram_meld_flat_source_combinations_20261013.evaluate import _build_model
        from experiments.osram_coupled_source_role_20261020.worker import reference_path
        first=json.loads((RUNS[66]/'SUMMARY.json').read_text())
        cfg=trainer.TrainConfig(**first['config'])
        _,_,tests,*dims=trainer.get_loaders(*manifest['feature_roots'],dataset='CMUMOSI',num_folder=1,
            batch_size=cfg.batch_size,num_workers=0,seed=66,validation_fraction=cfg.validation_fraction,evaluation_protocol=cfg.evaluation_protocol)
        loader=tests[0];dimensions=tuple(dims);device=torch.device('cuda:0')
        summary['data_manifest']=proof;summary['data_files']=manifest['files']
        torch.cuda.reset_peak_memory_stats()
        for seed,root in RUNS.items():
            s=json.loads((root/'SUMMARY.json').read_text())
            if s['status']!='complete' or s['epoch']!=150 or s['config'].get('osram_source_role_mode')!='direct':
                raise ValueError('Require completed direct-source150 checkpoints')
            for name,digest in s['identity']['source_hashes'].items():
                if name.startswith(('gcnet_missing_m3/','gcnet_modality_jepa/')) and h.sha(ROOT/name)!=digest:
                    raise ValueError('Numerical source changed: '+name)
            if json.loads(json.dumps(loader.protocol_metadata))!=s['identity']['split']['test']:
                raise ValueError('Official MOSI test split/order differs')
            cfg=trainer.TrainConfig(**s['config']);model=_build_model(cfg,dimensions,device).eval().requires_grad_(False)
            summary['anchors'][str(seed)]=dict(run=str(root),source_commit=s['source_commit'],summary_sha256=h.sha(root/'SUMMARY.json'))
            for rate in RATES:
                key=f'seed{seed}_rate'+rate.replace('.','p');best=s['best_by_rate'][rate]
                checkpoint=root/best['checkpoint']
                if h.sha(checkpoint)!=best['checkpoint_sha256']:raise ValueError('Protected checkpoint changed')
                state=torch.load(checkpoint,map_location='cpu',weights_only=False)
                if state['identity']!=s['identity']:raise ValueError('Checkpoint provenance differs')
                model.load_state_dict(state['model'],strict=True);del state
                digest=h.backbone_digest(model)
                pred=root/'evaluation'/('random_'+rate.replace('.','p')+'.npz')
                if h.sha(pred)!=s['evaluation'][rate]['output_sha256']:raise ValueError('Selected prediction changed')
                with np.load(pred,allow_pickle=False) as z:saved={k:z[k] for k in z.files}
                fp=reference_path('CMUMOSI',seed)/'evaluation'/pred.name
                with np.load(fp,allow_pickle=False) as z:flat={k:z[k] for k in z.files}
                arrays,checks=collect(model,trainer,loader,dimensions,saved,flat)
                if h.backbone_digest(model)!=digest or any(p.requires_grad or p.grad is not None for p in model.parameters()):
                    raise ValueError('Frozen model state/gradients changed')
                npz=args.output/(key+'.npz');np.savez_compressed(npz,**arrays)
                checks.update(checkpoint_sha256=best['checkpoint_sha256'],selected_epoch=best['epoch'],
                    frozen_state_exact=True,artifact_sha256=h.sha(npz),original_flat_prediction=str(fp))
                stats=analyze_arrays(arrays)
                target=args.output/(key+'.json');h.write_json(target,dict(seed=seed,rate=rate,checks=checks,statistics=stats))
                summary['conditions'][key]=dict(seed=seed,rate=rate,path=target.name)
                summary.update(seconds=time.monotonic()-tick,peak_memory_mib=torch.cuda.max_memory_allocated()/2**20)
                h.write_json(path,summary)
                print(json.dumps(dict(seed=seed,rate=rate,checks=checks,seconds=summary['seconds'])),flush=True)
            del model
            torch.cuda.empty_cache()
        for name,digest in source_hashes.items():
            if h.sha(Path(__file__).with_name(name))!=digest:raise ValueError('Audit source changed')
        summarize_run(args.output,summary)
        summary.update(status='complete',seconds=time.monotonic()-tick)
    except BaseException as error:
        summary.update(status='failed',error=repr(error));raise
    finally:h.write_json(path,summary)


if __name__=='__main__':main()
