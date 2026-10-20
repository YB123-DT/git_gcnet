"""MOSI/MOSEI frozen12-source readout audit; one fixed original Flat per dataset."""
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
PATTERNS=('A','T','V','AT','AV','TV','ATV')


def reference_root(dataset):
    roots={
        'CMUMOSI':'/data2/yb/remote_experiments/osram_mosi_gcnet_testbest_20261020/execution_v2/runs/MOSI/seed_66/flat',
        'CMUMOSEI':'/data2/yb/remote_experiments/osram_mosei_gcnet_router_20261020/execution_v1/runs/CMUMOSEI/seed_66/flat'}
    if dataset not in roots: raise ValueError('Only MOSI/MOSEI accepted scalar anchors')
    return Path(roots[dataset])


def conditions():
    return [('random',r,None) for r in RATES]+[('persistent',p,p) for p in PATTERNS]


def validate_anchor(summary,dataset):
    epochs=150 if dataset=='CMUMOSI' else 60
    if (summary.get('status')!='complete' or summary.get('phase')!='train'
            or summary.get('mode')!='flat' or summary.get('dataset')!=dataset
            or summary.get('seed')!=66 or summary.get('epoch')!=epochs):
        raise ValueError('Require completed original Flat seed66 anchor')
    cfg=summary['identity']['config']
    required=dict(dataset=dataset,seed=66,epochs=epochs,missing_mask_protocol='gcnet-random-mask-v1',
        checkpoint_selection='test-oracle-per-rate',checkpoint_selection_metric='weighted_f1',
        mosi_task_mode='regression',task_regression_loss='mse',training_objective='emotion-only',
        osram_readout_fusion='flat',osram_output_dim=1600,osram_bidirectional=False,
        osram_num_heads=8,osram_key_dim=64,osram_value_dim=64,latent_dim=256,
        osram_write_step=.6,completion_path='none',osram_tasr=False)
    if any(cfg.get(k)!=v for k,v in required.items()):
        raise ValueError('Require original scalar cfg84 Flat and GCNet missing')
    return cfg


def setup(dataset,gpu_uuid,output):
    import torch
    import sklearn
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_meld_write_retention_20261014.run import guard_gpu
    from experiments.osram_mosi_gcnet_router_3seed_20261019.report import CORE
    if (torch.__version__,np.__version__,sklearn.__version__)!=('2.2.2+cu121','1.24.3','1.3.2'):
        raise ValueError('Use original MOSI/MOSEI s0 environment')
    gpu=guard_gpu(gpu_uuid)
    root=reference_root(dataset)
    summary=json.loads((root/'SUMMARY.json').read_text())
    raw=validate_anchor(summary,dataset)
    for name in CORE:
        if h.sha(ROOT/name)!=summary['identity']['source_hashes'][name]:
            raise ValueError('Original model/sampler source changed: '+name)
    if dataset=='CMUMOSI':
        from experiments.osram_mosi_gcnet_router_3seed_20261019.worker import load_manifest
        from experiments.osram_mosi_recent_history_20261018 import run as original
        manifest,proof=load_manifest(original.DATA,original.DATA_SHA)
        for name,digest in manifest['files'].items():
            if h.sha(name)!=digest: raise ValueError('Feature/label changed: '+name)
    else:
        from experiments.osram_mosei_gcnet_router_20261020.worker import load_manifest
        manifest,proof=load_manifest()
    os.environ['GCNET_DATASET_ROOT']=manifest['dataset_root']
    os.environ['GCNET_CACHE_ROOT']=str(output.parent/'feature_cache'/dataset)
    from gcnet_missing_m3 import train_gcnet as trainer
    from experiments.osram_meld_flat_source_combinations_20261013.evaluate import _build_model
    cfg=trainer.TrainConfig(**raw)
    _,_,test,*dimensions=trainer.get_loaders(*manifest['feature_roots'],dataset=dataset,
        num_folder=1,batch_size=cfg.batch_size,num_workers=0,seed=66,
        validation_fraction=cfg.validation_fraction,evaluation_protocol=cfg.evaluation_protocol)
    if json.loads(json.dumps(test[0].protocol_metadata))!=summary['identity']['split']['test']:
        raise ValueError('Original test split/order metadata differs')
    device=torch.device('cuda:0');dimensions=tuple(dimensions)
    model=_build_model(cfg,dimensions,device).eval().requires_grad_(False)
    checkpoint=root/'best_miss_0p3.pt'
    expected=summary['best_by_rate']['0.3']['checkpoint_sha256']
    if h.sha(checkpoint)!=expected: raise ValueError('Protected checkpoint SHA differs')
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if json.loads(json.dumps(state['identity']))!=summary['identity']: raise ValueError('Checkpoint provenance differs')
    model.load_state_dict(state['model'],strict=True)
    del state
    selected={}
    for rate in RATES:
        f=root/'evaluation'/('random_'+rate.replace('.','p')+'.npz')
        if h.sha(f)!=summary['evaluation'][rate]['output_sha256']:
            raise ValueError('Exact selected prediction artifact changed')
        with np.load(f,allow_pickle=False) as z: selected[rate]={k:z[k] for k in z.files}
    bundle=dict(model=model,trainer=trainer,cfg=cfg,dimensions=dimensions,test_loader=test[0],selected=selected)
    masks=torch.load(root/'fixed_masks.pt',map_location='cpu',weights_only=False)
    from experiments.osram_text_path_router_joint_mosi_20261019 import run as scalar
    _,arrays=scalar.evaluate(bundle,model,'flat','test',masks['test']['0.3'])
    for name in ('labels','speakers','conversation_ids','utterance_indices','availability','prediction'):
        np.testing.assert_array_equal(arrays[name],selected['0.3'][name],err_msg='Exact anchor replay: '+name)
    bundle.update(case=dict(dataset=dataset,seed=66,fold=1,checkpoint=str(checkpoint),
        checkpoint_sha256=expected,selected_epoch=summary['selected_epoch_by_rate']['0.3'],
        source_summary=str(root/'SUMMARY.json'),source_summary_sha256=h.sha(root/'SUMMARY.json')),
        anchor_parity=dict(exact_predictions=True,exact_labels_masks_order=True,strict_state_dict=True),
        environment=dict(python=sys.executable,torch=torch.__version__,numpy=np.__version__,sklearn=sklearn.__version__),
        gpu=gpu,data_manifest=proof)
    model.osram.osram_source_tracking=True
    model.osram.osram_source_audit=True
    return bundle


def collect(bundle,*,saved,pattern,output,check_selected=False,max_batches=None):
    import torch
    from experiments.osram_12_source_evidence_audit_20261019.run import capture_flat,require_same_rng
    from experiments.osram_12_source_evidence_audit_20261019.readout import replay_readouts
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_text_path_router_joint_mosi_20261019.run import metadata
    from experiments.osram_persistent_patterns_20261018.evaluate import fixed_availability
    model,trainer,loader=bundle['model'],bundle['trainer'],bundle['test_loader']
    loader.sampler.set_epoch(0)
    parts,diagnostics,cursor,scans={},[],0,0
    tick=time.monotonic()
    with torch.no_grad():
        for index,raw in enumerate(loader):
            if max_batches is not None and index>=max_batches: break
            data=trainer._move_batch(raw,next(model.parameters()).device)
            valid=data[7].bool();meta=metadata(data);n=len(meta['labels'])
            for name,value in meta.items():
                np.testing.assert_array_equal(value,saved[name][cursor:cursor+n])
            if pattern is None:
                av=torch.zeros((*valid.shape,3),device=valid.device)
                av[valid]=torch.as_tensor(saved['availability'][cursor:cursor+n],device=valid.device,dtype=av.dtype)
                av=av.transpose(0,1)
            else: av=fixed_availability(data[7],pattern)
            view=trainer._prepare_view_from_primary_masks(data,av,av,bundle['dimensions'])
            if not torch.equal(view['availability'],av): raise ValueError('Availability changed')
            rng=h.rng_state()
            with capture_flat(model) as captured: logits=h.forward(model,view)
            scans+=1
            trace=model.osram.last_source_trace
            sources=torch.cat((trace['source_base'].unsqueeze(2),trace['source_gap']),2)
            replay=replay_readouts(model.osram,model.smax_fc,captured['local'],captured['emotion_input'],
                sources,data[7],av,logits)
            require_same_rng(rng,h.rng_state())
            fd=model.osram.num_heads*model.osram.value_dim
            contexts=captured['emotion_input'][...,model.osram.latent_dim:].reshape(*av.shape[:2],4,2*fd)
            flat=lambda v:v.transpose(0,1)[valid].detach().cpu().numpy()
            arrays=dict(logits_full=flat(logits),source_reads=flat(sources),original_reads=flat(contexts[...,:fd]),
                trajectory=flat(trace['trajectory']),state_source_norms=flat(trace['state_source_norms']),
                state_joint_norm=flat(trace['state_joint_norm']),availability=flat(av),**{k:flat(v) for k,v in replay.items()})
            for name,value in {**meta,**arrays}.items(): parts.setdefault(name,[]).append(value)
            diagnostics.append(dict(valid=n,**trace['diagnostics']));cursor+=n
    if max_batches is None and cursor!=len(saved['labels']): raise ValueError('Incomplete test cohort')
    arrays={k:np.concatenate(v) for k,v in parts.items()}
    if check_selected:
        np.testing.assert_array_equal(arrays['logits_full'],saved['prediction'][:cursor])
    temporary=output.with_suffix('.npz.tmp')
    with temporary.open('wb') as stream: np.savez_compressed(stream,**arrays)
    temporary.replace(output)
    return arrays,dict(scans=scans,diagnostics=diagnostics,seconds=time.monotonic()-tick,
        availability_sha256=hashlib.sha256(arrays['availability'].tobytes()).hexdigest(),
        selected_anchor_predictions_exact=bool(check_selected),no_rng_change_during_forward_and_readouts=True,
        original_readout_reproduction_max_abs_error=0.,
        source_sum_max_abs_error=float(np.abs(arrays['source_reads'].sum(-2)-arrays['original_reads']).max()),
        output_sha256=h.sha(output))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=('CMUMOSI','CMUMOSEI'),required=True)
    parser.add_argument('--gpu-uuid',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-commit',required=True)
    args=parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    import fcntl
    import torch
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from .analysis import analyze_arrays
    from .report import summarize_run
    args.output.mkdir(parents=True,exist_ok=True)
    lock=(args.output/'RUN.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=args.output/'SUMMARY.json'
    if path.exists(): raise FileExistsError('Fresh independent audit output required')
    tick=time.monotonic()
    report=dict(status='running',dataset=args.dataset,source_commit=args.source_commit,no_training=True,
        conditions={},pid=os.getpid(),protocol='INTERNAL DIAGNOSTIC ONLY',error=None)
    h.write_json(path,report)
    try:
        torch.set_num_threads(1)
        bundle=setup(args.dataset,args.gpu_uuid,args.output)
        model=bundle['model'];digest=h.backbone_digest(model)
        report.update(anchor=bundle['case'],anchor_parity=bundle['anchor_parity'],environment=bundle['environment'],
            gpu=bundle['gpu'],data_manifest=bundle['data_manifest'],backbone_sha256=digest,
            source_files={str(p.relative_to(ROOT)):h.sha(p) for p in (
                Path(__file__),Path(__file__).with_name('analysis.py'),Path(__file__).with_name('report.py'),
                ROOT/'gcnet_missing_m3/osram.py',ROOT/'gcnet_missing_m3/tasr.py',
                ROOT/'experiments/osram_12_source_evidence_audit_20261019/readout.py')})
        h.write_json(path,report);torch.cuda.reset_peak_memory_stats()
        # Real first-batch invariance test: not training, no optimizer/backward.
        smoke_arrays,smoke_checks=collect(bundle,saved=bundle['selected']['0.3'],pattern=None,
            output=args.output/'SMOKE.npz',check_selected=True,max_batches=1)
        analyze_arrays(smoke_arrays,dataset=args.dataset)
        report['smoke']=dict(checks=smoke_checks,no_training=True,finite=True)
        for group,name,pattern in conditions():
            key=group+'_'+name.replace('.','p');artifact=args.output/(key+'.npz')
            saved=bundle['selected'][name if group=='random' else '0.3']
            arrays,checks=collect(bundle,saved=saved,pattern=pattern,output=artifact,
                check_selected=(group=='random' and name=='0.3'))
            statistics=analyze_arrays(arrays,dataset=args.dataset)
            result=dict(source_commit=args.source_commit,anchor_sha256=bundle['case']['checkpoint_sha256'],
                group=group,condition=name,checks=checks,statistics=statistics)
            target=args.output/(key+'.json');h.write_json(target,result)
            report['conditions'][key]=dict(path=target.name,group=group,condition=name,
                full_metrics=statistics['full_metrics'],checks=checks)
            h.write_json(path,report)
            print(json.dumps(dict(dataset=args.dataset,condition=key,full_metrics=statistics['full_metrics'],
                seconds=time.monotonic()-tick)),flush=True)
        if h.backbone_digest(model)!=digest or any(p.requires_grad or p.grad is not None for p in model.parameters()):
            raise RuntimeError('Frozen original model changed')
        for name,sha in report['source_files'].items():
            if h.sha(ROOT/name)!=sha: raise ValueError('Immutable source changed')
        report.update(status='complete',backbone_unchanged=True,seconds=time.monotonic()-tick,
            peak_memory_mib=torch.cuda.max_memory_allocated()/2**20)
    except BaseException as error:
        report.update(status='failed',error=repr(error),seconds=time.monotonic()-tick)
        h.write_json(path,report);raise
    h.write_json(path,report)
    summarize_run(args.output)


if __name__=='__main__': main()
