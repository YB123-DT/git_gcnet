"""One MOSI seed66 run on the sealed 80.747% Flat150 protocol.

100 constant-LR epochs followed by 50 complete-state lower-LR epochs.
The historical ConversationMaskSchedule is intentionally not replaced by the
later MELD GCNet sampler. Scores are internal per-rate Test-oracle diagnostics.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
HISTORICAL_COMMIT='ad211c01c5317e5520890ba9acb1ae9baf54cdd1'
SOURCE=Path('/data2/yb/remote_experiments/osram_new40_gpu0123_20261004/source_ad211c0')
DATA=Path('/data2/yb/remote_experiments/osram_meaningful20_round2_20261004/DATA.json')
REFERENCE=Path('/data1/yb/remote_experiments/osram_nested_low_lr_extend150_20261010/runs/flat_seed66')
DATA_SHA='e790eee2c850418fde9652ab85ca4a1eeb12b155c5651d91b6caea8b1e36c213'
LABEL='INTERNAL DIAGNOSTIC ONLY; per-rate Test-oracle'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False))
    tmp.replace(path)


def now():return datetime.now(timezone.utc).isoformat()


def stage_configs(source):
    required=dict(dataset='CMUMOSI',seed=66,epochs=100,learning_rate=.001,
        lr_schedule='constant',batch_size=32,optimizer='adam',weight_decay=1e-5,
        dropout=.5,projector_dropout=.1,latent_dim=256,osram_output_dim=1600,
        osram_num_heads=8,osram_key_dim=64,osram_value_dim=64,osram_write_step=.6,
        osram_bidirectional=False,osram_readout_fusion='flat',training_objective='emotion-only',
        train_rate_mode='cyclic',checkpoint_selection='test-oracle-per-rate',
        emotion_loss_mode='sample-mean',task_regression_loss='mse',
        osram_meaningful_block='none',paired_history_views=False,
        osram_relation_block=False,completion_path='none',osram_ablation='full',
        osram_emotion_ablation='full',evaluate_test=True)
    for k,v in required.items():
        if source.get(k,v)!=v:raise ValueError('Original MOSI150 protocol changed: '+k)
    if source.get('missing_mask_protocol','conversation-mask-v2')!='conversation-mask-v2':
        raise ValueError('Historical MOSI mask must not be replaced by the MELD sampler')
    return deepcopy(source),dict(deepcopy(source),epochs=150,learning_rate=.0001)


def continuation_state(original,identity):
    if (original.get('next_epoch')!=100 or len(original.get('history',[]))!=100
            or not original.get('rng') or len(original.get('best_references',{}))!=8
            or original.get('scheduler') is not None or original.get('scaler') is not None
            or not original.get('optimizer',{}).get('state')
            or not original['optimizer'].get('param_groups')
            or any(g['lr']!=.001 for g in original['optimizer']['param_groups'])):
        raise ValueError('Require complete constant100 model/Adam/RNG/BEST state')
    result=deepcopy(original)
    result['identity']=deepcopy(identity)
    for group in result['optimizer']['param_groups']:group['lr']=.0001
    return result


def summarize(metrics,reference):
    rates=[str(i/10) for i in range(8)]
    if metrics['mask_sha256']!=reference['mask_sha256']:
        raise ValueError('Evaluation masks differ from original seed66 Flat150')
    if metrics['selection_metric']!='weighted_f1':raise ValueError('MOSI requires W-F1 selection')
    rows={r:{k:100*metrics['test'][r][k] for k in ('accuracy','weighted_f1')} for r in rates}
    mean=lambda source,rs:100*sum(source['test'][r]['weighted_f1'] for r in rs)/len(rs)
    return dict(label=LABEL,per_rate=rows,mask_sha256=metrics['mask_sha256'],
        mean_wf1=mean(metrics,rates),high_wf1=mean(metrics,rates[5:]),
        mean_acc=sum(rows[r]['accuracy'] for r in rates)/8,
        high_acc=sum(rows[r]['accuracy'] for r in rates[5:])/3,
        baseline_mean_wf1=mean(reference,rates),
        delta_mean_wf1=mean(metrics,rates)-mean(reference,rates),
        baseline_high_wf1=mean(reference,rates[5:]),
        selected_epoch_by_rate=metrics.get('selected_epoch_by_rate'))


def verify_source(source):
    snapshot=read(source/'SNAPSHOT.json')
    if snapshot['code_commit']!=HISTORICAL_COMMIT:raise ValueError('Wrong historical source commit')
    hashes={n:h for n,h in snapshot['source_sha256'].items() if n.endswith('.py')
        and n.startswith(('gcnet/','gcnet_missing_m3/','gcnet_modality_jepa/'))}
    if not hashes:raise ValueError('No sealed numerical source hashes')
    for name,digest in hashes.items():
        p=(source/name).resolve()
        if not p.is_relative_to(source.resolve()) or sha(p)!=digest:
            raise ValueError('Historical source changed: '+name)
    return dict(path=str(source),commit=HISTORICAL_COMMIT,hashes=hashes)


def verify_gpu(uuid,rows,*,visible):
    matches=[int(index) for index,value in rows if value==uuid]
    if visible!=uuid or len(matches)!=1 or matches[0]==4:
        raise ValueError('GPU binding is unknown, mismatched, or forbidden physical GPU4')
    return matches[0]


def load_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--gpu-uuid',required=True)
    parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--data-manifest',type=Path,default=DATA)
    parser.add_argument('--reference',type=Path,default=REFERENCE)
    parser.add_argument('--wrapper-commit',default='unrecorded')
    parser.add_argument('--smoke',action='store_true',help='One real epoch only; separate fresh output required')
    args=parser.parse_args(argv)
    args.output.mkdir(parents=True,exist_ok=True)
    lock=(args.output/'RUN.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (args.output/'PROVENANCE.json').exists():raise ValueError('Refuse duplicate or unsafe implicit resume')
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    rows=[tuple(x.strip() for x in line.split(',')) for line in raw.splitlines()]
    physical=verify_gpu(args.gpu_uuid,rows,visible=os.environ.get('CUDA_VISIBLE_DEVICES'))
    source=verify_source(args.source)
    if sha(args.data_manifest)!=DATA_SHA:raise ValueError('MOSI data manifest changed')
    data=read(args.data_manifest)
    for name,digest in data['files'].items():
        if sha(name)!=digest:raise ValueError('MOSI feature changed: '+name)
    # The config saved with150 has150/.0001; normalize ONLY these deliberate
    # continuation fields before requiring every original baseline setting.
    base=read(args.reference/'config.json')
    if base['epochs']!=150 or base['learning_rate']!=.0001:
        raise ValueError('Need the accepted150epoch seed66 reference')
    base['epochs'],base['learning_rate']=100,.001
    cfg100dict,cfg150dict=stage_configs(base)
    os.environ['GCNET_DATASET_ROOT']=data['dataset_root']
    os.environ['GCNET_CACHE_ROOT']=str(args.output/'feature_cache')
    sys.path.insert(0,str(args.source.resolve()))
    import torch,numpy,sklearn
    if (torch.__version__!='2.2.2+cu121' or numpy.__version__!='1.24.3'
            or sklearn.__version__!='1.3.2'):
        raise ValueError('Use the archived MOSI s0 environment, not the MELD environment')
    from gcnet_missing_m3 import train_gcnet as trainer
    from gcnet_missing_m3.training_resume import TrainingState,_atomic
    torch.set_num_threads(1)
    module=load_file('mosi_recent_history_module',ROOT/'gcnet_missing_m3/recent_history.py')
    integration=load_file('mosi_recent_history_integration',HERE/'integration.py')
    integration.install(trainer,module.RecentHistoryAttentionResidual,window=4,attention_dim=128,num_heads=4)
    cfg100=trainer.TrainConfig(**cfg100dict)
    cfg150=trainer.TrainConfig(**cfg150dict)
    helper_files=[Path(__file__),HERE/'integration.py',ROOT/'gcnet_missing_m3/recent_history.py']
    helper_hashes={str(p.relative_to(ROOT)):sha(p) for p in helper_files}
    identity=dict(source=HISTORICAL_COMMIT,data=DATA_SHA,
        helpers=hashlib.sha256(json.dumps(helper_hashes,sort_keys=True).encode()).hexdigest(),
        recent_history=dict(window=4,attention_dim=128,num_heads=4),wrapper_commit=args.wrapper_commit)
    record=dict(status='running',label=LABEL,started_at=now(),pid=os.getpid(),
        physical_gpu=physical,gpu_uuid=args.gpu_uuid,source=source,helper_hashes=helper_hashes,
        environment=dict(python=sys.executable,python_version=sys.version,torch=torch.__version__,
            numpy=numpy.__version__,sklearn=sklearn.__version__),data_manifest=str(args.data_manifest),
        data_sha256=DATA_SHA,data_files_verified=len(data['files']),stage100_config=asdict(cfg100),
        stage150_config=asdict(cfg150),identity=identity,smoke=args.smoke,
        original_mask='ConversationMaskSchedule; independent Bernoulli with nonempty repair; frozen eval epoch0',
        stages=[])
    write(args.output/'PROVENANCE.json',record)
    baseline=read(args.reference/'metrics.json')
    original_train=trainer.train_epoch;signature=inspect.signature(original_train)
    trace=args.output/'stage100'/'diagnostics';expected=.001
    parity_done=False
    def observed_train(*pos,**kw):
        nonlocal parity_done
        bound=signature.bind(*pos,**kw).arguments
        model,epoch,opt=bound['model'],bound['epoch'],bound['optimizer']
        if any(g['lr']!=expected for g in opt.param_groups):raise ValueError('Unexpected actual optimizer LR')
        branch=model.osram.recent_history
        optimizer_ids={id(p) for g in opt.param_groups for p in g['params']}
        if any(id(p) not in optimizer_ids for p in branch.parameters()):raise ValueError('Branch missing from optimizer')
        stats=dict(epoch=epoch+1,rates=[float(g['lr']) for g in opt.param_groups],steps=[])
        saved_forward=model.forward;original_clip=torch.nn.utils.clip_grad_norm_
        def forward(*inputs,**keywords):
            nonlocal parity_done
            if not parity_done:
                # Both calls use identical existing model Dropout streams;
                # restore them afterward so this check consumes no training RNG.
                cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all()
                try:
                    integration.set_enabled(model,False)
                    with torch.no_grad():off=saved_forward(*inputs,**keywords)
                    off_cpu=torch.get_rng_state();off_cuda=torch.cuda.get_rng_state_all()
                    torch.set_rng_state(cpu);torch.cuda.set_rng_state_all(cuda)
                    integration.set_enabled(model,True)
                    with torch.no_grad():on=saved_forward(*inputs,**keywords)
                    if (not torch.equal(off_cpu,torch.get_rng_state())
                            or any(not torch.equal(a,b) for a,b in zip(off_cuda,torch.cuda.get_rng_state_all()))):
                        raise ValueError('New branch consumed original model RNG')
                    def check(a,b):
                        if isinstance(a,torch.Tensor):
                            if not torch.equal(a,b):raise ValueError('Zero-init real-batch baseline parity failed')
                        elif isinstance(a,(tuple,list)):
                            for x,y in zip(a,b):check(x,y)
                        elif isinstance(a,dict):
                            for key in a:
                                if key in b:check(a[key],b[key])
                    check(off,on)
                    write(args.output/'ZERO_INIT_PARITY.json',dict(passed=True,real_batch=True,
                        rng_equal=True,training=model.training))
                    parity_done=True
                finally:
                    torch.set_rng_state(cpu);torch.cuda.set_rng_state_all(cuda)
                    integration.set_enabled(model,True)
            return saved_forward(*inputs,**keywords)
        def clip(parameters,max_norm,*a,**k):
            grads=[p.grad for p in branch.parameters() if p.grad is not None]
            if not grads or any(not torch.isfinite(g).all() for g in grads):
                raise ValueError('Missing or nonfinite recent-history gradients')
            stats['steps'].append(dict(branch_gradient_tensors=len(grads),
                branch_gradient_norm=float(torch.stack([g.norm() for g in grads]).norm().detach().cpu())))
            return original_clip(parameters,max_norm,*a,**k)
        model.forward=forward;torch.nn.utils.clip_grad_norm_=clip
        try:
            result=original_train(*pos,**kw)
            stats['diagnostics']={k:float(v.detach().cpu()) if isinstance(v,torch.Tensor) else v
                for k,v in getattr(branch,'last_diagnostics',{}).items()}
            stats['parameters']=sum(p.numel() for p in model.parameters())
            stats['added_parameters']=sum(p.numel() for p in branch.parameters())
            write(trace/f'epoch_{epoch+1:03d}.json',stats)
            write(args.output/'PROGRESS.json',dict(epoch=epoch+1,total=1 if args.smoke else 150,
                updated_at=now(),expected_lr=expected,stats=stats))
            return result
        finally:
            model.forward=saved_forward;torch.nn.utils.clip_grad_norm_=original_clip
    trainer.train_epoch=observed_train
    start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    try:
        stage100=args.output/'stage100'
        firstcfg=replace(cfg100,epochs=1) if args.smoke else cfg100
        id100=dict(identity,config=hashlib.sha256(json.dumps(asdict(firstcfg),sort_keys=True).encode()).hexdigest())
        metrics=trainer.run_experiment(firstcfg,*data['feature_roots'],str(stage100),
            training_state=TrainingState(stage100,identity=id100))
        if args.smoke:
            summary=summarize(metrics,baseline)
            summary.update(status='smoke-complete',epochs=1,performance_evidence=False)
            write(args.output/'SMOKE.json',summary)
        else:
            state=torch.load(stage100/'last_training.pt',map_location='cpu',weights_only=False)
            original_sha=sha(stage100/'last_training.pt')
            id150=dict(identity,config=hashlib.sha256(json.dumps(asdict(cfg150),sort_keys=True).encode()).hexdigest(),
                original_state_sha256=original_sha,lr_override=.0001)
            state=continuation_state(state,id150)
            stage150=args.output/'stage150';stage150.mkdir(exist_ok=False)
            for refs in state['best_references'].values():
                for ref in refs.values():
                    p=Path(ref['path'])
                    if p.is_absolute() or '..' in p.parts or sha(stage100/p)!=ref['sha256']:
                        raise ValueError('Invalid BEST recovery reference')
                    target=stage150/p;target.parent.mkdir(parents=True,exist_ok=True)
                    if not target.exists():shutil.copy2(stage100/p,target)
            history100=state['history'];schedule_identity=state['schedule_identity']
            _atomic(stage150/'last_training.pt',lambda f:torch.save(state,f));del state
            trace,expected=stage150/'diagnostics',.0001
            metrics=trainer.run_experiment(cfg150,*data['feature_roots'],str(stage150),
                training_state=TrainingState(stage150,identity=id150,schedule_identity=schedule_identity))
            history=read(stage150/'history.json')
            if len(history)!=150 or history[:100]!=history100:raise ValueError('Incomplete150 or changed100 prefix')
            if sha(stage100/'last_training.pt')!=original_sha:raise ValueError('Original100 state changed')
            summary=summarize(metrics,baseline);summary.update(status='complete',epochs=150)
            write(args.output/'SUMMARY.json',summary)
        record.update(status='smoke-complete' if args.smoke else 'complete',finished_at=now(),
            seconds=time.monotonic()-start,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            summary=summary)
    except BaseException as exc:
        record.update(status='failed',error=repr(exc),finished_at=now());raise
    finally:
        write(args.output/'PROVENANCE.json',record)


if __name__=='__main__':main()
