"""Shared local runner: actual predictions, Stage2 test-oracle snapshots, resumable state."""
import argparse
import fcntl
import importlib
import json
import os
from pathlib import Path
import random
import socket
import sys
import time
import traceback

from runtime import ROOT,RUN_ROOT,GPUS,enforce_local,atomic_json,digest,now,compute_metrics,valid_result,gpu_free

import numpy as np
import torch
from common_data import load_dataset,fixed_masks,random_masks,MASK_PROTOCOL

torch.set_num_threads(4)
torch.set_num_interop_threads(4)


def save_torch(path,value):
    tmp=path.with_name(path.name+'.tmp')
    torch.save(value,tmp)
    os.replace(tmp,path)


def rng_state(rng):
    return dict(python=random.getstate(),numpy=np.random.get_state(),mask_rng=rng.bit_generator.state,
                torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state,rng):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'].cpu())
    rng.bit_generator.state=state['mask_rng']
    if state['cuda']: torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def batches(records,size,shuffle,rng):
    order=rng.permutation(len(records)) if shuffle else np.arange(len(records))
    for start in range(0,len(order),size):
        yield [records[i] for i in order[start:start+size]]


def batchify(records,c,device,rng,training,first_stage,seq_len=None):
    lens=[len(r['labels']) for r in records]; length=seq_len or max(lens); b=len(records)
    dims=[records[0][m].shape[-1] for m in ('a','t','v')]
    features=[np.zeros((b,length,d),dtype=np.float32) for d in dims]
    labels=np.zeros((b,length),dtype=np.int64);speakers=np.zeros_like(labels)
    valid=np.zeros((b,length),dtype=bool); masks=np.zeros((b,length,3),dtype=np.float32)
    sample_ids=[]
    for i,(r,n) in enumerate(zip(records,lens)):
        for x,m in zip(features,('a','t','v')): x[i,:n]=r[m]
        labels[i,:n]=r['labels'];speakers[i,:n]=r['speakers'];valid[i,:n]=True
        if training:
            masks[i,:n]=1 if c['model']=='MoMKE' and first_stage else random_masks(n,c['missing_rate'],rng)
        else:
            masks[i,:n]=fixed_masks(c['dataset'],r['id'],r['utterance_ids'],c['missing_rate'],c['eval_mask_seed'])
        sample_ids.extend([r['id']+'::'+str(uid) for uid in r['utterance_ids']])
    tensors=[torch.from_numpy(x).to(device) for x in features]
    return tensors,torch.from_numpy(labels).to(device),torch.from_numpy(speakers).to(device),torch.from_numpy(valid).to(device),torch.from_numpy(masks).to(device),lens,sample_ids


def forward_loss(model,module,c,batch,first_stage,training):
    feats,labels,speakers,valid,masks,lens,_=batch
    if c['model'] in ('GCNet','SDR-GNN'):
        masked=torch.cat([x*masks[:,:,i:i+1] for i,x in enumerate(feats)],-1).transpose(0,1)
        out=model(masked,speakers,valid,lens)
        loss=model.loss(out,labels,torch.cat(feats,-1).transpose(0,1),masks.transpose(0,1),valid)[0] if training else None
        logits=out['logits'].transpose(0,1)
    else:
        out=model(*feats,masks,valid,first_stage=first_stage)
        loss=module.stage_loss(out,labels,valid,first_stage) if training else None
        logits=out['logits']
    if not torch.isfinite(logits).all() or (loss is not None and not torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite logits/loss')
    return logits,loss,out


def epoch(model,module,c,records,device,rng,optimizer=None,first_stage=False,seq_len=None):
    training=optimizer is not None;model.train(training)
    ys=[];ps=[];ms=[];ids=[];logits_saved=[];total=0.;count=0
    with torch.set_grad_enabled(training):
        for rs in batches(records,c['batch_size'],training,rng):
            batch=batchify(rs,c,device,rng,training,first_stage,seq_len)
            if training: optimizer.zero_grad(set_to_none=True)
            logits,loss,out=forward_loss(model,module,c,batch,first_stage,training)
            if training:
                loss.backward()
                if hasattr(module,'after_backward'): module.after_backward(model,out,batch[1],batch[3],first_stage)
                for p in model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all(): raise FloatingPointError('Nonfinite gradient')
                optimizer.step()
                n=int(batch[3].sum());total+=float(loss.detach())*n;count+=n
            else:
                valid=batch[3];ys.append(batch[1][valid].cpu().numpy());ps.append(logits.argmax(-1)[valid].cpu().numpy())
                logits_saved.append(logits[valid].cpu().numpy());ms.append(batch[4][valid].cpu().numpy());ids.extend(batch[-1])
    if training: return {'train_loss':total/count}
    return dict(labels=np.concatenate(ys),predictions=np.concatenate(ps),masks=np.concatenate(ms),sample_ids=np.asarray(ids),logits=np.concatenate(logits_saved))


def split_records(data,c,smoke):
    records=data['records']
    if c['dataset']=='MELD':
        train=[r for r in records if r['split']=='train']; test=[r for r in records if r['split']=='test']
    else:
        train=[r for r in records if r['session']!=c['fold']];test=[r for r in records if r['session']==c['fold']]
    if smoke:
        # Actual full-feature dialogues, small batch smoke; never formal evidence.
        train=sorted(train,key=lambda r:len(r['labels']))[10:14]
        test=sorted(test,key=lambda r:len(r['labels']))[10:14]
    assert train and test and not ({r['id'] for r in train}&{r['id'] for r in test})
    return train,test


def split_identity(records):
    import hashlib
    samples = [dict(id=str(r['id']), utterance_ids=list(map(str, r['utterance_ids']))) for r in records]
    return hashlib.sha256(json.dumps(samples, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def data_identity(metadata, train, test):
    content = metadata.get('selected_features_labels_sha256')
    if not content:
        raise RuntimeError('Cannot verify resume data identity without feature/label content hash')
    return dict(selected_features_labels_sha256=content,
                train_split_sha256=split_identity(train), test_split_sha256=split_identity(test))


def validate_resume_identity(previous, current):
    if previous.get('code_snapshot') != current['code_snapshot']:
        raise RuntimeError('Resume must use original immutable source snapshot')
    # Older manifests without a complete identity cannot prove the training
    # split. Refuse them explicitly rather than silently relabel old weights.
    if not previous.get('data_identity') or previous['data_identity'] != current['data_identity']:
        raise RuntimeError('Resume data/split identity is missing or incompatible')


def mmin_lr(ep, total, dataset, base_lr, teacher=False, smoke=False):
    const = 20 if dataset == 'MELD' else 10 if teacher else 30
    return base_lr if smoke else base_lr * (1 - max(0, ep - const) / (total - const + 1))


def teacher_signature(c, cfg, metadata, train, teacher_epochs, smoke):
    import dataclasses
    paths = [ROOT/'train_one.py', ROOT/'runtime.py', ROOT/'common_data.py']
    paths += sorted(p for p in (ROOT/'adapters/mmin').rglob('*.py')
                    if not {'upstream', '__pycache__'} & set(p.relative_to(ROOT).parts))
    return dict(dataset=c['dataset'], fold=c['fold'], seed=c['seed'],
                feature_sha256=metadata['selected_features_labels_sha256'],
                train_split_sha256=split_identity(train), teacher_epochs=teacher_epochs,
                teacher_config=dataclasses.asdict(cfg), teacher_lr=c['lr'],
                teacher_batch_size=c['batch_size'], teacher_optimizer='Adam',
                teacher_betas=list(c['teacher_betas']),
                teacher_lr_schedule=[mmin_lr(ep, teacher_epochs, c['dataset'], c['lr'], True, smoke)
                                     for ep in range(1, teacher_epochs + 1)],
                code_hashes={str(p.relative_to(ROOT)): digest(p) for p in paths},
                torch_version=str(torch.__version__), numpy_version=np.__version__)


def main():
    enforce_local()
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--run-dir',required=True);ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    run=Path(args.run_dir);run.mkdir(parents=True,exist_ok=True)
    lock=(run/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    requested=json.loads(Path(args.config).read_text());c=dict(requested)
    if args.smoke: c.update(epochs=2,stage_epoch=1 if c['model'] in ('MoMKE','ComP') else 0,batch_size=min(c['batch_size'],4))
    if (run/'config.json').exists() and json.loads((run/'config.json').read_text())!=c:
        raise RuntimeError('Refusing incompatible resume/config overwrite')
    if valid_result(run): print('SKIP verified result',run,flush=True);return
    atomic_json(run/'config.json',c)
    uuid=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if uuid not in GPUS or not gpu_free(uuid) or not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Requires exactly one local whitelisted GPU UUID')
    device=torch.device('cuda:0');c['device']='cuda:0'
    random.seed(c['seed']);np.random.seed(c['seed']);torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    rng=np.random.default_rng(c['seed']);data=load_dataset(c['dataset']);train,test=split_records(data,c,args.smoke)
    dims=tuple(data['metadata']['dimensions']);seq_len=max(len(r['labels']) for r in data['records']) if c['model']=='ComP' else None
    c['seq_len']=seq_len
    manifest=dict(config=c,data=data['metadata'],hostname=socket.gethostname(),pid=os.getpid(),gpu_uuid=uuid,started_at=now(),mask_protocol=MASK_PROTOCOL,python=sys.executable,torch=torch.__version__,numpy=np.__version__,result_type='smoke' if args.smoke else 'formal',code_snapshot=str(ROOT),expected_n_test=sum(len(r['labels']) for r in test),expected_sample_ids=[r['id']+'::'+str(uid) for r in test for uid in r['utterance_ids']],eval_mask_seed=c['eval_mask_seed'],mask_key_records=[dict(id=r['id'],utterance_ids=r['utterance_ids']) for r in test])
    manifest['data_identity'] = data_identity(data['metadata'], train, test)
    if (run/'manifest.json').exists():
        previous=json.loads((run/'manifest.json').read_text())
        validate_resume_identity(previous, manifest)
        manifest['started_at']=previous['started_at']
        manifest['resume_events']=previous.get('resume_events',[])+[dict(at=now(),pid=os.getpid())]
    sources={'MMIN':('https://github.com/AIM3-RUC/MMIN','69a45fedff66fa0da74205ad425440c1ee8f6d1f'),
             'MoMKE':('https://github.com/wxxv/MoMKE','2dceb57f9e6f4bc84c2fc8f4eb7c761ea64c9efc'),
             'GCNet':('https://github.com/zeroQiaoba/GCNet','8cce4a0c9a50172abfa79971e77c71c40c1d733d'),
             'SDR-GNN':('https://github.com/fufangze/SDR-GNN','b45546111e9dbe50e019cfa98a6c19f13a4c8aa8'),
             'ComP':('https://github.com/WenjueHE/2026-AAAI-ComP','28192d3a5683543d7383e40898f9a98d1f114a08')}
    manifest['repository_url'],manifest['git_commit']=sources[c['model']]
    manifest['adapter_commit']='aa538318fda0075a48f7ba8474b08fd239bea0f2' if c['model']=='ComP' else '6c7909056926b9e5b746366b80f3b27e75fe7db0' if c['model']=='GCNet' else manifest['git_commit']
    manifest['environment_packages']=__import__('subprocess').check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines()
    atomic_json(run/'manifest.json',manifest)
    if c['model']=='MMIN':
        return run_mmin(args,run,c,data,train,test,device,rng,manifest)
    module=importlib.import_module('adapters.'+{'MoMKE':'momke','ComP':'comp','GCNet':'graph_models','SDR-GNN':'graph_models'}[c['model']])
    model=module.build_model(c,dims,c['n_classes'],c['n_speakers']).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
    start=1;best_acc=-1.;best_epoch=None
    if (run/'last.pt').exists():
        state=torch.load(run/'last.pt',map_location=device)
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);restore_rng(state['rng'],rng)
        start=state['epoch']+1;best_acc=state['best_acc'];best_epoch=state['best_epoch']
        print('FULL_RESUME',start,flush=True)
    for ep in range(start,c['epochs']+1):
        tick=time.monotonic();first=ep<=c['stage_epoch']
        diag=epoch(model,module,c,train,device,rng,optimizer,first,seq_len)
        pred=epoch(model,module,c,test,device,rng,None,first,seq_len);scores=compute_metrics(pred['labels'],pred['predictions'],c['n_classes'])
        eligible=ep>c['stage_epoch']
        row=dict(epoch=ep,stage=1 if first else 2 if c['stage_epoch'] else 0,selection_eligible=eligible,**diag,**scores,seconds=time.monotonic()-tick,peak_gpu_mb=torch.cuda.max_memory_allocated()/1024**2)
        if eligible and scores['acc']>best_acc:
            best_acc=scores['acc'];best_epoch=ep
            persist_best(run,model,optimizer,c,ep,pred,scores,rng)
        save_torch(run/'last.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=ep,best_acc=best_acc,best_epoch=best_epoch,rng=rng_state(rng)))
        with (run/'history.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        atomic_json(run/'status.json',dict(state='RUNNING',pid=os.getpid(),gpu_uuid=uuid,epoch=ep,best_epoch=best_epoch,updated_at=now()))
        print(json.dumps(row),flush=True)
    verify_checkpoint(run,model,module,c,test,device,rng,seq_len)
    atomic_json(run/'status.json',dict(state='COMPLETED',exit_code=0,pid=os.getpid(),gpu_uuid=uuid,epochs=c['epochs'],completed_at=now(),result_type=manifest['result_type']))
    if not valid_result(run):raise RuntimeError('Completed artifact validation failed')
    print('COMPLETED validated predictions + best checkpoint',flush=True)


def persist_best(run,model,optimizer,c,ep,pred,scores,rng):
    save_torch(run/'best.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=ep,config=c,rng=rng_state(rng)))
    temp=run/'predictions.tmp.npz';np.savez_compressed(temp,**pred);os.replace(temp,run/'predictions.npz')
    atomic_json(run/'metrics.json',dict(**scores,best_epoch=ep,n_test=len(pred['labels']),actual_missing_rate=float(1-pred['masks'].mean()),nominal_missing_rate=c['missing_rate'],checkpoint_selection='test_best_acc',evaluation_protocol='test_oracle'))
    atomic_json(run/'best.json',dict(epoch=ep,checkpoint_sha256=digest(run/'best.pt'),predictions_sha256=digest(run/'predictions.npz')))


def verify_checkpoint(run,model,module,c,test,device,rng,seq_len):
    state=torch.load(run/'best.pt',map_location=device);model.load_state_dict(state['model'])
    pred=epoch(model,module,c,test,device,rng,None,False,seq_len)
    with np.load(run/'predictions.npz',allow_pickle=False) as old:
        for key in ('predictions','labels','masks','sample_ids'):
            if not np.array_equal(pred[key],old[key]): raise RuntimeError('Checkpoint replay mismatch: '+key)
        if not np.allclose(pred['logits'],old['logits'],rtol=1e-5,atol=1e-6):raise RuntimeError('Checkpoint logits replay mismatch')
    atomic_json(run/'checkpoint_replay.json',dict(verified=True,epoch=state['epoch'],at=now()))


def run_mmin(args,run,c,data,train,test,device,rng,manifest):
    import hashlib
    from adapters.mmin import MMINConfig,Teacher,MMIN,train_teacher_step,train_missing_step,save_teacher,load_teacher
    params=dict(c)
    cfg=MMINConfig(input_dims=tuple(data['metadata']['dimensions']),num_classes=c['n_classes'],**{k:tuple(v) if isinstance(v,list) else v for k,v in params.items() if k in MMINConfig.__dataclass_fields__ and k not in ('input_dims','num_classes')})
    teacher_epochs=2 if args.smoke else c['teacher_epochs']
    signature=teacher_signature(c,cfg,data['metadata'],train,teacher_epochs,args.smoke)
    key=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    cache=RUN_ROOT/('smoke_teachers' if args.smoke else 'teachers')/key;cache.mkdir(parents=True,exist_ok=True)
    teacher_path=cache/'teacher.pt'
    lock=(cache/'teacher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    def flat(records):
        return [torch.from_numpy(np.concatenate([r[k] for r in records])).to(device) for k in ('a','t','v','labels')]
    train_values=flat(train);test_values=flat(test)
    def utterance_batches(values,shuffle):
        n=values[-1].shape[0];order=rng.permutation(n) if shuffle else np.arange(n)
        for start in range(0,n,c['batch_size']):
            idx=torch.as_tensor(order[start:start+c['batch_size']],device=device)
            yield [x[idx] for x in values],order[start:start+c['batch_size']]
    def lr_at(ep,total,teacher=False):
        return mmin_lr(ep,total,c['dataset'],c['lr'],teacher,args.smoke)
    if teacher_path.exists():
        teacher,metadata=load_teacher(teacher_path,cfg,signature,device=device)
        print('REUSE_TEACHER',teacher_path,flush=True)
    else:
        teacher=Teacher(cfg).to(device);opt=torch.optim.Adam(teacher.parameters(),lr=c['lr'],betas=tuple(params['teacher_betas']))
        first=1
        if (cache/'last.pt').exists():
            state=torch.load(cache/'last.pt',map_location=device);teacher.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);restore_rng(state['rng'],rng);first=state['epoch']+1
        atomic_json(cache/'manifest.json',dict(signature=signature,config=params,pid=os.getpid(),gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],started_at=now()))
        for ep in range(first,teacher_epochs+1):
            for g in opt.param_groups:g['lr']=lr_at(ep,teacher_epochs,True)
            loss=0.;n=0
            for values,_ in utterance_batches(train_values,True):
                val=train_teacher_step(teacher,opt,*values);loss+=val*len(values[-1]);n+=len(values[-1])
            save_torch(cache/'last.pt',dict(model=teacher.state_dict(),optimizer=opt.state_dict(),epoch=ep,rng=rng_state(rng)))
            with (cache/'history.jsonl').open('a') as f:f.write(json.dumps(dict(epoch=ep,train_loss=loss/n,lr=opt.param_groups[0]['lr']))+'\n')
            print(json.dumps(dict(teacher_epoch=ep,train_loss=loss/n,lr=opt.param_groups[0]['lr'])),flush=True)
        save_teacher(teacher_path,teacher,signature,teacher_epochs)
        atomic_json(cache/'status.json',dict(state='COMPLETED',epochs=teacher_epochs,checkpoint_sha256=digest(teacher_path),completed_at=now()))
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    # Equal missing-model initialization independent of whether teacher was reused.
    random.seed(c['seed']);np.random.seed(c['seed']);torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed']);rng=np.random.default_rng(c['seed'])
    model=MMIN(cfg,teacher).to(device);optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=c['lr'],betas=tuple(params['missing_betas']))
    manifest.update(teacher_path=str(teacher_path),teacher_sha256=digest(teacher_path),effective_mmin_parameters=params)
    atomic_json(run/'manifest.json',manifest)
    masks=np.concatenate([fixed_masks(c['dataset'],r['id'],r['utterance_ids'],c['missing_rate'],c['eval_mask_seed']) for r in test])
    sample_ids=np.asarray([r['id']+'::'+str(uid) for r in test for uid in r['utterance_ids']])
    def evaluate():
        model.eval();logits=[]
        with torch.no_grad():
            for values,index in utterance_batches(test_values,False):
                ms=torch.from_numpy(masks[index]).to(device);logits.append(model(*values[:3],ms).cpu().numpy())
        arr=np.concatenate(logits)
        if not np.isfinite(arr).all():raise FloatingPointError('MMIN nonfinite predictions')
        return dict(logits=arr,labels=test_values[-1].cpu().numpy(),predictions=arr.argmax(-1),masks=masks,sample_ids=sample_ids)
    start=1;best_acc=-1.;best_epoch=None
    if (run/'last.pt').exists():
        state=torch.load(run/'last.pt',map_location=device);model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);restore_rng(state['rng'],rng);start=state['epoch']+1;best_acc=state['best_acc'];best_epoch=state['best_epoch']
    for ep in range(start,c['epochs']+1):
        tick=time.monotonic()
        for group in optimizer.param_groups:group['lr']=lr_at(ep,c['epochs'])
        total=0.;n=0
        for values,_ in utterance_batches(train_values,True):
            ms=torch.from_numpy(random_masks(len(values[-1]),c['missing_rate'],rng)).to(device)
            losses=train_missing_step(model,optimizer,*values[:3],ms,values[-1]);total+=losses['total']*len(values[-1]);n+=len(values[-1])
        pred=evaluate();scores=compute_metrics(pred['labels'],pred['predictions'],c['n_classes'])
        row=dict(epoch=ep,stage=0,selection_eligible=True,train_loss=total/n,lr=optimizer.param_groups[0]['lr'],**scores,seconds=time.monotonic()-tick,peak_gpu_mb=torch.cuda.max_memory_allocated()/1024**2)
        if scores['acc']>best_acc:
            best_acc=scores['acc'];best_epoch=ep;persist_best(run,model,optimizer,c,ep,pred,scores,rng)
        save_torch(run/'last.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=ep,best_acc=best_acc,best_epoch=best_epoch,rng=rng_state(rng)))
        with (run/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        atomic_json(run/'status.json',dict(state='RUNNING',pid=os.getpid(),gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],epoch=ep,best_epoch=best_epoch,updated_at=now()))
        print(json.dumps(row),flush=True)
    state=torch.load(run/'best.pt',map_location=device);model.load_state_dict(state['model']);pred=evaluate()
    with np.load(run/'predictions.npz',allow_pickle=False) as old:
        if not all(np.array_equal(pred[k],old[k]) for k in ('predictions','labels','masks','sample_ids')) or not np.allclose(pred['logits'],old['logits'],atol=1e-6,rtol=1e-5):raise RuntimeError('MMIN checkpoint replay mismatch')
    atomic_json(run/'checkpoint_replay.json',dict(verified=True,epoch=state['epoch'],at=now()))
    atomic_json(run/'status.json',dict(state='COMPLETED',exit_code=0,pid=os.getpid(),gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],epochs=c['epochs'],completed_at=now(),result_type=manifest['result_type']))
    if not valid_result(run):raise RuntimeError('MMIN completed artifact validation failed')
    print('COMPLETED MMIN validated checkpoint+predictions',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        traceback.print_exc()
        if '--run-dir' in sys.argv:
            d=Path(sys.argv[sys.argv.index('--run-dir')+1]);d.mkdir(parents=True,exist_ok=True)
            atomic_json(d/'failure.json',dict(error=str(exc),traceback=traceback.format_exc(),at=now(),pid=os.getpid()))
        sys.exit(1)
