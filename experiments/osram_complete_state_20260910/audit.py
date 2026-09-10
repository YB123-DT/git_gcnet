"""Frozen .5 checkpoints: utility, prediction quality, paired shared gradients."""
import hashlib
import inspect
import json
from dataclasses import asdict
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from gcnet_missing_m3 import train_gcnet as tr
from gcnet_missing_m3.model import MissingM3GraphModel
from experiments.osram_predictor_reaudit_20260909.audit import old
from experiments.osram_complete_state_20260910.run import ROOT,base

PAT={1:'V',2:'T',3:'TV',4:'A',5:'AV',6:'AT',7:'ATV'}


def ridge_metrics(x,y,v,z):
    scaler=StandardScaler().fit(x)
    predictor=Ridge(alpha=10.,solver='cholesky').fit(scaler.transform(x),y)
    out=predictor.predict(scaler.transform(v));keep=z!=0
    return dict(weighted_f1=float(f1_score(z[keep]>0,out[keep]>0,average='weighted')),
                accuracy=float(accuracy_score(z[keep]>0,out[keep]>0)),
                mae=float(np.abs(z-out).mean()))


def gradient_record(model,view,cfg,batch_index):
    # One shared forward, deterministic eval-style dropout, no optimizer/update.
    logits,h,_,_=model([view['incomplete']],view['availability'],view['qmask'],view['umask'],view['lengths'])
    cls=tr._task_loss(cfg.dataset,logits,view['labels'],view['umask'],cfg.mosi_task_mode,
                     cfg.task_regression_loss,cfg.task_smooth_l1_beta)
    state,n=model.complete_state_loss(h,view['complete'],view['availability'],view['umask'])
    named=[(k,p) for k,p in model.named_parameters() if p.requires_grad and
           (k.startswith('osram.') or k.startswith('observed_set.'))]
    params=[p for _,p in named]
    g1=torch.autograd.grad(cls,params,retain_graph=True,allow_unused=True)
    g2=torch.autograd.grad(state,params,allow_unused=True)
    rows=[]
    for group,prefix in [('encoder','observed_set.'),('osram','osram.'),('shared','')]:
        dot=n1=n2=0.;used=0
        for (name,p),a,b in zip(named,g1,g2):
            if not name.startswith(prefix):continue
            if a is not None:n1+=float(a.double().square().sum())
            if b is not None:n2+=float(b.double().square().sum())
            if a is not None and b is not None:dot+=float((a.double()*b.double()).sum());used+=p.numel()
        rows.append(dict(batch=batch_index,group=group,cosine=dot/np.sqrt(n1*n2) if n1*n2>0 else None,
            emotion_norm=np.sqrt(n1),state_norm=np.sqrt(n2),
            weighted_state_to_emotion=cfg.jepa_weight*np.sqrt(n2)/max(np.sqrt(n1),1e-12),
            shared_active_parameters=used,state_targets=n,emotion_loss=float(cls),state_loss=float(state)))
    return rows


def audit(seed):
    path=ROOT/'mosi'/f'seed_{seed}'/'best_miss_0p5.pt'
    cp=torch.load(path,map_location='cpu',weights_only=False)
    cfg=tr.TrainConfig(**cp['config']);tr.set_random_seed(seed)
    roots=[str(base.runner.FEATURES/n) for n in ('wav2vec-large-c-UTT','deberta-large-4-UTT','manet_UTT')]
    train,val,_,*dims=tr.get_loaders(audio_root=roots[0],text_root=roots[1],video_root=roots[2],
        num_folder=1,dataset=cfg.dataset,batch_size=cfg.batch_size,num_workers=0,seed=seed,
        evaluation_protocol=cfg.evaluation_protocol,validation_fraction=cfg.validation_fraction)
    names=inspect.signature(MissingM3GraphModel.__init__).parameters
    kw={k:v for k,v in asdict(cfg).items() if k in names}
    kw.update(adim=dims[0],tdim=dims[1],vdim=dims[2],D_e=cfg.hidden,graph_hidden_size=cfg.hidden//2,
              n_speakers=1,n_classes=1,time_attn=cfg.time_attention,complete_state_jepa=True)
    device=torch.device('cuda:0');model=MissingM3GraphModel(**kw).to(device).eval()
    model.load_state_dict(cp['model'],strict=True)
    collected={};gradients=[];mask_hash={};ids={}
    for split,loader in [('train',train[cfg.fold-1]),('validation',val[cfg.fold-1])]:
        schedule=tr._build_schedule(cfg,split,.5);arrays={};digest=hashlib.sha256();ids[split]=[]
        for bi,raw in enumerate(loader):
            view=tr._prepare_view(tr._move_batch(raw,device),schedule,0,tuple(dims))
            valid=view['umask'].T.bool();a=view['availability'];sel=valid & a.sum(-1).lt(3)
            digest.update(a[valid].cpu().numpy().tobytes());ids[split]+=list(map(str,view['conversation_ids']))
            if split=='train' and bi<3:gradients+=gradient_record(model,view,cfg,bi)
            with torch.no_grad():
                _,h,_,_=model([view['incomplete']],a,view['qmask'],view['umask'],view['lengths'])
                full=valid[...,None].expand(-1,-1,3).to(a.dtype)
                e,_=model.state_jepa.teacher_encoder(view['complete'],full,view['umask'])
                s=(e+model.state_jepa.teacher_local(e)).masked_fill(~valid[...,None],0)
                pred=model.state_jepa.predict(h,a,view['umask'])
                values=dict(h=h[sel],e=e[sel],s=s[sel],pred=pred[sel],
                    y=view['labels'].T[sel],pattern=(a[...,0]*4+a[...,1]*2+a[...,2])[sel])
                for k,v in values.items():arrays.setdefault(k,[]).append(v.cpu())
        collected[split]={k:torch.cat(v) for k,v in arrays.items()};mask_hash[split]=digest.hexdigest()
    assert not set(ids['train']) & set(ids['validation'])
    assert all(torch.equal(v.cpu(),cp['model'][k]) for k,v in model.state_dict().items())
    assert model.ema_step==0 # Never updated in this audit.
    a,b=collected['train'],collected['validation'];utility=[]
    rng=np.random.default_rng(seed)
    arrays={k:(a[k].numpy(),b[k].numpy()) for k in ('h','e','s')}
    for k in ('e','s'):arrays['h+'+k]=(np.concatenate([arrays['h'][0],arrays[k][0]],1),np.concatenate([arrays['h'][1],arrays[k][1]],1))
    arrays['h+shuffled_s']=(np.concatenate([arrays['h'][0],rng.permutation(arrays['s'][0])],1),
                            np.concatenate([arrays['h'][1],rng.permutation(arrays['s'][1])],1))
    for name,(x,v) in arrays.items():
        utility.append(dict(representation=name,dimension=x.shape[1],alpha=10.,
                            **ridge_metrics(x,a['y'].numpy(),v,b['y'].numpy())))
    quality=[]
    for pattern in ['ALL']+[PAT[i] for i in range(1,7)]:
        sel=torch.ones(len(b['y']),dtype=torch.bool) if pattern=='ALL' else b['pattern']==next(k for k,v in PAT.items() if v==pattern)
        if sel.sum()<2:continue
        p,t=b['pred'][sel],b['s'][sel]
        metrics=old._metrics(p,t,cfg.temperature,seed)
        metrics['std_ratio']=metrics['channel_std']/max(metrics['target_channel_std'],1e-12)
        metrics['prediction_mean_variance']=float(p.var(0,unbiased=False).mean())
        metrics['teacher_mean_variance']=float(t.var(0,unbiased=False).mean())
        quality.append(dict(pattern=pattern,**metrics))
    output=ROOT/'audit';output.mkdir(exist_ok=True)
    result=dict(seed=seed,checkpoint=str(path),checkpoint_sha256=base.runner.sha(path),epoch=cp['epoch'],rate=.5,
        protocol='Frozen Test-oracle checkpoint; probes train-fit/validation-eval; missing utterances only; alpha10 fixed',
        dropout='eval for paired deterministic gradients',state_unchanged=True,mask_sha256=mask_hash,
        ids=ids,utility=utility,prediction=quality,gradients=gradients,
        count={k:len(v['y']) for k,v in collected.items()})
    (output/f'seed_{seed}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(seed=seed,utility=utility,prediction=quality[0],gradients=gradients)),flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    for seed in range(66,71):audit(seed)
