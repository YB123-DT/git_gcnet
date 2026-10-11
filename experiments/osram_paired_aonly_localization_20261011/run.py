"""Freeze three existing checkpoints and locate persistent A-only polarity bias."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--commit',required=True);args=p.parse_args()
    assert not args.output.exists()
    uuid='GPU-e4cafb17-818e-216a-b94a-7440063a9153'
    assert os.environ['CUDA_VISIBLE_DEVICES']==uuid
    assert subprocess.check_output(['nvidia-smi','--id=6','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==uuid
    import numpy as np
    import torch
    from gcnet_modality_jepa.train_gcnet import get_loaders
    from gcnet_missing_m3.train_gcnet import (TrainConfig,_attach_history_projector,_move_batch,
        _prepare_view,_prepare_view_from_primary_masks,_schedules,_metrics)
    from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model,_roots
    from experiments.osram_history_drift_20261002.run import read,write,sha,state_hash
    torch.set_num_threads(2)
    os.environ['GCNET_DATASET_ROOT']='/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset'
    sources={
        'Flat':Path('/data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full/seed_66'),
        'rho050':Path('/data1/yb/remote_experiments/osram_paired_history_views_20261001/runs/control'),
        'rho010':Path('/data2/yb/remote_experiments/osram_paired_rho_sweep_20261002/runs/C02')}
    args.output.mkdir(parents=True)
    state=dict(status='running',code_commit=args.commit,backbone_commit='6e2a998',gpu=6,
        label='INTERNAL DIAGNOSTIC ONLY',training=False,pid=os.getpid(),records=[])
    write(args.output/'STATUS.json',state)
    rows=[]
    for name,source in sources.items():
        c=TrainConfig(**read(source/'config.json'))
        _,_,test,*dims=get_loaders(audio_root=_roots()[0],text_root=_roots()[1],video_root=_roots()[2],
            num_folder=1,dataset='CMUMOSI',batch_size=c.batch_size,num_workers=0,seed=66,
            validation_fraction=c.validation_fraction,evaluation_protocol=c.evaluation_protocol)
        model=_build_model(c,tuple(dims)).cuda()
        if c.paired_history_views:_attach_history_projector(model,c)
        checkpoint=source/'best_miss_0p7.pt';digest=sha(checkpoint)
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model'],strict=True);epoch=int(saved['epoch']);del saved
        model.eval().requires_grad_(False);before=state_hash(model)
        archive=read(source/'metrics.json')
        loader=test[0];loader.sampler.set_epoch(0)
        preds=[];labels=[];avs=[];max_parity=0.;max_local=0.
        with torch.no_grad():
            for raw in loader:
                data=_move_batch(raw,torch.device('cuda:0'))
                random=_prepare_view(data,_schedules(c,'test')[.7],0,tuple(dims))
                mask=torch.zeros_like(random['availability']);mask[...,0]=random['umask'].T
                fixed=_prepare_view_from_primary_masks(data,mask,mask,tuple(dims))
                records={}
                for regime,v in [('random',random),('persistent',fixed)]:
                    inputs=[]
                    hook=model.osram.emotion_adapter.register_forward_pre_hook(lambda module,x:inputs.append(x[0].detach().clone()))
                    try:
                        result=model([v['incomplete']],v['availability'],v['qmask'],v['umask'],v['lengths'],predict_missing=False)
                    finally:hook.remove()
                    assert len(inputs)==1
                    x=inputs[0];local=x[...,:256];base=x[...,256:1280];gap=x[...,1280:]
                    assert x.shape[-1]==4352
                    valid=v['umask'].T.bool();outputs={}
                    for key,b,g in [('L',base*0,gap*0),('LB',base,gap*0),('LG',base*0,gap),('LBG',base,gap)]:
                        hidden=model.osram.emotion_norm(model.osram.local_skip(local)+model.osram.emotion_adapter(torch.cat((local,b,g),-1)))
                        outputs[key]=torch.where(valid,model.smax_fc(hidden).squeeze(-1),0.)
                    err=float((outputs['LBG'][valid]-result[0].squeeze(-1)[valid]).abs().max())
                    max_parity=max(max_parity,err);assert max_parity<=1e-5
                    records[regime]=(local,outputs)
                    prior_text=v['availability'][...,1].long().cumsum(0)-v['availability'][...,1].long()
                    for t,b in valid.nonzero().tolist():
                        pattern=''.join(k for k,a in zip('ATV',v['availability'][t,b].tolist()) if a)
                        if pattern!='A':continue
                        matched=bool(torch.equal(random['availability'][t,b],mask[t,b]))
                        rows.append(dict(model=name,regime=regime,conversation=str(v['conversation_ids'][b]),t=t,
                            label=float(v['labels'][b,t]),matched_random_A=matched,prior_text_count=int(prior_text[t,b]),
                            base_norm=float(base[t,b].norm()),gap_norm=float(gap[t,b].norm()),
                            **{k:float(value[t,b]) for k,value in outputs.items()}))
                    if regime=='random':
                        bv=v['umask'].bool();preds.extend(outputs['LBG'].T[bv].cpu().tolist())
                        labels.extend(v['labels'][bv].cpu().tolist());avs.extend(v['availability'].transpose(0,1)[bv].cpu().tolist())
                anchor=(random['availability']==mask).all(-1)&random['umask'].T.bool()
                if anchor.any():
                    max_local=max(max_local,float((records['random'][0][anchor]-records['persistent'][0][anchor]).abs().max()))
                assert max_local<=1e-6, 'Current Local not isolated'
        old=np.load(source/'predictions_miss_0p7.npz')
        assert np.array_equal(labels,old['labels']) and np.array_equal(avs,old['availability'])
        archive_error=float(abs(np.array(preds)-old['predictions'].reshape(-1)).max());assert archive_error<=1e-5
        assert before==state_hash(model) and digest==sha(checkpoint)
        r=dict(model=name,epoch=epoch,state_sha256=before,checkpoint_sha256=digest,
            replay_max_error=max_parity,archive_max_error=archive_error,matched_local_max_error=max_local)
        state['records'].append(r);write(args.output/'STATUS.json',state);print(json.dumps(r),flush=True)
    table=[]
    for name in sources:
        for regime in ('random','persistent'):
            for cohort in ('all_A','matched_A','first_A','matched_A_with_prior_Text'):
                chosen=[r for r in rows if r['model']==name and r['regime']==regime and
                    (cohort=='all_A' or (cohort=='matched_A' and r['matched_random_A']) or
                    (cohort=='first_A' and r['t']==0) or (cohort=='matched_A_with_prior_Text' and
                     r['matched_random_A'] and any(s['model']==name and s['regime']=='random' and
                        s['conversation']==r['conversation'] and s['t']==r['t'] and s['prior_text_count']>0 for s in rows)))]
                for readout in ('L','LB','LG','LBG'):
                    y=np.array([r['label'] for r in chosen]);pred=np.array([r[readout] for r in chosen])
                    if not len(y):continue
                    keep=y!=0;truth=y[keep]>0;positive=pred[keep]>0
                    metric=_metrics('CMUMOSI',y,pred,'regression')
                    table.append(dict(model=name,regime=regime,cohort=cohort,readout=readout,n=len(y),nonneutral_n=int(keep.sum()),
                        mean=float(pred.mean()),std=float(pred.std()),pred_positive=int(positive.sum()),
                        tp=int((positive&truth).sum()),tn=int((~positive&~truth).sum()),
                        fp=int((positive&~truth).sum()),fn=int((~positive&truth).sum()),**metric))
    for name,data in [('utterances',rows),('summary',table)]:
        with (args.output/f'{name}.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    write(args.output/'SUMMARY.json',dict(records=state['records'],table=table))
    state['status']='complete';write(args.output/'STATUS.json',state)


if __name__=='__main__':main()
