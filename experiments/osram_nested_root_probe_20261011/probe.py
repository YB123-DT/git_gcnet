"""Capacity-matched train-only probes, fixed final epoch; no test selection."""
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from experiments.osram_frozen_memory_audit_20261009.probe import metrics,choose_donors

ARMS=('Flat','Rooted','Flat+Rooted','Flat+DonorRooted','PreDecoder','Flat+PreDecoder')

def prepare(splits):
    projections={k:None for k in ('flat','rooted','predecoder')}
    norms={}
    transformed={name:{} for name in splits}
    for key in projections:
        x=splits['train'][key]
        if key=='flat':
            active=np.ones_like(x,dtype=bool)
        else: active=np.repeat(splits['train']['role_active'],64,axis=1)
        count=active.sum(0).clip(1)
        mean=np.where(active,x,0).sum(0)/count
        std=np.sqrt(np.where(active,(x-mean)**2,0).sum(0)/count).clip(1e-6)
        norms[key]=(mean,std)
        for name,data in splits.items():
            z=(data[key]-mean)/std
            if key!='flat':z=np.where(np.repeat(data['role_active'],64,axis=1),z,0)
            transformed[name][key]=z.astype('float32')
    return transformed,projections,norms

def run_probes(splits,destination,epochs=100):
    torch.set_num_threads(1)
    transformed,projection,norms=prepare(splits)
    np.savez_compressed(destination/'preprocessing.npz',
        **{k+'_mean':v[0] for k,v in norms.items()},**{k+'_scale':v[1] for k,v in norms.items()})
    result=[]
    for seed in (66,67,68):
        features={arm:{} for arm in ARMS}; selected={};donor_record={}
        for split_number,(name,data) in enumerate(splits.items()):
            donors,keep=choose_donors(splits['train'],data,seed+1009*split_number)
            selected[name]=keep
            donor=np.zeros_like(transformed[name]['rooted']);active=donors>=0
            donor[active]=transformed['train']['rooted'][donors[active]]
            donor_record[name]={'indices':donors.tolist(),'included':keep.tolist()}
            h=transformed[name]['flat'];r=transformed[name]['rooted'];d=transformed[name]['predecoder']
            zh=np.zeros_like(h);zr=np.zeros_like(r);a=data['availability'].astype('float32')
            for arm,left,right in [('Flat',h,zr),('Rooted',zh,r),('Flat+Rooted',h,r),
                                   ('Flat+DonorRooted',h,donor),('PreDecoder',zh,d),('Flat+PreDecoder',h,d)]:
                features[arm][name]=torch.tensor(np.concatenate((left,right,a),-1)[keep])
        (destination/f'donors_seed{seed}.json').write_text(json.dumps(donor_record))
        targets={n:torch.tensor(data['labels'][selected[n]],dtype=torch.float32) for n,data in splits.items()}
        for arm in ARMS:
            torch.manual_seed(seed)
            model=nn.Sequential(nn.Linear(3715,32),nn.GELU(),nn.Linear(32,1))
            optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-5)
            generator=torch.Generator().manual_seed(seed)
            x=features[arm];curve=[]
            for epoch in range(epochs):
                model.train();loss_sum=0.
                for ids in torch.randperm(len(x['train']),generator=generator).split(128):
                    optimizer.zero_grad(set_to_none=True)
                    loss=(model(x['train'][ids]).squeeze(-1)-targets['train'][ids]).square().mean()
                    assert torch.isfinite(loss)
                    loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                    optimizer.step();loss_sum+=float(loss)*len(ids)
                curve.append(loss_sum/len(x['train']))
            model.eval()
            with torch.no_grad():pred=model(x['test']).squeeze(-1).numpy()
            labels=targets['test'].numpy()
            folder=destination/f'seed{seed}'/arm.replace('+','_');folder.mkdir(parents=True)
            torch.save({'model':model.state_dict(),'epoch':epochs,'seed':seed},folder/'final.pt')
            np.savez_compressed(folder/'predictions.npz',predictions=pred,labels=labels,
                conversation_ids=splits['test']['conversation_ids'][selected['test']],
                utterance_indices=splits['test']['utterance_indices'][selected['test']],
                availability=splits['test']['availability'][selected['test']])
            row={'arm':arm,'probe_seed':seed,'epoch':epochs,'selection':'fixed final epoch; no test selection',
                 'parameter_count':sum(p.numel() for p in model.parameters()),'train_loss_curve':curve,
                 'test':metrics(labels,pred),'coverage':{n:int(v.sum()) for n,v in selected.items()},'groups':{}}
            availability=splits['test']['availability'][selected['test']]
            for pattern in ('A','T','V','AT','AV','TV','ATV'):
                bits=np.array([int(m in pattern) for m in 'ATV'])
                keep=(availability==bits).all(-1)
                if keep.any():row['groups'][pattern]=metrics(labels[keep],pred[keep])
            result.append(row)
    (destination/'PROBES.json').write_text(json.dumps(result,indent=2))
    return result
