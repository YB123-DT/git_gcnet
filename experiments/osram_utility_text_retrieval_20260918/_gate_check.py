import torch,numpy as np,random
from gcnet_missing_m3 import train_gcnet as tr
from experiments.osram_utility_text_retrieval_20260918.utility_common import build_reader, query_utility_from_batch
seed=66; rate='0.5'; device=torch.device('cuda')
cp=torch.load('/tmp/utility_cache_new2/seed_66/rate_0p5/test.pt',map_location='cpu',weights_only=False)['tensors']
model,cfg,dims,shape,loaders=build_reader(seed,rate,device)
loader=loaders[2][cfg.fold-1]
schedule=tr._build_schedule(cfg,'test',float(rate))
raws=[]
for batch_index,raw in enumerate(loader):
    raws.append((batch_index,tr._move_batch(raw,device)))
# candidate0 normal forward direct
for bi,data in raws:
    view=tr._prepare_view(data,schedule,0,dims)
    with torch.no_grad():
        logits,_,_,_=model(view['incomplete'],view['availability'],view['qmask'],view['umask'],view['lengths'])
    scores=logits[...,0]
    rows=torch.where((cp['batch_index']==bi))[0]
    if len(rows):
        for row in rows.tolist():
            t=int(cp['query_t'][row]); b=int(cp['query_b'][row])
            diff=abs(float(scores[t,b])-float(cp['scores'][row,0]))
            assert diff<1e-5,(row,diff)
print('A_pass')
# B: 32 random nonzero candidates
eligible=[]
for row in range(len(cp['labels'])):
    for k in range(1,int(cp['mask'][row].sum())):
        eligible.append((row,k))
random.seed(0); random.shuffle(eligible)
# index raw by batch
for row,k in eligible[:32]:
    bi=int(cp['batch_index'][row]); t=int(cp['query_t'][row]); b=int(cp['query_b'][row])
    data=dict(raws)[bi]
    view=tr._prepare_view(data,schedule,0,dims)
    avail=view['availability']; umask=view['umask']; qmask=view['qmask']; lengths=view['lengths']
    tiled_incomplete=view['incomplete'][:,b:b+1]
    tiled_avail=avail[:,b:b+1]
    tiled_qmask=qmask[b].unsqueeze(0)
    tiled_umask=umask[b].unsqueeze(0)
    override=torch.zeros((avail.shape[0],1,model.latent_dim),device=device)
    mask=torch.zeros((avail.shape[0],1),dtype=torch.bool,device=device)
    override[t,0]=cp['values'][row,k].to(device)
    mask[t,0]=True
    with torch.no_grad():
        logits,_,_,_=model(tiled_incomplete,tiled_avail,tiled_qmask,tiled_umask,[lengths[b]],pam_text_override=override,pam_override_mask=mask)
    direct=float(logits[t,0,0])
    cached=float(cp['scores'][row,k])
    if abs(direct-cached)>=1e-5:
        print('B_fail',row,k,direct,cached); break
else:
    print('B_pass')
