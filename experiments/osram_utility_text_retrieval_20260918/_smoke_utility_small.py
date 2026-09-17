import time, torch
from gcnet_missing_m3 import train_gcnet as tr
from experiments.osram_utility_text_retrieval_20260918.utility_common import build_reader, query_utility_from_batch
model,cfg,dims,shape,loaders=build_reader(66,'0.5',torch.device('cuda'))
train_loaders,_,_=loaders
raw=next(iter(train_loaders[cfg.fold-1]))
B=4
small=list(raw)
for i in range(6): small[i]=small[i][:, :B]
small[6]=small[6][:B]; small[7]=small[7][:B]; small[8]=small[8][:B]; small[9]=small[9][:B]
data=tr._move_batch(small,torch.device('cuda'))
schedule=tr._build_schedule(cfg,'train',0.5)
view=tr._prepare_view(data,schedule,0,dims)
t0=time.time()
with torch.no_grad(): out=query_utility_from_batch(model,cfg,view)
print('elapsed',time.time()-t0)
if out is None: print('none')
else: print({k:tuple(v.shape) for k,v in out.items()})
