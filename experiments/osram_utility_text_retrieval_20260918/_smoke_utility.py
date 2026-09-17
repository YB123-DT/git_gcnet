import torch
from gcnet_missing_m3 import train_gcnet as tr
from experiments.osram_utility_text_retrieval_20260918.utility_common import build_reader, query_utility_from_batch, collate_records
device=torch.device('cuda')
model,cfg,dims,shape,loaders=build_reader(66,'0.5',device)
train_loaders,val_loaders,test_loaders=loaders
loader=train_loaders[cfg.fold-1]
raw=next(iter(loader)); data=tr._move_batch(raw,device)
schedule=tr._build_schedule(cfg,'train',0.5); view=tr._prepare_view(data,schedule,0,dims)
with torch.no_grad(): recs,maxc=query_utility_from_batch(model,cfg,view)
print('n_queries',len(recs),'maxc',maxc)
if recs:
    r=recs[0]
    print('context',r['context'].shape,'values',r['values'].shape,'scores',r['scores'].shape,'losses',r['losses'].shape)
    print('finite',all(torch.isfinite(r[k]).all().item() for k in ('context','values','scores','losses','utility')))
    print('counts',[len(x['values']) for x in recs[:5]])
    batch=collate_records(recs,maxc); print({k:v.shape for k,v in batch.items()})
