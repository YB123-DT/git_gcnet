import torch
from gcnet_missing_m3 import train_gcnet as tr
from experiments.osram_utility_text_retrieval_20260918.utility_common import build_reader
model,cfg,dims,shape,loaders=build_reader(66,'0.5',torch.device('cpu'))
train_loaders,val_loaders,test_loaders=loaders
loader=train_loaders[cfg.fold-1]
raw=next(iter(loader))
for i,v in enumerate(raw):
    if torch.is_tensor(v): print(i,'tensor',tuple(v.shape),v.dtype)
    else: print(i,type(v),len(v) if hasattr(v,'__len__') else v)
print('cfg batch',cfg.batch_size,'fold',cfg.fold)
