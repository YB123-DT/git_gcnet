"""Bounded real-model verification and optional actual-batch CUDA resource check."""
import argparse
import json
import os
from dataclasses import asdict
import torch
from experiments.osram_core20_20261005.run import candidate_config
from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model
from gcnet_missing_m3.train_gcnet import TrainConfig

parser = argparse.ArgumentParser()
parser.add_argument('reference')
parser.add_argument('--actual-batch',action='store_true')
args = parser.parse_args()
torch.set_num_threads(2)
ref = json.load(open(args.reference))
c,delta = candidate_config(ref,'small_flat_nested_dim704')
assert set(delta)=={'osram_adapter_hidden_dim','osram_meaningful_block'}
assert c.osram_adapter_hidden_dim==256 and c.osram_meaningful_block=='nested_dim704'
net = _build_model(c,(512,1024,1024))
assert sum(p.numel() for p in net.parameters())==13560676
assert sum(p.numel() for p in net.osram.meaningful_block.parameters())==8051715
base = _build_model(TrainConfig(**dict(asdict(c),osram_meaningful_block='none')),(512,1024,1024))
assert all(torch.equal(v,net.state_dict()[k]) for k,v in base.state_dict().items())
node = torch.randn(3,2,256)
latents = {m:torch.randn_like(node) for m in ('audio','text','visual')}
av = torch.tensor([[[1,1,1],[1,0,1]],[[1,0,0],[0,1,0]],[[0,0,1],[0,0,0]]])
um = torch.tensor([[1,1,1],[1,1,0]])
inputs = (node,latents,av,torch.zeros(2,3,dtype=torch.long),um)
base.eval(); net.eval()
with torch.no_grad():
    assert torch.equal(base.osram(*inputs)[0],net.osram(*inputs)[0])
optimizer=torch.optim.Adam(net.parameters(),lr=.001)
before=net.osram.meaningful_block.core.core.layers[0][0].weight.detach().clone()
target=torch.randn(3,2,1600)
for _ in range(3):
    optimizer.zero_grad()
    hidden=net.osram(*inputs)[0]
    assert torch.count_nonzero(hidden[2,1])==0
    (hidden-target).square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
    optimizer.step()
assert not torch.equal(before,net.osram.meaningful_block.core.core.layers[0][0].weight)
# After updates, inactive slots cannot leak via learned biases; first turn is bypassed.
branch=net.osram.meaningful_block
b=torch.randn(3,2,1024); g=torch.randn(3,2,3,1024)
clean=branch(node,b,g,av,um)
poison=g.clone(); inactive=(av.bool() | ~um.T.bool().unsqueeze(-1))
poison[inactive]=float('nan')
masked=branch(node,b,poison,av,um)
for x,y in zip(clean,masked):
    assert torch.equal(x,y) and torch.isfinite(y).all()
    assert torch.count_nonzero(y[2,1])==0
assert torch.equal(clean[0][0],node[0])
assert torch.equal(clean[1][0],b[0])
assert torch.count_nonzero(clean[2][inactive])==0
print('PASS full13560676/module8051715, parent init, zero-init, finite updates, inactive NaN masks, first turn, padding',flush=True)
if args.actual_batch:
    from gcnet_modality_jepa.train_gcnet import get_loaders
    from gcnet_missing_m3.train_gcnet import train_epoch,evaluate_rate,_schedules
    uuid='GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a'
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==uuid
    data=json.load(open('/data2/yb/remote_experiments/osram_meaningful20_round2_20261004/DATA.json'))
    os.environ['GCNET_DATASET_ROOT']=data['dataset_root']
    roots=data['feature_roots']
    train,_,test,ad,td,vd=get_loaders(audio_root=roots[0],text_root=roots[1],video_root=roots[2],num_folder=1,
        dataset=c.dataset,batch_size=c.batch_size,num_workers=0,seed=66,validation_fraction=c.validation_fraction,
        evaluation_protocol=c.evaluation_protocol)
    # _prepare_view_from_primary_masks uses data[7] as [B,L] umask.
    batches=list(train[0])
    batch=max(batches,key=lambda x:float(x[7].sum()))
    net=_build_model(c,(ad,td,vd)).cuda()
    optimizer=torch.optim.Adam(net.parameters(),lr=c.learning_rate,weight_decay=c.weight_decay)
    torch.cuda.reset_peak_memory_stats()
    train_epoch(net,[batch],optimizer,c,_schedules(c,'train'),7,(ad,td,vd),torch.device('cuda'))
    evaluate_rate(net,test[0],_schedules(c,'test')[.7],c.dataset,(ad,td,vd),torch.device('cuda'),False)
    print(json.dumps(dict(actual_batch_pass=True,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                          peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)),flush=True)
