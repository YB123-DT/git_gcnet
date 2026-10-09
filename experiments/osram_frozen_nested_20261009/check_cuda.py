"""Locate CUDA zero-init discrepancies before any training; no writes or selection."""
import json
import os
from dataclasses import asdict
import numpy as np
import torch
from experiments.osram_frozen_nested_20261009.dispatch import REFERENCE, DATA
from experiments.osram_frozen_nested_20261009.run import load_parent, freeze_parent
from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model
from gcnet_missing_m3.train_gcnet import TrainConfig, evaluate_rate, _schedules
from gcnet_modality_jepa.train_gcnet import get_loaders

torch.set_num_threads(2)
data = json.load(open(DATA))
os.environ['GCNET_DATASET_ROOT'] = data['dataset_root']
c = TrainConfig(**json.load(open(REFERENCE)))
roots = data['feature_roots']
_, _, test, ad, td, vd = get_loaders(audio_root=roots[0], text_root=roots[1], video_root=roots[2],
    num_folder=1, dataset=c.dataset, batch_size=c.batch_size, num_workers=0, seed=c.seed,
    validation_fraction=c.validation_fraction, evaluation_protocol=c.evaluation_protocol)
dims = (ad,td,vd)
state = torch.load(REFERENCE.replace('config.json', 'best_miss_0p7.pt'), map_location='cpu', weights_only=False)['model']
flat = _build_model(c,dims).cuda()
flat.load_state_dict(state)
nested = _build_model(TrainConfig(**dict(asdict(c), osram_meaningful_block='nested_gnn_rooted_evidence')),dims).cuda()
load_parent(nested,state)
freeze_parent(nested)
records = {}
def capture(name):
    def hook(module, inputs, output):
        if name not in records:
            records[name] = ([v.detach().cpu() for v in inputs if torch.is_tensor(v)],
                             output.detach().cpu(), [tuple(v.stride()) for v in inputs if torch.is_tensor(v)])
    return hook
for name, net in [('flat',flat), ('nested',nested)]:
    for part in ('local_skip','emotion_adapter','emotion_norm'):
        getattr(net.osram, part).register_forward_hook(capture(name+'.'+part))
outputs = []
for net in (flat,nested):
    outputs.append(evaluate_rate(net,test[0],_schedules(c,'test')[0.0],c.dataset,dims,torch.device('cuda'),True,
                               c.mosi_task_mode,c.task_regression_loss,c.task_smooth_l1_beta))
for part in ('local_skip','emotion_adapter','emotion_norm'):
    a,b = records['flat.'+part],records['nested.'+part]
    print(part, 'input maxdiff', [(x-y).abs().max().item() for x,y in zip(a[0],b[0])],
          'strides',a[2],b[2], 'output maxdiff',(a[1]-b[1]).abs().max().item())
a,b = outputs[0][1],outputs[1][1]
print('labels_equal',np.array_equal(a['labels'],b['labels']), 'mask_equal',np.array_equal(a['availability'],b['availability']))
print('pred maxdiff',abs(a['predictions']-b['predictions']).max(), 'sign differences',((a['predictions']>0)!=(b['predictions']>0)).sum())
print('WF1',[o[0]['weighted_f1'] for o in outputs])
