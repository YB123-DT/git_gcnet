"""Actual-dimension GPU check against pre-change training source; no benchmark."""
import argparse
import copy
from dataclasses import fields, replace
import importlib.util
import json
from pathlib import Path
import sys
import torch
from gcnet_missing_m3 import train_gcnet as current
from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--legacy', required=True)
    args = parser.parse_args()
    name = 'gcnet_missing_m3.pre_paired_verification'
    spec = importlib.util.spec_from_file_location(name, args.legacy)
    legacy = importlib.util.module_from_spec(spec)
    sys.modules[name] = legacy
    spec.loader.exec_module(legacy)
    raw = json.loads(Path('/data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full/seed_66/config.json').read_text())
    config = current.TrainConfig(**raw)
    old_config = legacy.TrainConfig(**raw)
    base = _build_model(config, (512,1024,1024)).cuda()
    new = copy.deepcopy(base)
    torch.manual_seed(13)
    batch = [torch.randn(5,2,d) for d in (512,1024,1024,512,1024,1024)]
    batch += [torch.zeros(2,5),torch.tensor([[1.,1,1,1,1],[1.,1,1,0,0]]),
              torch.tensor([[1.,-1,1,-1,1],[-1.,1,-1,0,0]]),['a','b']]
    for module, model, cfg in ((legacy,base,old_config),(current,new,config)):
        torch.manual_seed(81)
        result=module.train_epoch(model,[batch,batch],torch.optim.Adam(model.parameters(),lr=.001),
            cfg,module._schedules(cfg,'train'),0,(512,1024,1024),torch.device('cuda'))
        if module is legacy:
            expected=result
        else:
            assert result == expected, 'disabled training metrics differ'
    for key, value in base.state_dict().items():
        assert torch.equal(value,new.state_dict()[key]), key
    paired_cfg=replace(config,paired_history_views=True)
    current._attach_history_projector(new,paired_cfg)
    assert new.history_contrast_projector[0].in_features == 1600
    before=new.history_contrast_projector[0].weight.detach().clone()
    groups,_=current._optimizer_parameter_groups(new,paired_cfg)
    optimizer=torch.optim.Adam(groups,weight_decay=paired_cfg.weight_decay)
    result=current.train_epoch(new,[batch,batch],optimizer,paired_cfg,current._schedules(paired_cfg,'train'),
        0,(512,1024,1024),torch.device('cuda'))
    assert result['paired_history']['eligible_anchor_count'] > 0
    assert not torch.equal(before,new.history_contrast_projector[0].weight)
    assert all(torch.isfinite(p).all() for p in new.parameters())
    print(json.dumps(dict(default_off_metrics_and_updated_weights_bitexact=True,
        actual_dimensions=[512,1024,1024],projector_input=1600,
        projector_updated=True,paired_history=result['paired_history']),indent=2))


if __name__ == '__main__': main()
