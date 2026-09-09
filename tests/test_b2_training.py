import importlib
from dataclasses import asdict

import pytest
import torch

from gcnet_missing_m3.model import MissingM3GraphModel, MODALITIES
from gcnet_missing_m3.train_gcnet import TrainConfig
from test_b2_completion import model_kwargs, inputs


def module():
    assert importlib.util.find_spec("gcnet_missing_m3.b2_training"), "B2 stage transfer is missing"
    return importlib.import_module("gcnet_missing_m3.b2_training")


def checkpoint(tmp_path):
    model=MissingM3GraphModel(**model_kwargs())
    cfg=TrainConfig(dataset="CMUMOSI",fold=1,latent_dim=8,hidden=4,num_experts=2,top_k=1,
        dropout=0.,projector_dropout=0.,predictor_dropout=0.,backbone_type="osram",
        osram_output_dim=10,osram_num_heads=2,osram_key_dim=3,osram_value_dim=4,
        osram_bidirectional=False,osram_write_step=.6,device="cpu")
    path=tmp_path/"base.pt"
    torch.save(dict(model=model.state_dict(),config=asdict(cfg),epoch=30),path)
    return path,cfg


def test_stage1_freeze_no_osram_target_leak_and_transfer(tmp_path):
    b=module()
    path,cfg=checkpoint(tmp_path)
    stage=b.SourceOnlyPretrainer.from_checkpoint(path)
    stage.train()
    assert not hasattr(stage,"osram")
    assert not hasattr(stage,"smax_fc")
    assert not stage.projectors.training and not stage.teacher.training
    assert all(not p.requires_grad for p in stage.projectors.parameters())
    assert all(not p.requires_grad for p in stage.teacher.parameters())
    _,a,u=inputs()
    complete=torch.randn(8,1,12)
    expanded=torch.cat([a[...,i:i+1].expand(-1,-1,w) for i,w in enumerate((3,4,5))],-1)
    masked=complete*expanded
    stage.eval()
    first=stage(masked,a,u)
    changed=masked+(1-expanded)*torch.randn_like(masked)*100
    second=stage(changed,a,u)
    assert torch.equal(first.reg_predictions,second.reg_predictions)
    assert not torch.equal(stage.targets(complete)["text"],stage.targets(complete+torch.randn_like(complete))["text"])
    before={k:v.clone() for k,v in stage.state_dict().items() if not k.startswith("predictor.")}
    optimizer=torch.optim.Adam(stage.predictor.parameters(),lr=.001)
    result=b.pretrain_step(stage,dict(incomplete=masked,complete=complete,availability=a,umask=u),optimizer,cfg)
    assert result["target_count"]>0 and result["gradient_norm"]>0
    assert all(torch.equal(v,stage.state_dict()[k]) for k,v in before.items())
    saved=tmp_path/"source_only_completion_pretrain.pt"
    stage.save(saved,epoch=1,optimizer=optimizer)
    b2=MissingM3GraphModel(**model_kwargs(),completion_path="pre_osram_b2")
    provenance=b.load_b2_initialization(b2,path,saved)
    assert provenance["source_checkpoint_sha256"]==b.file_sha256(path)
    assert all(torch.equal(v,b2.source_only_predictor.state_dict()[k]) for k,v in stage.predictor.state_dict().items())
    assert all(torch.equal(v,b2.observed_set.projectors.state_dict()[k]) for k,v in stage.projectors.state_dict().items())
    assert all(p.requires_grad for p in b2.source_only_predictor.parameters())
    assert all(not p.requires_grad for p in b2.teacher.parameters())
    changed_routing=MissingM3GraphModel(**(model_kwargs() | {"top_k":2}),completion_path="pre_osram_b2")
    with pytest.raises(ValueError,match="config"):
        b.load_b2_initialization(changed_routing,path,saved)
    changed_ridge=MissingM3GraphModel(**(model_kwargs() | {"osram_read_ridge":.2}),completion_path="pre_osram_b2")
    with pytest.raises(ValueError,match="config"):
        b.load_b2_initialization(changed_ridge,path,saved)
    bad=torch.load(saved,weights_only=False)
    bad["source_checkpoint_sha256"]="wrong"
    torch.save(bad,saved)
    with pytest.raises(ValueError,match="provenance"):
        b.load_b2_initialization(b2,path,saved)


def test_stage1_complete_batch_skips_update(tmp_path):
    b=module()
    path,cfg=checkpoint(tmp_path)
    stage=b.SourceOnlyPretrainer.from_checkpoint(path)
    x=torch.randn(2,1,12)
    view=dict(incomplete=x,complete=x,availability=torch.ones(2,1,3),umask=torch.ones(1,2))
    before={k:v.clone() for k,v in stage.predictor.state_dict().items()}
    result=b.pretrain_step(stage,view,torch.optim.Adam(stage.predictor.parameters()),cfg)
    assert result["target_count"]==0 and result["skipped"]
    assert all(torch.equal(v,stage.predictor.state_dict()[k]) for k,v in before.items())


def test_b2_runner_requires_explicit_stage_and_provenance(tmp_path):
    assert importlib.util.find_spec("gcnet_missing_m3.train_b2"), "B2 runner missing"
    runner=importlib.import_module("gcnet_missing_m3.train_b2")
    args=runner.build_parser().parse_args(["--stage","smoke","--base-checkpoint","base.pt",
                                         "--feature-root","features","--output-dir",str(tmp_path)])
    assert args.stage=="smoke"
    with pytest.raises(SystemExit):
        runner.build_parser().parse_args([])
