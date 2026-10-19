"""From-scratch joint TASR must not accidentally use the frozen adapter runner."""
import copy
import importlib
from dataclasses import replace
from types import SimpleNamespace

import torch
import pytest


def api():
    return importlib.import_module('experiments.osram_tasr_joint_20261019.run')


def test_joint_configuration_only_enables_tasr_and_declared_validation_protocol():
    from experiments.osram_tasr_20261018.protocol import build_cases
    base = build_cases()['MELD']['expected_config']
    cfg = api().make_config()
    from dataclasses import asdict
    from gcnet_missing_m3.train_gcnet import TrainConfig
    assert asdict(cfg) == asdict(TrainConfig(**dict(base, osram_source_tracking=True, osram_tasr=True,
        osram_tasr_source_mode='real', checkpoint_selection='validation',
        preserve_selection_artifacts=False)))
    assert cfg.initial_backbone_checkpoint is None
    assert cfg.epochs==50 and cfg.learning_rate==1e-5
    assert cfg.paired_history_views is False and cfg.training_objective=='emotion-only'


def test_fresh_initialization_matches_flat_and_joint_gradients_update_original_and_tasr():
    module = api()
    cfg = replace(module.make_config(), latent_dim=8, osram_output_dim=10,
        osram_num_heads=2, osram_key_dim=3, osram_value_dim=4,
        dropout=0.,projector_dropout=0.)
    model = module.fresh_model(cfg,(4,5,6),torch.device('cpu'))
    state = torch.get_rng_state().clone()
    baseline = module.fresh_model(replace(cfg,osram_source_tracking=False,osram_tasr=False),
                                  (4,5,6),torch.device('cpu'))
    assert torch.equal(state,torch.get_rng_state())
    for name,value in baseline.state_dict().items():
        assert torch.equal(value,model.state_dict()[name]),name
    assert all(p.requires_grad for n,p in model.named_parameters() if not n.startswith('teacher.'))
    inputs=torch.randn(4,2,15)
    availability=torch.tensor([[[1,1,1],[1,1,1]],[[1,0,1],[1,0,0]],
                               [[0,0,1],[0,1,0]],[[1,0,0],[1,0,0]]]).float()
    umask=torch.ones(2,4); speakers=torch.zeros(2,4,dtype=torch.long)
    args=([inputs],availability,speakers,umask,[4,4])
    model.eval();baseline.eval()
    torch.testing.assert_close(model(*args,predict_missing=False)[0],
                               baseline(*args,predict_missing=False)[0],rtol=0,atol=0)
    optimizer,_=module.make_optimizer(model,cfg)
    assert {id(p) for g in optimizer.param_groups for p in g['params']} == {
        id(p) for p in model.parameters() if p.requires_grad}
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    labels=torch.arange(8).remainder(7)
    model.train()
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        logits=model(*args,predict_missing=False)[0]
        torch.nn.functional.cross_entropy(logits.reshape(-1,7),labels).backward()
        report=module.gradient_report(model)
        assert report['finite']
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
    changed=[n for n,p in model.named_parameters() if not torch.equal(before[n],p)]
    for prefix in ('observed_set.','osram.key_projectors.','osram.query_projector.',
                   'osram.emotion_adapter.','smax_fc.','osram.tasr.'):
        assert any(n.startswith(prefix) for n in changed),prefix
    payload=module.checkpoint_payload(model,optimizer,epoch=4,identity={'run':'x'},history=[],best_key=(1.,0.))
    assert set(payload['model'])==set(model.state_dict())
    expected=copy.deepcopy(model.state_dict())
    with torch.no_grad():next(model.parameters()).add_(1)
    module.restore_checkpoint(model,optimizer,payload,identity={'run':'x'})
    for name,value in expected.items():assert torch.equal(value,model.state_dict()[name])


@pytest.mark.parametrize('smoke',[False,True])
def test_original_trainer_joint_epoch_is_single_random_view(smoke):
    module=api()
    from gcnet_missing_m3 import train_gcnet as trainer
    cfg=replace(module.make_config(),latent_dim=8,osram_output_dim=10,
        osram_num_heads=2,osram_key_dim=3,osram_value_dim=4,dropout=0.,projector_dropout=0.)
    model=module.fresh_model(cfg,(4,5,6),torch.device('cpu'))
    raw=[torch.randn(4,2,d) for d in (4,5,6,4,5,6)]
    raw += [torch.zeros(2,4,dtype=torch.long),torch.ones(2,4),
            torch.arange(8).reshape(2,4).remainder(7),['one','two']]
    class Loader:
        sampler=SimpleNamespace(set_epoch=lambda epoch:None)
        dataset=[None,None]
        def __iter__(self):return iter([raw,raw])
    optimizer,_=module.make_optimizer(model,cfg)
    report=module.run_epoch(dict(model=model,cfg=cfg,trainer=trainer,
        dimensions=(4,5,6),train_loader=Loader()),optimizer,0,smoke=smoke)
    assert report['model_forward_count']==report['optimizer_steps']==report['random_batches']==2
    assert report['jepa_loss']==0
    assert model.training
