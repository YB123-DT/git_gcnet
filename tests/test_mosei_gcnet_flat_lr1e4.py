from dataclasses import asdict, replace
import importlib

import pytest


def worker():
    return importlib.import_module('experiments.osram_mosei_gcnet_flat_lr1e4_20261021.worker')


@pytest.mark.parametrize('seed',[66,67,68])
def test_only_seed_and_learning_rate_change_the_current_flat_config(seed):
    from gcnet_missing_m3.train_gcnet import TrainConfig
    cfg=replace(TrainConfig(),dataset='CMUMOSEI',fold=1,epochs=60,learning_rate=.001,
        backbone_type='osram',osram_bidirectional=False,osram_output_dim=1600,osram_write_step=.6,
        osram_num_heads=8,osram_key_dim=64,osram_value_dim=64,training_objective='emotion-only',
        checkpoint_selection='test-oracle-per-rate',checkpoint_selection_metric='weighted_f1',
        missing_mask_protocol='gcnet-random-mask-v1')
    changed=worker().low_lr_config(cfg,seed)
    assert asdict(changed)==dict(asdict(cfg),seed=seed,learning_rate=1e-4)
    with pytest.raises(ValueError):worker().low_lr_config(replace(cfg,missing_mask_protocol='conversation-mask-v2'),seed)
    with pytest.raises(ValueError):worker().low_lr_config(replace(cfg,osram_source_role=True),seed)


def test_plan_only_runs_flat67_and68_with_no_other_models_or_seeds(tmp_path):
    from experiments.osram_mosei_gcnet_flat_lr1e4_20261021.run import plan_jobs
    jobs=plan_jobs(tmp_path,'sha',['GPU-good'])
    assert [j['seed'] for j in jobs]==[67,68]
    assert all(j['dataset']=='CMUMOSEI' and j['mode']=='flat' for j in jobs)
    assert len({j['output'] for j in jobs})==2
    assert all(j['command'][2]=='experiments.osram_mosei_gcnet_flat_lr1e4_20261021.worker' for j in jobs)


def test_invalid_setup_is_not_implicitly_enabled():
    from gcnet_missing_m3.train_gcnet import TrainConfig
    with pytest.raises(ValueError):worker().low_lr_config(TrainConfig(),66)
