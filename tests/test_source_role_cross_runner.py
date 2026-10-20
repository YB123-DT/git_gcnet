import importlib
from dataclasses import asdict, replace

import pytest


def module(name):
    return importlib.import_module('experiments.osram_source_role_direct_cross_20261020.' + name)


def test_finite_plan_has_all_three_seeds_and_all_five_iemocap_folds(tmp_path):
    jobs = module('run').plan_jobs(tmp_path, 'sha', 'GPU-good')
    expected = {(d, s, f) for d, folds in (('CMUMOSEI', (1,)), ('MELD', (1,)),
                ('IEMOCAPSix', range(1, 6))) for s in (66, 67, 68) for f in folds}
    assert {(j['dataset'], j['seed'], j['fold']) for j in jobs} == expected
    assert len(jobs) == len({j['output'] for j in jobs}) == 21
    assert all(j['gpu_uuid'] == 'GPU-good' for j in jobs)
    for job in jobs:
        command = job['command']
        assert command[command.index('--dataset') + 1] == job['dataset']
        assert command[command.index('--fold') + 1] == str(job['fold'])
        assert 's0/bin/python' in command[0] if job['dataset'] == 'CMUMOSEI' else 'cmerc-repro/bin/python' in command[0]


def test_bad_gpu_and_unapproved_folds_are_rejected(tmp_path):
    from experiments.osram_mosi_gcnet_router_3seed_20261019.run import BAD_UUID
    with pytest.raises(ValueError): module('run').plan_jobs(tmp_path, 'sha', BAD_UUID)
    for d, s, f in [('MELD', 66, 5), ('CMUMOSI', 66, 1), ('IEMOCAPSix', 65, 1), ('IEMOCAPSix', 66, 6)]:
        with pytest.raises(ValueError): module('worker').settings(d, s, f)


@pytest.mark.parametrize('dataset,epochs,lr,metric', [('CMUMOSEI',60,.001,'weighted_f1'),
    ('MELD',50,1e-5,'accuracy'),('IEMOCAPSix',100,.001,'accuracy')])
def test_only_source_role_changes_accepted_config(dataset,epochs,lr,metric):
    from gcnet_missing_m3.train_gcnet import TrainConfig
    cfg = replace(TrainConfig(), dataset=dataset, fold=1, epochs=epochs, learning_rate=lr,
        weight_decay=1e-5, backbone_type='osram', osram_bidirectional=False,
        osram_output_dim=1600, osram_write_step=.6, osram_num_heads=8,
        osram_key_dim=64, osram_value_dim=64,
        training_objective='emotion-only', missing_mask_protocol='gcnet-random-mask-v1',
        checkpoint_selection='test-oracle-per-rate', checkpoint_selection_metric=metric)
    new = module('worker').direct_config(cfg)
    assert asdict(new) == dict(asdict(cfg), osram_source_role=True, osram_source_role_mode='direct')
    assert not new.osram_disable_local_skip and not new.paired_history_views
    with pytest.raises(ValueError): module('worker').direct_config(replace(cfg, checkpoint_selection='validation'))
    with pytest.raises(ValueError): module('worker').direct_config(replace(cfg, osram_tasr=True))


def test_admission_measures_headroom_without_silently_reducing_batch():
    admit = module('run').admission_limit
    assert admit(32768, 6, 2048, 6) == 6
    assert admit(32768, 27000, 2048, 6) == 1
    assert admit(32768, 32000, 2048, 6) == 0
    assert admit(32768, 6, 12000, 6) == 1


def test_meld_keeps_resampled_eval_and_other_datasets_keep_their_fixed_banks():
    settings = module('worker').settings
    assert settings('MELD',66,1)['test_masks'] == 'original-per-epoch-resampled'
    assert settings('IEMOCAPSix',68,5)['test_masks'] == 'original-fixed-GCNet-bank'
    assert settings('CMUMOSEI',67,1)['test_masks'] == 'original-fixed-GCNet-bank'
