import copy
from dataclasses import asdict, replace
import importlib
from types import SimpleNamespace

import pytest


def api():return importlib.import_module('experiments.osram_coupled_source_role_20261020.worker')


def test_config_only_changes_the_new_switch():
    from gcnet_missing_m3.train_gcnet import TrainConfig
    cfg=replace(TrainConfig(),dataset='CMUMOSI',backbone_type='osram',training_objective='emotion-only',
        epochs=150,missing_mask_protocol='gcnet-random-mask-v1',checkpoint_selection='test-oracle-per-rate',
        checkpoint_selection_metric='weighted_f1',osram_output_dim=1600,osram_bidirectional=False)
    expected=asdict(cfg);expected['osram_source_role']=True
    assert asdict(api().source_role_config(cfg))==expected
    with pytest.raises(ValueError):api().source_role_config(replace(cfg,osram_tasr=True))


def test_original_dataset_lr_schedules_and_no_router_group():
    op=SimpleNamespace(param_groups=[dict(name='backbone',lr=.001)])
    for dataset in ('CMUMOSI','CMUMOSEI'):
        for epoch in ((0,99,100,149) if dataset=='CMUMOSI' else (0,59)):
            api().set_learning_rates(op,dataset,epoch)
            assert op.param_groups[0]['lr']==(.0001 if dataset=='CMUMOSI' and epoch>=100 else .001)
    with pytest.raises(ValueError):api().set_learning_rates(op,'CMUMOSEI',60)


def test_reference_comparison_rejects_any_other_configuration_change():
    module=api()
    current=dict(config=dict(osram_source_role=True,seed=66,learning_rate=.001),
        mask_hashes={'test':{'0.3':'same'}},split={'train':'official'},seed_protocol={'seed':66})
    old=dict(status='complete',epoch=150,initial_backbone_sha256='initial',
        identity=copy.deepcopy(current))
    del old['identity']['config']['osram_source_role']
    assert module.validate_reference(current,old,'initial',150)['initialization_exact']
    current['config']['learning_rate']=.00002
    with pytest.raises(ValueError,match='config'):module.validate_reference(current,old,'initial',150)


def test_finite_dataset_plan_has_one_seed66_job_and_no_bad_gpu(tmp_path):
    from experiments.osram_coupled_source_role_20261020.run import plan_jobs
    for dataset in ('CMUMOSI','CMUMOSEI'):
        jobs=plan_jobs(tmp_path,'sha',['GPU-healthy'],dataset)
        assert len(jobs)==1 and jobs[0]['seed']==66 and jobs[0]['dataset']==dataset
        assert jobs[0]['command'][2].endswith('.'+('mosi' if dataset=='CMUMOSI' else 'mosei'))
    with pytest.raises(ValueError):plan_jobs(tmp_path,'sha',['GPU-46fb379f-cc90-dc82-9b5e-5d011f552264'],'CMUMOSI')


def test_final_checkpoint_resume_has_publishable_selection_and_peak_fields():
    best={f'{i/10:.1f}':{'epoch':i+1} for i in range(8)}
    progress=api().progress_fields(150,best,4096.)
    assert progress['epoch']==150 and progress['best_by_rate']==best
    assert progress['selected_epoch_by_rate']=={k:v['epoch'] for k,v in best.items()}
    assert progress['peak_memory_mib']==4096.
