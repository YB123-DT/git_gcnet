from types import SimpleNamespace
import pytest
from test_nested_cmerc_worker import config


class Optimizer:
    def __init__(self, params):
        self.param_groups = [dict(params=list(params), lr=1e-4, name='backbone')]
    def add_param_group(self, group):
        self.param_groups.append(group)


def fixture():
    backbone, nested = [object(),object()], [object(),object(),object()]
    model = SimpleNamespace(parameters=lambda:iter(backbone+nested),
        osram=SimpleNamespace(meaningful_block=SimpleNamespace(parameters=lambda:iter(nested))))
    task = SimpleNamespace(make_optimizer=lambda model,cfg,mode:Optimizer(model.parameters()))
    cfg = config('CMUMOSEI');cfg.osram_meaningful_block='nested_dim32'
    return model,cfg,task,backbone,nested


def test_optin_lr_partitions_every_parameter_and_preserves_backbone_lr():
    from experiments.osram_nested_cmerc_tuning_20261023 import worker
    model,cfg,task,backbone,nested=fixture()
    for rate in (5e-5,2e-4):
        opt=worker.make_task_optimizer(model,cfg,task,rate)
        assert len(opt.param_groups)==2
        assert opt.param_groups[0]==dict(params=backbone,lr=1e-4,name='backbone')
        assert opt.param_groups[1]==dict(params=nested,lr=rate,name='nested')
        assigned=[id(p) for g in opt.param_groups for p in g['params']]
        assert len(assigned)==len(set(assigned))==5
    default=worker.make_task_optimizer(model,cfg,task)
    assert default.param_groups==[dict(params=backbone+nested,lr=1e-4,name='backbone')]


def test_lr_scope_rejects_unapproved_values_methods_or_datasets():
    from experiments.osram_nested_cmerc_tuning_20261023 import worker
    model,cfg,task,_,_=fixture()
    for rate in (0.,1e-3,float('nan')):
        with pytest.raises(ValueError):worker.make_task_optimizer(model,cfg,task,rate)
    cfg.osram_meaningful_block='nested_dim128'
    with pytest.raises(ValueError):worker.make_task_optimizer(model,cfg,task,5e-5)
    cfg.osram_meaningful_block='nested_dim32';cfg.dataset='IEMOCAPSix'
    with pytest.raises(ValueError):worker.make_task_optimizer(model,cfg,task,5e-5)


def test_exact_four_experiments_no_extra_flat_or_global_lr(tmp_path):
    from experiments.osram_mosei_nested_local_20261024 import run
    jobs=run.plan_jobs(tmp_path,'sha',run.GPU)
    assert [(j['variant'],j['method'],j['nested_learning_rate']) for j in jobs]==[
        ('nested32_lr5e5','nested_dim32',5e-5),
        ('nested32_lr2e4','nested_dim32',2e-4),
        ('nested32_depth2','mosei_nested_dim32_depth2',None),
        ('nested32_groups4','mosei_nested_d32_l3_g4_m32',None)]
    for job in jobs:
        assert job['learning_rate']==1e-4 and job['epochs']==60 and job['seed']==66
        assert job['dataset']=='CMUMOSEI' and job['fold']==1
        assert ('--nested-learning-rate' in job['command'])==(job['nested_learning_rate'] is not None)
    assert len({j['output'] for j in jobs})==4
    with pytest.raises(ValueError):run.plan_jobs(tmp_path,'sha','GPU-bad')


def test_structural_choices_keep_mlp32_and_old_scope():
    from gcnet_missing_m3.meaningful_new40_registry import NESTED_SWEEP
    from experiments.osram_nested_cmerc_tuning_20261023 import worker
    assert NESTED_SWEEP['mosei_nested_d32_l3_g4_m32']==dict(dim=32,depth=3,groups=4,mlp_hidden=32)
    for method in ('mosei_nested_dim32_depth2','mosei_nested_d32_l3_g4_m32'):
        worker.validate_scope('CMUMOSEI',method,66,1)
        with pytest.raises(ValueError):worker.validate_scope('IEMOCAPSix',method,66,5)
    assert len(worker.METHODS)==7
