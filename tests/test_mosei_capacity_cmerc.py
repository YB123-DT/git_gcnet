from dataclasses import asdict, make_dataclass, replace
from pathlib import Path
import inspect

import pytest
from test_nested_cmerc_worker import config


def cfg():
    values=dict(asdict(config('CMUMOSEI')),osram_adapter_hidden_dim=0)
    cls=make_dataclass('Config',[(k,type(v)) for k,v in values.items()])
    return cls(**values)


def test_exact_eight_configs_keep_cmerc_lr_and_protocol():
    from experiments.osram_mosei_capacity_cmerc_20261024.worker import MODES,candidate_config
    assert list(MODES)==['f128','f384','f768','f256_n64','f256','f256_n128','f512_n128','f512_n256']
    original=cfg()
    for mode,(width,method) in MODES.items():
        assert asdict(candidate_config(original,mode))==dict(asdict(original),
            osram_adapter_hidden_dim=width,osram_meaningful_block=method)
    for change in (dict(learning_rate=.001),dict(seed=67),dict(epochs=100),
                   dict(missing_mask_protocol='conversation-mask-v2')):
        with pytest.raises(ValueError):candidate_config(replace(original,**change),'f256')


def test_new_queue_has_eight_fresh_jobs_no_external_s0_adoption():
    from experiments.osram_mosei_capacity_cmerc_20261024.run import plan_jobs
    from experiments.osram_mosei_flat_nested_capacity_20261024.run import GPU0,GPU1
    jobs=plan_jobs(Path('/new'),'sha')
    assert len(jobs)==8
    assert [j['gpu_uuid'] for j in jobs]==[GPU0]*4+[GPU1]*4
    assert all('cmerc-repro' in j['command'][0] and '--resume' not in j['command'] for j in jobs)
    assert all(j['command'][j['command'].index('-m')+1]=='experiments.osram_mosei_capacity_cmerc_20261024.worker' for j in jobs)


def test_shared_worker_defaults_still_preserve_old_scope():
    from experiments.osram_nested_cmerc_tuning_20261023.worker import main,setup,validate_scope
    sig=inspect.signature(main)
    assert sig.parameters['setup_function'].default is None
    assert sig.parameters['scope_validator'].default is None
    assert sig.parameters['method_choices'].default is None
    assert sig.parameters['additional_source_files'].default==()
    with pytest.raises(ValueError):validate_scope('CMUMOSEI','mosei_capacity_nested256',66,1)
