from dataclasses import asdict, replace
from pathlib import Path

import pytest
from test_mosei_flat256 import baseline


def test_seven_new_arms_only_change_width_and_nested_switch():
    from experiments.osram_mosei_flat_nested_capacity_20261024.worker import MODES, candidate_config
    assert MODES == {'f128':(128,'none'),'f384':(384,'none'),'f768':(768,'none'),
        'f256_n64':(256,'nested_gnn_rooted_evidence'),'f256_n128':(256,'nested_dim128'),
        'f512_n128':(512,'nested_dim128'),'f512_n256':(512,'mosei_capacity_nested256')}
    cfg=baseline()
    for mode,(width,method) in MODES.items():
        assert asdict(candidate_config(cfg,mode)) == dict(asdict(cfg),
            osram_adapter_hidden_dim=width,osram_meaningful_block=method)
    for cfg2 in (replace(cfg,seed=67),replace(cfg,learning_rate=1e-4),
                 replace(cfg,missing_mask_protocol='conversation-mask-v2')):
        with pytest.raises(ValueError):candidate_config(cfg2,'f128')


def test_seven_jobs_plus_existing256_use_only_physical_zero_one():
    from experiments.osram_mosei_flat_nested_capacity_20261024.run import plan_jobs, GPU0, GPU1
    jobs=plan_jobs(Path('/output'),'sha')
    assert len(jobs)==7 and len({j['output'] for j in jobs})==7
    assert [j['gpu_uuid'] for j in jobs]==[GPU0]*4+[GPU1]*3
    assert all(j['mode']!='flat256' for j in jobs)
    assert all('--seed' in j['command'] and j['command'][j['command'].index('--seed')+1]=='66' for j in jobs)
    assert all(j['command'][j['command'].index('--phase')+1]=='train' for j in jobs)


def test_nested256_is_only_a_registered_width_not_a_new_algorithm():
    from gcnet_missing_m3.meaningful_new40_registry import NESTED_SWEEP
    assert NESTED_SWEEP['mosei_capacity_nested256']==dict(dim=256)
