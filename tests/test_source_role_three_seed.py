import importlib
from pathlib import Path

import pytest


def test_source_reference_resolves_each_approved_mosi_seed_without_changing_seed66():
    common=importlib.import_module('experiments.osram_coupled_source_role_20261020.worker')
    assert hasattr(common,'reference_path')
    assert common.reference_path('CMUMOSI',66)==common.REFERENCES['CMUMOSI']
    for seed in (67,68):
        assert common.reference_path('CMUMOSI',seed)==common.REFERENCES['CMUMOSI'].parent.parent/f'seed_{seed}'/'flat'
    with pytest.raises(ValueError):common.reference_path('CMUMOSEI',67)
    with pytest.raises(ValueError):common.reference_path('CMUMOSI',65)


def test_followup_plan_only_adds_two_seeds_and_never_repeats_seed66(tmp_path):
    module=importlib.import_module('experiments.osram_mosi_source_role_direct_3seed_20261020.run')
    jobs=module.plan_jobs(tmp_path,'commit',['GPU-healthy'])
    assert [j['seed'] for j in jobs]==[67,68]
    assert len({j['output'] for j in jobs})==2
    assert all(j['dataset']=='CMUMOSI' and j['mode']=='joint' for j in jobs)
    assert all(j['command'][2]=='experiments.osram_mosi_source_role_direct_3seed_20261020.worker' for j in jobs)
    assert module.UPSTREAM==Path('/data2/yb/remote_experiments/osram_mosi_source_role_direct_20261020/execution_v1/QUEUE.json')


def test_three_seed_statistics_average_rates_within_seed_before_sample_std():
    module=importlib.import_module('experiments.osram_mosi_source_role_direct_3seed_20261020.report')
    rows=[dict(seed=seed,rate=f'{r/10:.1f}',WF1=base+r,ACC=base+r+.5,
        flat_WF1=base+r-1,flat_ACC=base+r-.5) for seed,base in ((66,70.),(67,72.),(68,74.)) for r in range(8)]
    stats=module.summarize_rows(rows)
    assert stats['mean']['WF1']==dict(mean=75.5,std=2.)
    assert stats['high']['WF1']==dict(mean=78.,std=2.)
    assert stats['by_rate']['0.7']['WF1']==dict(mean=79.,std=2.)
    assert stats['mean']['delta_WF1_pp']==dict(mean=1.,std=0.)
    with pytest.raises(ValueError):module.summarize_rows(rows[:-1])


def test_worker_opt_in_preserves_legacy_seed_restriction():
    import inspect
    common=importlib.import_module('experiments.osram_coupled_source_role_20261020.worker')
    assert 'allowed_seeds' in inspect.signature(common.main).parameters
    assert inspect.signature(common.main).parameters['allowed_seeds'].default==(66,)
