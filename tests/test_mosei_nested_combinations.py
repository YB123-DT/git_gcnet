from dataclasses import asdict
import pytest
from test_nested_cmerc_worker import config


EXPECTED = {
    'mosei_nested_d32_l3_g8_m128': dict(dim=32, depth=3, groups=8, mlp_hidden=128),
    'mosei_nested_d32_l3_g8_m64': dict(dim=32, depth=3, groups=8, mlp_hidden=64),
    'mosei_nested_d32_l2_g8_m128': dict(dim=32, depth=2, groups=8, mlp_hidden=128),
    'mosei_nested_d32_l3_g4_m128': dict(dim=32, depth=3, groups=4, mlp_hidden=128),
}


def test_exact_registered_combinations_and_scope():
    from gcnet_missing_m3.meaningful_new40_registry import NESTED_SWEEP
    from experiments.osram_nested_cmerc_tuning_20261023 import worker
    for method, options in EXPECTED.items():
        assert NESTED_SWEEP.get(method) == options
        original = config('CMUMOSEI')
        changed = asdict(worker.candidate_config(original, method))
        expected = asdict(original); expected['osram_meaningful_block'] = method
        assert changed == expected
        for dataset, seed, fold in [('IEMOCAPSix',66,5), ('CMUMOSEI',67,1), ('CMUMOSEI',66,2)]:
            with pytest.raises(ValueError):
                worker.validate_scope(dataset, method, seed, fold)
    assert len(worker.METHODS) == 7  # no expansion of historical screen/report


def test_four_jobs_gpu2_only_and_no_flat_or_extra_seed(tmp_path):
    from experiments.osram_mosei_nested_combinations_20261024 import run
    jobs = run.plan_jobs(tmp_path, 'sha', run.GPU)
    assert tuple(j['method'] for j in jobs) == tuple(EXPECTED)
    assert len({j['output'] for j in jobs}) == 4
    for j in jobs:
        assert (j['dataset'],j['seed'],j['fold'],j['epochs'],j['learning_rate']) == ('CMUMOSEI',66,1,60,1e-4)
        assert j['gpu_uuid'] == 'GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4'
    with pytest.raises(ValueError):
        run.plan_jobs(tmp_path, 'sha', run.BAD_UUID)


def test_four_worker_reservation_and_resource_boundaries():
    from experiments.osram_mosei_nested_combinations_20261024 import run
    resources = dict(total_mib=32768,used_mib=15000,own_used_mib=15000,
                     active_count=3,ram_gib=100,disk_gib=100)
    assert run.can_admit(**resources)
    assert not run.can_admit(**dict(resources,active_count=4))
    assert not run.can_admit(**dict(resources,used_mib=27000))
    assert not run.can_admit(**dict(resources,ram_gib=63))
    assert not run.can_admit(**dict(resources,disk_gib=24))
