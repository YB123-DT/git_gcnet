from dataclasses import asdict, replace

import pytest
from test_mosei_capacity_cmerc import cfg


def test_exact_twenty_one_structures_and_seed_slots():
    from experiments.osram_mosei_flat128_nested_20261024.config import VARIANTS, run_spec
    assert len(VARIANTS) == 21
    assert VARIANTS['N01'] == dict(dim=32, depth=2, groups=2, mlp_hidden=128)
    assert VARIANTS['N18'] == dict(dim=128, depth=3, groups=8, mlp_hidden=128)
    assert VARIANTS['N19'] == dict(dim=32, depth=3, groups=8, mlp_hidden=32)
    assert VARIANTS['N20'] == dict(dim=64, depth=3, groups=8, mlp_hidden=64)
    assert VARIANTS['N21'] == dict(dim=128, depth=3, groups=8, mlp_hidden=256)
    assert len({tuple(row.values()) for row in VARIANTS.values()}) == 21
    for seed in (66, 67, 68):
        assert run_spec(f'F{seed}') == dict(method='none', seed=seed)
    for name in VARIANTS:
        assert run_spec(name) == dict(method='mosei_f128_'+name.lower(), seed=66)


def test_config_changes_only_width_method_and_named_seed():
    from experiments.osram_mosei_flat128_nested_20261024.config import candidate_config, VARIANTS, run_spec
    for variant in ('F66','F67','F68', *VARIANTS):
        base = cfg()
        spec = run_spec(variant)
        assert asdict(candidate_config(base, variant)) == dict(asdict(base),
            seed=spec['seed'], osram_adapter_hidden_dim=128, osram_meaningful_block=spec['method'])
    for changes in (dict(learning_rate=.001), dict(osram_write_step=.4),
                    dict(missing_mask_protocol='conversation-mask-v2'), dict(osram_adapter_hidden_dim=256)):
        with pytest.raises(ValueError):
            candidate_config(replace(cfg(), **changes), 'N01')
    with pytest.raises(ValueError):
        candidate_config(cfg(), 'N22')


def test_new_registry_exactly_matches_planned_existing_algorithm_options():
    from gcnet_missing_m3.meaningful_new40_registry import NESTED_SWEEP
    from experiments.osram_mosei_flat128_nested_20261024.config import VARIANTS, method_name
    for variant, options in VARIANTS.items():
        assert NESTED_SWEEP.get(method_name(variant)) == options
