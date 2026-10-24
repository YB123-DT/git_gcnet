import pytest


def test_explicit_variant_scope_does_not_expand_existing_shared_worker():
    from experiments.osram_mosei_flat128_nested_20261024.worker import validate_scope
    validate_scope('F67', 'CMUMOSEI', 'none', 67, 1)
    validate_scope('N01', 'CMUMOSEI', 'mosei_f128_n01', 66, 1)
    for values in [('F67','CMUMOSEI','none',66,1),
                   ('N01','CMUMOSEI','mosei_f128_n01',67,1),
                   ('N01','CMUMOSI','mosei_f128_n01',66,1),
                   ('N01','CMUMOSEI','mosei_f128_n01',66,5)]:
        with pytest.raises(ValueError):
            validate_scope(*values)
