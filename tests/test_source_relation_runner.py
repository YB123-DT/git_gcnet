from types import SimpleNamespace
from dataclasses import dataclass
import pytest

from experiments.osram_source_relation_attention_20261021.worker import settings, set_learning_rates


def test_only_two_seed66_dataset_budgets_and_original_lrs():
    assert settings('CMUMOSI') == dict(epochs=150, first_lr=3e-4, final_lr=3e-5)
    assert settings('CMUMOSEI') == dict(epochs=60, first_lr=1e-3, final_lr=1e-3)
    op = SimpleNamespace(param_groups=[dict(name='backbone', lr=0.)])
    for dataset, boundary, expected in [('CMUMOSI', 100, 3e-5), ('CMUMOSEI', 59, 1e-3)]:
        set_learning_rates(op, dataset, 0)
        assert op.param_groups[0]['lr'] == settings(dataset)['first_lr']
        set_learning_rates(op, dataset, boundary)
        assert op.param_groups[0]['lr'] == expected
    with pytest.raises(ValueError): settings('MELD')
    with pytest.raises(ValueError): set_learning_rates(op, 'CMUMOSEI', 60)


def test_reference_config_normalizes_only_reviewed_legacy_disabled_fields():
    from experiments.osram_source_relation_attention_20261021.worker import validate_reference_config
    @dataclass
    class Config:
        seed: int = 66
        osram_source_role: bool = False
        osram_source_role_mode: str = 'residual'
        osram_source_relation: bool = True
    validate_reference_config(Config(), {'seed': 66})
    with pytest.raises(ValueError): validate_reference_config(Config(), {'seed': 67})


def test_output_layer_only_is_not_successful_gradient_coverage():
    from experiments.osram_source_relation_attention_20261021.worker import validate_gradients
    with pytest.raises(RuntimeError): validate_gradients({'output_projection.weight': True})
    observed = {k: True for k in ('base_projection.weight', 'gap_projections.0.weight',
        'relation_mlp.1.weight', 'source_self_attention.in_proj_weight',
        'local_projection.weight', 'local_cross_attention.in_proj_weight',
        'source_embedding', 'output_projection.weight', 'input.source_base', 'input.source_gap')}
    validate_gradients(observed)
