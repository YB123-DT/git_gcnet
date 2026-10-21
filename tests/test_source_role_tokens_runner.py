from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.osram_source_relation_attention_20261021 import worker


def test_reference_permits_only_explicit_new_layout_schema():
    @dataclass
    class Config:
        seed: int = 66
        osram_source_role: bool = False
        osram_source_role_mode: str = 'residual'
        osram_source_relation: bool = True
        osram_source_relation_layout: str = 'role-tokens'

    worker.validate_reference_config(Config(), {'seed': 66})
    with pytest.raises(ValueError):
        worker.validate_reference_config(Config(seed=67), {'seed': 66})
    with pytest.raises(ValueError):
        worker.validate_reference_config(Config(osram_source_relation_layout='unknown'), {'seed': 66})


def test_role_tokens_catalog_records_module_and_controller_sources():
    paths = worker.source_catalog(('model.py',), 'role-tokens')
    assert len(paths) == len(set(paths))
    assert 'gcnet_missing_m3/source_role_tokens.py' in paths
    assert 'experiments/osram_source_relation_attention_20261021/run.py' in paths
    assert 'experiments/osram_source_relation_attention_20261021/report.py' in paths
    assert 'gcnet_missing_m3/source_role_tokens.py' not in worker.source_catalog((), 'source-mean')


def test_role_tokens_gradient_validation_checks_embedding_and_every_active_slot():
    observed = {name: True for name in ('base_projection.weight', 'gap_projections.0.weight',
        'relation_mlp.1.weight', 'source_self_attention.in_proj_weight', 'local_projection.weight',
        'local_cross_attention.in_proj_weight', 'source_embedding', 'role_embedding',
        'output_projection.weight', 'input.source_base', 'input.source_gap')}
    observed.update({f'input.source_base.source_{s}': True for s in range(3)})
    observed.update({f'input.source_gap.target_{r}.source_{s}': True for r in range(3) for s in range(3)})
    worker.validate_gradients(observed, layout='role-tokens')
    for name in ('role_embedding', 'input.source_gap.target_2.source_1'):
        changed = dict(observed, **{name: False})
        with pytest.raises(RuntimeError):
            worker.validate_gradients(changed, layout='role-tokens')


def test_role_tokens_diagnostics_keep_all_four_by_three_slots(monkeypatch):
    from experiments.osram_coupled_source_role_20261020 import worker as old
    monkeypatch.setattr(old, 'serial', lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    values = np.arange(1., 13.).reshape(1, 1, 4, 3)
    weights = np.concatenate((values / values.sum(), np.zeros_like(values)), axis=0)
    result = worker.compact_diagnostics(SimpleNamespace(last_diagnostics={
        'cross_attention_weights': weights, 'active_token_mean': 12.}))
    assert np.asarray(result['role_source_attention_mean']).shape == (4, 3)
    np.testing.assert_array_equal(result['role_source_attention_mean'], weights[0, 0])
    assert 'source_attention_mean' not in result


def test_controller_propagates_layout_to_both_real_workers(tmp_path):
    from experiments.osram_source_relation_attention_20261021.run import worker_command
    args = SimpleNamespace(gpu_uuid='GPU-health', source_commit='source', output=tmp_path,
                           layout='role-tokens')
    for phase in ('smoke', 'train'):
        command = worker_command(args, 'CMUMOSI', phase)
        assert command[command.index('--layout') + 1] == 'role-tokens'
        assert command[command.index('--phase') + 1] == phase


def test_report_names_role_tokens_and_never_publishes_running_scores(tmp_path, monkeypatch):
    import json
    from experiments.osram_source_relation_attention_20261021 import report
    reference = tmp_path/'reference'; reference.mkdir()
    output = tmp_path/'output'; run = output/'runs'/'CMUMOSI'; run.mkdir(parents=True)
    metrics = dict(ACC=80., WF1=79., MSE=.8)
    completed = dict(status='complete', epoch=150, evaluation={'0.0': {'router': metrics}},
        best_by_rate={'0.0': {'epoch': 54}}, aggregate={'router': {'mean': metrics, 'high': metrics}})
    (reference/'SUMMARY.json').write_text(json.dumps(completed))
    current = dict(completed, layout='role-tokens')
    (run/'SUMMARY.json').write_text(json.dumps(current))
    monkeypatch.setattr(report, 'REFERENCES', {'CMUMOSI': reference})
    monkeypatch.setattr(report, 'RATES', ('0.0',))
    payload = report.summarize(output)
    assert {r['model'] for r in payload['per_rate']} == {'Flat', 'SourceRoleTokens'}
    assert payload['states']['CMUMOSI']['layout'] == 'role-tokens'
    assert '12-role-source tokens' in (output/'RESULT.md').read_text()
    (run/'SUMMARY.json').write_text(json.dumps(dict(current, status='running', epoch=1)))
    payload = report.summarize(output)
    assert payload['per_rate'] == []
