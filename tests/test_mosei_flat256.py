from dataclasses import asdict, make_dataclass, replace

import pytest


def baseline():
    values = dict(dataset='CMUMOSEI', fold=1, seed=66, epochs=60, learning_rate=.001,
        weight_decay=1e-5, batch_size=32, optimizer='adam', lr_schedule='constant',
        missing_mask_protocol='gcnet-random-mask-v1', training_objective='emotion-only',
        osram_readout_fusion='flat', osram_output_dim=1600, osram_num_heads=8,
        osram_key_dim=64, osram_value_dim=64, osram_bidirectional=False,
        osram_write_step=.6, osram_source_role_v3=False, osram_source_role=False,
        osram_source_relation=False, osram_tasr=False, osram_meaningful_block='none',
        paired_history_views=False, train_rate_mode='cyclic', task_regression_loss='mse',
        osram_adapter_hidden_dim=0, train_missing_rates=tuple(i/10 for i in range(8)),
        osram_source_role_v3_dim=64, osram_source_role_v3_pair_hidden=128)
    cls = make_dataclass('Config', [(k, type(v)) for k,v in values.items()])
    return cls(**values)


def test_only_flat_hidden_width_changes():
    from experiments.osram_mosei_flat256_20261024.worker import candidate_config, expected_counts
    cfg = baseline(); actual = candidate_config(cfg, 'flat256')
    assert asdict(actual) == dict(asdict(cfg), osram_adapter_hidden_dim=256)
    assert expected_counts('flat256') == dict(total=5508961, flat_adapter=1534272, v3=0)


def test_one_seed_no_new_missing_lr_or_modules():
    from experiments.osram_mosei_flat256_20261024.worker import candidate_config, expected_counts
    for change in (dict(seed=67),dict(learning_rate=1e-4),dict(epochs=100),
                   dict(missing_mask_protocol='conversation-mask-v2'),dict(osram_source_role_v3=True)):
        with pytest.raises(ValueError):
            candidate_config(replace(baseline(), **change), 'flat256')
    for mode in ('wide_v3', 'small_flat', 'nested', 'f512_plain'):
        with pytest.raises(ValueError):
            candidate_config(baseline(), mode)
        with pytest.raises(ValueError):
            expected_counts(mode)


def test_reuses_existing_numerical_worker(monkeypatch, tmp_path):
    from experiments.osram_mosei_flat256_20261024 import worker
    seen = []
    monkeypatch.setattr(worker.base, 'main', lambda *a,**kw: seen.append((a,kw)))
    monkeypatch.setattr(worker, 'write_report', lambda p: seen.append(('report',p)))
    argv=['--mode','flat256','--phase','train','--output',str(tmp_path)]
    worker.main(argv)
    assert seen[0][1]['allowed_modes'] == ('flat256',)
    assert seen[0][1]['config_factory'] is worker.candidate_config
    assert seen[0][1]['expected_counts_factory'] is worker.expected_counts
    assert seen[0][1]['capture_first_epoch_only'] is True
    assert 'experiments/osram_mosei_flat256_20261024/worker.py' in seen[0][1]['source_catalog_additions']
    assert seen[1] == ('report', tmp_path)
