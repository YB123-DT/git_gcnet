"""The confirmation must change only the approved block ID."""
import json
from pathlib import Path

from experiments.osram_core20_20261005.run import candidate_config


def test_rootaware_fixed_configuration():
    root = Path(__file__).resolve().parents[1]
    reference = json.loads((root / 'experiments/osram_core20_20261005/REFERENCE_CONFIG.json').read_text())
    config, delta = candidate_config(reference, 'NestedRootAware', seed=66)
    assert config.osram_meaningful_block == 'nested_gnn_rootaware_evidence'
    assert set(delta) == {'osram_meaningful_block'}
    assert config.core20_method == 'none'
    assert config.epochs == 100


def test_summary_records_missing_comparators_without_partial_mean(tmp_path, monkeypatch):
    from experiments.osram_nested_rootaware_20261005 import dispatch
    monkeypatch.setattr(dispatch, 'REMOTE', tmp_path / 'unavailable_references')
    monkeypatch.setattr(dispatch, 'ORIGINALS', {'nested_gnn_rooted_evidence': tmp_path / 'unavailable_old'})
    rows = []
    for seed in (66, 67, 68):
        output = tmp_path / f'seed_{seed}'
        output.mkdir()
        (output / 'metrics.json').write_text(json.dumps({
            'selected_weighted_f1_by_rate': {str(i / 10): .8 for i in range(8)}}))
        rows.append(dict(seed=seed, status='complete', output=str(output)))
    dispatch.summarize(tmp_path, rows)
    value = json.loads((tmp_path / 'SUMMARY.json').read_text())
    assert value['means']['rootaware']['mean8'] == 80
    assert 'flat' not in value['means'] and 'old_nested' not in value['means']
    assert all(row['old_nested']['status'] == 'unavailable' for row in value['seeds'])


def test_real_model_task_updates_shared_root_pool():
    from dataclasses import replace
    import torch
    from test_core20_integration import batch
    from gcnet_missing_m3.train_gcnet import train_epoch, _schedules
    from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model
    root = Path(__file__).resolve().parents[1]
    reference = json.loads((root / 'experiments/osram_core20_20261005/REFERENCE_CONFIG.json').read_text())
    cfg, _ = candidate_config(reference, 'NestedRootAware', seed=66)
    cfg = replace(cfg, device='cpu')
    torch.set_num_threads(1)
    model = _build_model(cfg, (3, 4, 5))
    name, root_weight = next((name, p) for name, p in model.named_parameters() if name.endswith('root_pool.weight'))
    before = root_weight.detach().clone()
    v = batch()
    blocks = list(v['complete'].split((3, 4, 5), -1))
    raw = blocks + blocks + [v['qmask'], v['umask'], v['labels'], ['train-a', 'train-b']]
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    for epoch in range(3):
        metrics = train_epoch(model, [raw], optimizer, cfg, _schedules(cfg, 'train'),
                              epoch, (3, 4, 5), torch.device('cpu'))
        assert metrics['optimizer_steps'] == 1
        assert metrics['jepa_loss'] == 0
    assert not torch.equal(before, root_weight), name
    assert all(torch.isfinite(p).all() and (p.grad is None or torch.isfinite(p.grad).all())
               for p in model.parameters())
