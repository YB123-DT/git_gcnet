import json
from pathlib import Path

from experiments.osram_core20_20261005.run import candidate_config


def test_sweep_changes_only_block_id():
    names = ('nested_ab_plain_gin', 'nested_ab_no_markers',
             'nested_ab_no_head_edges', 'nested_ab_last_layer',
             'nested_dim32', 'nested_dim128', 'nested_depth1',
             'nested_depth2', 'nested_groups1', 'nested_groups4')
    root = Path(__file__).resolve().parents[1]
    reference = json.loads((root / 'experiments/osram_core20_20261005/REFERENCE_CONFIG.json').read_text())
    for name in names:
        cfg, delta = candidate_config(reference, name)
        assert cfg.osram_meaningful_block == name
        assert set(delta) == {'osram_meaningful_block'}
        assert cfg.epochs == 100 and cfg.osram_num_heads == 8


def test_summary_keeps_all_pending_and_failed_runs(tmp_path, monkeypatch):
    from experiments.osram_nested_sweep_20261005 import dispatch
    monkeypatch.setattr(dispatch, 'REFERENCE', tmp_path / 'missing')
    monkeypatch.setattr(dispatch, 'ORIGINALS', {'nested_gnn_rooted_evidence': tmp_path / 'missing'})
    names = list(dispatch.NESTED_SWEEP)
    rows = [dict(method=m, category='ablation', status='pending') for m in names]
    rows[0].update(status='failed', error='OOM retained')
    dispatch.summarize(tmp_path, rows)
    value = json.loads((tmp_path / 'SUMMARY.json').read_text())
    assert len(value['methods']) == 10
    assert value['methods'][0]['error'] == 'OOM retained'
    assert all('mean8' not in r for r in value['methods'])
    assert value['references']['flat']['status'] == 'unavailable'


def test_real_task_training_updates_grouped_module():
    from dataclasses import replace
    import torch
    from test_core20_integration import batch
    from gcnet_missing_m3.train_gcnet import train_epoch, _schedules
    from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model
    root = Path(__file__).resolve().parents[1]
    reference = json.loads((root / 'experiments/osram_core20_20261005/REFERENCE_CONFIG.json').read_text())
    cfg, _ = candidate_config(reference, 'nested_groups4')
    cfg = replace(cfg, device='cpu')
    torch.set_num_threads(1)
    model = _build_model(cfg, (3, 4, 5))
    weights = next(p for name, p in model.named_parameters()
                   if 'meaningful' in name and name.endswith('core.layers.0.0.weight'))
    before = weights.detach().clone()
    data = batch()
    blocks = list(data['complete'].split((3, 4, 5), -1))
    raw = blocks + blocks + [data['qmask'], data['umask'], data['labels'], ['a', 'b']]
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    for epoch in range(3):
        metrics = train_epoch(model, [raw], optimizer, cfg, _schedules(cfg, 'train'),
                              epoch, (3, 4, 5), torch.device('cpu'))
        assert metrics['jepa_loss'] == 0 and metrics['optimizer_steps'] == 1
    assert not torch.equal(before, weights)
    assert all(torch.isfinite(p).all() and (p.grad is None or torch.isfinite(p.grad).all())
               for p in model.parameters())
