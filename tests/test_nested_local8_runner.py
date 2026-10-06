import json
from pathlib import Path

def test_local8_registered_in_runner_source():
    import ast
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / 'experiments/osram_core20_20261005/run.py').read_text())
    assert any(isinstance(node, ast.Constant) and node.value == 'nested_local8_evidence'
               for node in ast.walk(tree))


def test_local8_only_changes_block_id():
    from experiments.osram_core20_20261005.run import candidate_config
    root = Path(__file__).resolve().parents[1]
    reference = json.loads((root / 'experiments/osram_core20_20261005/REFERENCE_CONFIG.json').read_text())
    cfg, delta = candidate_config(reference, 'nested_local8_evidence')
    assert set(delta) == {'osram_meaningful_block'}
    assert cfg.osram_meaningful_block == 'nested_local8_evidence'
    assert cfg.osram_num_heads == 8 and cfg.osram_output_dim == 1600
    assert cfg.epochs == 100 and cfg.seed == 66


def test_local8_real_task_updates_local_projection():
    from experiments.osram_core20_20261005.run import candidate_config
    from dataclasses import replace
    import torch
    from test_core20_integration import batch
    from gcnet_missing_m3.train_gcnet import train_epoch, _schedules
    from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model
    root = Path(__file__).resolve().parents[1]
    reference = json.loads((root / 'experiments/osram_core20_20261005/REFERENCE_CONFIG.json').read_text())
    cfg, _ = candidate_config(reference, 'nested_local8_evidence')
    cfg = replace(cfg, device='cpu')
    torch.set_num_threads(1)
    model = _build_model(cfg, (3, 4, 5))
    local_modules = model.osram.meaningful_block.core
    # Select the explicitly separate local group projection list.
    projection = local_modules.local_projections[0].weight
    before = projection.detach().clone()
    data = batch()
    blocks = list(data['complete'].split((3, 4, 5), -1))
    raw = blocks + blocks + [data['qmask'], data['umask'], data['labels'], ['a', 'b']]
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    for epoch in range(3):
        metrics = train_epoch(model, [raw], optimizer, cfg, _schedules(cfg, 'train'),
                              epoch, (3, 4, 5), torch.device('cpu'))
        assert metrics['jepa_loss'] == 0 and metrics['optimizer_steps'] == 1
    assert not torch.equal(before, projection)
    assert all(torch.isfinite(p).all() and (p.grad is None or torch.isfinite(p.grad).all())
               for p in model.parameters())
