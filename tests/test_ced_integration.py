"""Real-model CED wiring, opt-out identity and finite training updates."""
from dataclasses import replace
import torch
import pytest

from test_core20_integration import config, batch, _build_model
from gcnet_missing_m3.core20 import attach
from gcnet_missing_m3.train_gcnet import train_epoch, _schedules


def test_default_off_preserves_original_weights_rng_and_prediction():
    torch.set_num_threads(1)
    cfg = config()
    model = _build_model(cfg, (3, 4, 5)).eval()
    state = {name: p.clone() for name, p in model.state_dict().items()}
    v = batch()
    args = ([v['incomplete']], v['availability'], v['qmask'], v['umask'], v['lengths'])
    expected = model(*args)[0]
    rng = torch.get_rng_state().clone()
    attach(model, cfg)
    assert not hasattr(model, 'ced_block')
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(p, model.state_dict()[name]) for name, p in state.items())
    assert torch.equal(expected, model(*args)[0])


def test_ced_real_epoch_updates_block_encoder_and_osram():
    torch.set_num_threads(1)
    cfg = replace(config(), osram_ced_block=True)
    model = _build_model(cfg, (3, 4, 5))
    common = {name: p.clone() for name, p in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    attach(model, cfg)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(p, model.state_dict()[name]) for name, p in common.items())
    v = batch()
    blocks = list(v['complete'].split((3, 4, 5), -1))
    raw = blocks + blocks + [v['qmask'], v['umask'], v['labels'], ['train-a', 'train-b']]
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.001)
    before = {name: p.clone() for name, p in model.named_parameters()}
    for epoch in range(2):
        metrics = train_epoch(model, [raw], optimizer, cfg, _schedules(cfg, 'train'),
                              epoch, (3, 4, 5), torch.device('cpu'))
        assert metrics['optimizer_steps'] == 1
        assert metrics['jepa_loss'] == 0
    for prefix in ('ced_block.', 'observed_set.', 'osram.emotion_adapter.'):
        assert any(not torch.equal(p, before[name]) for name, p in model.named_parameters()
                   if name.startswith(prefix)), prefix
    assert all(torch.isfinite(p).all() and (p.grad is None or torch.isfinite(p.grad).all())
               for p in model.parameters())
    model.eval()
    with torch.no_grad():
        pred, hidden, _, _ = model([v['incomplete']], v['availability'], v['qmask'], v['umask'], v['lengths'])
    assert torch.isfinite(pred).all()
    assert torch.count_nonzero(hidden[3, 1]) == 0
    assert model.osram.last_diagnostics['coalition_evidence_decomposition']['reconstruction_max_error'] < 1e-5


def test_ced_rejects_other_method_combinations():
    with pytest.raises(ValueError):
        replace(config('R12'), osram_ced_block=True)
    with pytest.raises(ValueError):
        replace(config(), osram_ced_block=True, osram_relation_block=True)
