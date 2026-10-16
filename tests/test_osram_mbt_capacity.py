"""B3 capacity controls and exclusive adapter learning-rate ownership."""
from dataclasses import replace
from pathlib import Path
import subprocess
import types

import pytest
import torch

from gcnet_missing_m3.meaningful_input import MeaningfulInputAdapter
from gcnet_missing_m3.train_gcnet import TrainConfig, _optimizer_parameter_groups


def test_default_representation_exact_pre_capacity_state_rng_and_learned_output():
    """Pinned pre-change implementation, not a reference built from new layers."""
    from gcnet_missing_m3.meaningful_input_representation import RepresentationInputAdapter
    root = Path(__file__).resolve().parents[1]
    source = subprocess.check_output(['git', 'show',
        '107023e7fb4f6da3175e4fffebee813559bef43e:gcnet_missing_m3/meaningful_input_representation.py'],
        cwd=root, text=True)
    legacy_module = types.ModuleType('gcnet_missing_m3._legacy_mbt_capacity_test')
    legacy_module.__package__ = 'gcnet_missing_m3'
    exec(compile(source, '<pinned-pre-capacity>', 'exec'), legacy_module.__dict__)
    torch.set_num_threads(1)
    torch.manual_seed(66)
    legacy = legacy_module.RepresentationInputAdapter('mbt_role_bottleneck')
    rng = torch.get_rng_state().clone()
    torch.manual_seed(66)
    actual = RepresentationInputAdapter('mbt_role_bottleneck')
    assert torch.equal(rng, torch.get_rng_state())
    assert actual.state_dict().keys() == legacy.state_dict().keys()
    assert all(torch.equal(v, actual.state_dict()[k]) for k, v in legacy.state_dict().items())
    # Defeat the zero-init shortcut: compare actual nontrivial branch outputs.
    with torch.no_grad():
        for decoder in [legacy.local_decoder, *legacy.memory_decoders]:
            decoder.weight.normal_(std=.02)
            decoder.bias.normal_(std=.02)
    actual.load_state_dict(legacy.state_dict(), strict=True)
    local = torch.randn(2, 256)
    evidence = torch.randn(2, 4, 512)
    active = torch.tensor([[True, False, True, False], [True, True, False, True]])
    availability = (~active[:, 1:]).float()
    assert all(torch.equal(a, b) for a, b in zip(
        actual(local, evidence, active, availability),
        legacy(local, evidence, active, availability)))


@pytest.mark.parametrize('dim,heads,hidden', [(64, 2, 128), (128, 8, 512), (256, 4, 512)])
def test_capacity_reaches_tokens_attention_ffn_and_preserves_masked_interface(dim, heads, hidden):
    torch.set_num_threads(1)
    block = MeaningfulInputAdapter(256, 1024, 1600, 'mbt_role_bottleneck', 8, 64,
        osram_mbt_dim=dim, osram_mbt_heads=heads, osram_mbt_ffn_hidden=hidden)
    assert block.core.core.initial_bottleneck.shape == (1, 4, dim)
    for stage in (block.core.core.private, *block.core.core.exchange):
        for layer in stage:
            assert layer.attention.heads == heads
            assert layer.ffn[0].out_features == hidden
    local = torch.randn(3, 1, 256)
    base = torch.cat((torch.randn(3, 1, 512), torch.zeros(3, 1, 512)), -1)
    gap = torch.cat((torch.randn(3, 1, 3, 512), torch.zeros(3, 1, 3, 512)), -1)
    availability = torch.tensor([[[1., 0., 1.]]] * 3)
    umask = torch.tensor([[1., 1., 0.]])
    valid = umask.T.bool()
    local[~valid] = 0
    base[~valid] = 0
    gap = torch.where((valid[..., None] & ~availability.bool())[..., None], gap, 0.)
    dirty = gap.masked_fill(availability.bool()[..., None], float('nan'))
    outputs = block(local, base, dirty, availability, umask)
    assert all(torch.equal(a, b) for a, b in zip(outputs, (local, base, gap)))
    optimizer = torch.optim.Adam(block.parameters(), lr=.001)
    core_nonzero = False
    for _ in range(3):
        optimizer.zero_grad()
        outputs = block(local, base, dirty, availability, umask)
        sum(x.square().mean() for x in outputs).backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in block.parameters())
        core_nonzero |= any(p.grad is not None and bool(p.grad.count_nonzero())
                            for p in block.core.core.parameters())
        optimizer.step()
    assert core_nonzero
    assert all(not x[~valid].count_nonzero() for x in outputs)
    assert not outputs[2][availability.bool()].count_nonzero()
    assert not outputs[1][..., 512:].count_nonzero()
    assert not outputs[2][..., 512:].count_nonzero()


def test_capacity_validation_and_lr_restrictions():
    cfg = TrainConfig()
    assert (cfg.osram_mbt_dim, cfg.osram_mbt_heads, cfg.osram_mbt_ffn_hidden,
            cfg.osram_mbt_lr_multiplier) == (128, 4, 256, 1.)
    for name in ('osram_mbt_dim', 'osram_mbt_heads', 'osram_mbt_ffn_hidden'):
        for value in (0, -1, True, 2.5):
            with pytest.raises(ValueError):
                replace(cfg, **{name: value})
    with pytest.raises(ValueError, match='divisible'):
        replace(cfg, osram_mbt_heads=3)
    for value in (0., -1., float('nan'), float('inf'), True):
        with pytest.raises(ValueError):
            replace(cfg, osram_mbt_lr_multiplier=value)
    with pytest.raises(ValueError, match='B3'):
        replace(cfg, osram_mbt_lr_multiplier=.1)


def test_mbt_lr_owns_entire_adapter_and_no_original_parameters():
    from tests.test_meaningful_block_integration import config
    from experiments.osram_meld_flat_source_combinations_20261013.evaluate import _build_model
    cfg = config('mbt_role_bottleneck')
    model = _build_model(cfg, (3, 4, 5), torch.device('cpu')).requires_grad_(True)
    original, original_info = _optimizer_parameter_groups(model, cfg)
    default, info = _optimizer_parameter_groups(model, replace(cfg, osram_mbt_lr_multiplier=1.))
    assert original_info == info
    assert [[id(p) for p in g['params']] for g in original] == [[id(p) for p in g['params']] for g in default]
    groups, info = _optimizer_parameter_groups(model, replace(cfg, osram_mbt_lr_multiplier=.1))
    adapter_ids = {id(p) for p in model.osram.meaningful_block.parameters() if p.requires_grad}
    matching = [g for g in groups if {id(p) for p in g['params']} == adapter_ids]
    assert len(matching) == 1
    assert matching[0]['lr'] == pytest.approx(cfg.learning_rate * .1)
    assert info['mbt']['parameter_count'] == sum(p.numel() for p in model.osram.meaningful_block.parameters() if p.requires_grad)
    all_ids = [id(p) for g in groups for p in g['params']]
    assert len(all_ids) == len(set(all_ids))
    assert set(all_ids) == {id(p) for p in model.parameters() if p.requires_grad}
    assert id(model.osram.local_skip.weight) not in adapter_ids


def test_full_model_capacity_wiring_flat_off_and_zero_init_equivalence():
    from tests.test_meaningful_block_integration import config
    from experiments.osram_meld_flat_source_combinations_20261013.evaluate import _build_model
    cfg = config()
    kwargs = dict(osram_mbt_dim=64, osram_mbt_heads=2, osram_mbt_ffn_hidden=128)
    flat = _build_model(cfg, (3, 4, 5), torch.device('cpu')).eval()
    rng = torch.get_rng_state().clone()
    off = _build_model(replace(cfg, **kwargs), (3, 4, 5), torch.device('cpu')).eval()
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, off.state_dict()[k]) for k, v in flat.state_dict().items())
    off.load_state_dict(flat.state_dict(), strict=True)
    enabled = _build_model(replace(cfg, osram_meaningful_block='mbt_role_bottleneck', **kwargs),
                           (3, 4, 5), torch.device('cpu')).eval()
    assert enabled.osram.meaningful_block.core.core.initial_bottleneck.shape[-1] == 64
    assert enabled.osram.meaningful_block.core.core.private[0].attention.heads == 2
    assert enabled.osram.meaningful_block.core.core.private[0].ffn[0].out_features == 128
    av = torch.tensor([[[1., 0., 1.]]] * 3)
    args = ([torch.randn(3, 1, 12)], av, torch.zeros(1, 3, dtype=torch.long),
            torch.ones(1, 3), [3])
    with torch.no_grad():
        expected = flat(*args, predict_missing=False)[0]
        assert torch.equal(expected, off(*args, predict_missing=False)[0])
        assert torch.equal(expected, enabled(*args, predict_missing=False)[0])
