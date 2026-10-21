"""Guard old Flat numerics and the new source-relation execution route."""
from pathlib import Path
import os
import subprocess

import pytest
import torch

from gcnet_missing_m3.osram import OSRAMBackbone
from test_osram import _inputs


def make(**kwargs):
    return OSRAMBackbone(latent_dim=8, output_dim=10, num_heads=2, key_dim=3,
        value_dim=4, n_speakers=2, dropout=.2, bidirectional=False, write_step=.6, **kwargs)


def test_default_legacy_outputs_initialization_training_rng_and_gradients():
    path = os.environ.get('OSRAM_LEGACY_SOURCE')
    source = Path(path).read_text() if path else subprocess.check_output(
        ['git', 'show', '8cde443:gcnet_missing_m3/osram.py'], text=True)
    ns = {'__name__': 'gcnet_missing_m3._old_relation', '__package__': 'gcnet_missing_m3'}
    exec(compile(source, 'legacy_osram.py', 'exec'), ns)
    torch.manual_seed(66)
    old = ns['OSRAMBackbone'](latent_dim=8, output_dim=10, num_heads=2, key_dim=3,
        value_dim=4, n_speakers=2, dropout=.2, bidirectional=False, write_step=.6)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(66); new = make(osram_source_relation=False)
    assert torch.equal(rng, torch.get_rng_state())
    new.load_state_dict(old.state_dict(), strict=True)
    args = _inputs(); before = torch.get_rng_state().clone()
    a, ac = old(*args); a[..., 0].sum().backward(); after = torch.get_rng_state().clone()
    torch.set_rng_state(before); b, bc = new(*args); b[..., 0].sum().backward()
    assert torch.equal(a, b) and torch.equal(after, torch.get_rng_state())
    for name in ac: assert torch.equal(ac[name], bc[name])
    for name, p in old.named_parameters():
        q = dict(new.named_parameters())[name]
        assert (p.grad is None) == (q.grad is None)
        if p.grad is not None: assert torch.equal(p.grad, q.grad), name


def test_enabled_zero_init_preserves_original_weights_outputs_rng_and_contexts():
    torch.manual_seed(66); base = make(); rng = torch.get_rng_state().clone()
    torch.manual_seed(66); new = make(osram_source_relation=True)
    assert torch.equal(rng, torch.get_rng_state())
    for name, p in base.state_dict().items(): assert torch.equal(p, new.state_dict()[name])
    args = _inputs(); before = torch.get_rng_state().clone()
    a, ac = base(*args); after = torch.get_rng_state().clone()
    torch.set_rng_state(before); b, bc = new(*args)
    assert torch.equal(a, b) and torch.equal(after, torch.get_rng_state())
    for name in ac: assert torch.equal(ac[name], bc[name])
    assert new.last_source_trace['source_base'].requires_grad
    assert new.last_source_trace['source_gap'].requires_grad
    assert not b[-1].any()


def test_opt_in_builder_route_and_mutual_exclusion():
    from dataclasses import replace
    from gcnet_missing_m3.train_gcnet import TrainConfig
    from experiments.osram_meld_flat_source_combinations_20261013.evaluate import _build_model
    cfg = replace(TrainConfig(), dataset='CMUMOSI', backbone_type='osram',
        training_objective='emotion-only', disable_unused_aux_modules=True,
        latent_dim=8, osram_output_dim=10, osram_num_heads=2, osram_key_dim=3,
        osram_value_dim=4, osram_bidirectional=False, osram_source_relation=True)
    model = _build_model(cfg, (4, 5, 6), torch.device('cpu'))
    assert model.osram.osram_source_relation and hasattr(model.osram, 'source_relation')
    for extra in ({'osram_source_role': True}, {'osram_tasr': True}):
        with pytest.raises(ValueError): make(osram_source_relation=True, **extra)


def test_actual_tracker_reads_receive_finite_nonzero_task_gradients_after_opening():
    model = make(osram_source_relation=True).eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    observed = {'base': False, 'gap': False}
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        hidden, _ = model(*_inputs())
        trace = model.last_source_trace
        for key in observed: trace['source_'+key].retain_grad()
        (hidden[..., 0]-.3).square().mean().backward()
        for key in observed:
            gradient = trace['source_'+key].grad
            assert gradient is not None and torch.isfinite(gradient).all()
            observed[key] |= bool(gradient.ne(0).any())
        optimizer.step()
    assert all(observed.values())
