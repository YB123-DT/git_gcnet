"""Fixed write-step compatibility, differentiability, and configuration contracts."""

import copy
import base64
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from gcnet_missing_m3.model import MissingM3GraphModel
from gcnet_missing_m3.osram import OSRAMBackbone
from gcnet_missing_m3.write_intervention import WriteIntervention
from gcnet_missing_m3 import train_gcnet
from test_osram import _inputs


def _backbone(**kwargs):
    return OSRAMBackbone(latent_dim=8, output_dim=10, num_heads=2,
                         key_dim=3, value_dim=4, n_speakers=2,
                         dropout=kwargs.pop('dropout', 0.0),
                         bidirectional=False, **kwargs)


def _old_write(model, memory, keys, values, availability, beta=None):
    if beta is None:
        beta = model.beta_logits.sigmoid()
    scale = (availability.unsqueeze(1) * beta.sqrt().unsqueeze(0)).unsqueeze(2)
    k_bar, v_bar = keys * scale, values * scale
    residual = v_bar - memory @ k_bar
    gram = k_bar.transpose(-1, -2) @ k_bar
    solved = torch.linalg.solve(gram + model.write_ridge * torch.eye(3),
                                k_bar.transpose(-1, -2))
    return memory + residual @ solved


def test_default_matches_historical_forward_backward_state_and_rng():
    encoded = os.environ.get('OSRAM_HISTORICAL_SOURCE_B64')
    source = base64.b64decode(encoded).decode() if encoded else subprocess.check_output(
        ['git', 'show', 'd27b58f:gcnet_missing_m3/osram.py'],
        cwd=Path(__file__).resolve().parents[1], text=True,
    )
    namespace = {'__name__': 'historical_osram'}
    exec(compile(source, 'historical_osram.py', 'exec'), namespace)
    torch.manual_seed(19)
    old = namespace['OSRAMBackbone'](
        latent_dim=8, output_dim=10, num_heads=2, key_dim=3,
        value_dim=4, n_speakers=2, dropout=0.2, bidirectional=False)
    old_rng = torch.get_rng_state().clone()
    torch.manual_seed(19)
    current = _backbone(dropout=0.2)
    assert current.write_step == 1.0
    assert torch.equal(torch.get_rng_state(), old_rng)
    assert list(current.state_dict()) == list(old.state_dict())
    for key, value in old.state_dict().items():
        assert torch.equal(current.state_dict()[key], value)
    torch.nn.init.normal_(old.emotion_adapter[-1].weight, std=0.1)
    current.load_state_dict(old.state_dict(), strict=True)
    args = _inputs()
    state = torch.get_rng_state().clone()
    old_hidden, old_context = old(*args)
    old_hidden.square().sum().backward()
    after = torch.get_rng_state().clone()
    torch.set_rng_state(state)
    hidden, context = current(*args)
    hidden.square().sum().backward()
    assert torch.equal(torch.get_rng_state(), after)
    assert torch.equal(hidden, old_hidden)
    for name in ('base', 'gap'):
        assert torch.equal(context[name], old_context[name])
    for name, parameter in current.named_parameters():
        reference = dict(old.named_parameters())[name].grad
        assert (parameter.grad is None) == (reference is None)
        if reference is not None:
            assert torch.equal(parameter.grad, reference), name


@pytest.mark.parametrize('step', [0.0, 0.6, 1.0])
def test_write_step_matches_formula_and_is_differentiable(step):
    torch.manual_seed(21)
    model = _backbone(write_step=step)
    memory = torch.randn(2, 2, 4, 3, requires_grad=True)
    keys = torch.randn(2, 2, 3, 3, requires_grad=True)
    values = torch.randn(2, 2, 4, 3, requires_grad=True)
    availability = torch.tensor([[1., 0., 1.], [0., 1., 1.]])
    output = model.block_write(memory, keys, values, availability)
    old = _old_write(model, memory, keys, values, availability)
    torch.testing.assert_close(output, memory + step * (old - memory),
                               rtol=1e-5, atol=1e-6)
    if step == 1.0:
        assert torch.equal(output, old)
    if step == 0.0:
        assert torch.equal(output, memory)
    output.square().sum().backward()
    for tensor in (memory, keys, values, model.beta_logits):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        if step > 0:
            assert torch.count_nonzero(tensor.grad) > 0
    assert not any('write_step' in key for key in model.state_dict())


@pytest.mark.parametrize('step', [-0.1, 1.1, float('nan'), float('inf'), -float('inf')])
def test_invalid_write_step_rejected(step):
    with pytest.raises(ValueError, match='write_step'):
        _backbone(write_step=step)
    with pytest.raises(ValueError, match='osram_write_step'):
        train_gcnet.TrainConfig(osram_write_step=step)


def test_fractional_step_preserves_causality_and_padding():
    model = _backbone(write_step=0.6).eval()
    torch.nn.init.normal_(model.emotion_adapter[-1].weight, std=0.1)
    args = _inputs()
    hidden, context = model(*args)
    changed = copy.deepcopy(args)
    changed[0][2:] += 30
    for value in changed[1].values():
        value[2:] -= 30
    changed_hidden, changed_context = model(*changed)
    assert torch.equal(hidden[:2], changed_hidden[:2])
    assert torch.count_nonzero(hidden[3]) == 0
    for name in ('base', 'gap'):
        assert torch.equal(context[name][:2], changed_context[name][:2])
        assert torch.count_nonzero(context[name][3]) == 0
    unpadded = (args[0][:3], {k: v[:3] for k, v in args[1].items()},
                args[2][:3], args[3][:, :3], args[4][:, :3], args[5])
    short_hidden, short_context = model(*unpadded)
    torch.testing.assert_close(hidden[:3], short_hidden)
    for name in ('base', 'gap'):
        torch.testing.assert_close(context[name][:3], short_context[name])


def test_fractional_eval_matches_frozen_intervention():
    normal = _backbone(write_step=0.6).eval()
    torch.nn.init.normal_(normal.emotion_adapter[-1].weight, std=0.1)
    frozen = _backbone().eval()
    frozen.load_state_dict(normal.state_dict(), strict=True)
    with torch.no_grad():
        hidden, context = normal(*_inputs())
        with WriteIntervention(frozen, 'fixed0.6'):
            reference, reference_context = frozen(*_inputs())
    torch.testing.assert_close(hidden, reference, rtol=1e-5, atol=1e-6)
    for name in ('base', 'gap'):
        torch.testing.assert_close(context[name], reference_context[name],
                                   rtol=1e-5, atol=1e-6)


def test_cli_config_checkpoint_and_model_roundtrip(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(train_gcnet, 'run_experiment',
                        lambda config, *args, **kwargs: captured.append(config))
    for name in ('a', 't', 'v'):
        (tmp_path / name).mkdir()
    argv = ['--feature-root', str(tmp_path), '--audio-feature', 'a',
            '--text-feature', 't', '--video-feature', 'v',
            '--output-dir', str(tmp_path / 'output'), '--osram-write-step', '0.6']
    train_gcnet.main(argv)
    config = captured[0]
    assert config.osram_write_step == 0.6
    assert train_gcnet.build_parser().get_default('osram_write_step') == 1.0
    path = tmp_path / 'best.pt'
    train_gcnet._save_best_checkpoint(path, {}, config, 1, 0.5)
    payload = torch.load(path, weights_only=False)
    restored = train_gcnet.TrainConfig(**json.loads(json.dumps(payload['config'])))
    assert restored.osram_write_step == 0.6
    legacy = asdict(restored)
    legacy.pop('osram_write_step')
    assert train_gcnet.TrainConfig(**legacy).osram_write_step == 1.0
    model = MissingM3GraphModel(
        base_model='LSTM', adim=2, tdim=3, vdim=4, D_e=4,
        graph_hidden_size=2, n_speakers=2, window_past=1, window_future=1,
        n_classes=6, dropout=0., no_cuda=True, latent_dim=8, num_experts=2,
        top_k=1, backbone_type='osram', osram_output_dim=10,
        osram_num_heads=2, osram_key_dim=3, osram_value_dim=4,
        osram_write_step=restored.osram_write_step)
    assert model.osram.write_step == 0.6
