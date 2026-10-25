"""Joint branches preserve the actual original dropout/normalization flow."""
import copy

import pytest
import torch

from gcnet_missing_m3.model import MissingM3GraphModel
from osram_four_block_wiring.csdi_latent import FixedLatentProjection
from experiments.osram_three_block_joint_mosi_20261025.model import JointBlockModel
from tests.test_completion_memory_write import model_kwargs


def parent_model():
    kw = model_kwargs()
    kw.update(latent_dim=256, osram_output_dim=1600, osram_num_heads=8,
              osram_key_dim=4, osram_value_dim=64, completion_path='none',
              dropout=.35, projector_dropout=.2)
    return MissingM3GraphModel(**kw)


def view():
    a = torch.tensor([[[1, 0, 1]], [[0, 1, 0]], [[1, 1, 1]]]).float()
    complete = torch.randn(3, 1, 12)
    mask = torch.cat([a[..., i:i+1].expand(-1, -1, width)
                      for i, width in enumerate((3, 4, 5))], -1)
    return dict(incomplete=complete * mask, complete=complete, availability=a,
                qmask=torch.zeros(1, 3).long(), umask=torch.ones(1, 3), lengths=[3])


def projection():
    return FixedLatentProjection(torch.zeros(3, 256), torch.eye(256)[:64].repeat(3, 1, 1))


@pytest.mark.parametrize('mode', ['neumiss', 'signature', 'csdi'])
@pytest.mark.parametrize('training', [False, True])
def test_zero_exit_exact_original_rng_one_scan(mode, training):
    torch.set_num_threads(1)
    torch.manual_seed(4)
    original = parent_model().train(training)
    joint = JointBlockModel(copy.deepcopy(original), mode, projection() if mode == 'csdi' else None).train(training)
    data = view()
    scans = []
    handle = joint.parent.osram.register_forward_hook(lambda *args: scans.append(1))
    torch.manual_seed(91)
    expected = original([data['incomplete']], data['availability'], data['qmask'],
                        data['umask'], data['lengths'])[0]
    expected_rng = torch.random.get_rng_state()
    torch.manual_seed(91)
    actual = joint(data, generator=torch.Generator().manual_seed(19))
    handle.remove()
    assert scans == [1]
    assert torch.equal(expected_rng, torch.random.get_rng_state())
    torch.testing.assert_close(actual['prediction'], expected, rtol=0, atol=0)
    torch.testing.assert_close(actual['base_prediction'], expected, rtol=0, atol=0)
    assert all(p.requires_grad for p in joint.parent.parameters())
    assert actual['frame'].latents.requires_grad
    assert actual['frame'].context.requires_grad
    if training and mode == 'csdi':
        assert actual['aux_loss'].item() > 0


@pytest.mark.parametrize('mode', ['neumiss', 'signature', 'csdi'])
def test_task_gradients_reach_parent_and_branch_upstream(mode):
    torch.set_num_threads(1)
    joint = JointBlockModel(parent_model(), mode, projection() if mode == 'csdi' else None).train()
    exit_layer = joint.bridge.net[-1] if mode == 'csdi' else (joint.block.out if mode == 'neumiss' else joint.block.out[-1])
    torch.nn.init.normal_(exit_layer.weight, std=.002)
    result = joint(view(), generator=torch.Generator().manual_seed(21))
    result['prediction'].square().mean().backward()
    modules = [joint.parent.observed_set, joint.parent.osram.emotion_adapter, joint.parent.smax_fc]
    modules += [joint.denoiser, joint.bridge] if mode == 'csdi' else [joint.block.slots]
    for module in modules:
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum().item() for g in grads) > 0


def test_eval_csdi_never_reads_complete_features():
    torch.set_num_threads(1)
    joint = JointBlockModel(parent_model(), 'csdi', projection()).eval()
    data = view()
    data.pop('complete')
    joint._target = lambda *args: (_ for _ in ()).throw(AssertionError('eval target leak'))
    out = joint(data, generator=torch.Generator().manual_seed(3))
    assert out['prediction'].shape == (3, 1, 1)
    assert out['aux_loss'].item() == 0


def test_noise_targets_restore_projector_modes_and_do_not_train_parent():
    torch.set_num_threads(1)
    joint = JointBlockModel(parent_model(), 'csdi', projection()).train()
    projectors = joint.parent.observed_set.projectors
    projectors['text'].eval()  # Preserve mixed flags too.
    modes = [module.training for module in projectors.modules()]
    result = joint(view(), generator=torch.Generator().manual_seed(12))
    assert [module.training for module in projectors.modules()] == modes
    result['aux_loss'].backward()
    assert all(p.grad is None for p in joint.parent.parameters())
    assert joint.denoiser.output.weight.grad.abs().sum() > 0
    assert joint.branch is joint.bridge
    assert result['aux_loss'].ndim == 0
