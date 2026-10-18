"""Protect class count, exact identities, strict-history and original protocol."""
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def impl():
    path = Path(__file__).parents[1] / 'experiments/osram_iemocap_information_availability_20261018/run.py'
    assert path.exists(), 'IEMOCAP-specific frozen probe runner is not implemented'
    return importlib.import_module('experiments.osram_iemocap_information_availability_20261018.run')


def original_config():
    return dict(dataset='IEMOCAPFour', seed=66, fold=1, epochs=100,
        missing_mask_protocol='gcnet-random-mask-v1', training_objective='emotion-only',
        osram_output_dim=1600, osram_readout_fusion='flat', osram_disable_local_skip=False,
        osram_meaningful_block='none', osram_bidirectional=False, train_rate_mode='cyclic')


def test_accepts_only_the_new_mask_original_flat():
    module = impl()
    module.validate_source_config(original_config(), 4, 1)
    for change in (dict(missing_mask_protocol='gcnet-legacy-fixed-v1'),
                   dict(osram_disable_local_skip=True), dict(seed=67),
                   dict(osram_meaningful_block='mbt_role_bottleneck')):
        with pytest.raises(ValueError):
            module.validate_source_config(dict(original_config(), **change), 4, 1)


def test_packs_only_valid_rows_and_masks_inactive_forward_gaps():
    module = impl()
    local = torch.randn(2, 2, 256)
    flat = torch.randn(2, 2, 4352)
    flat[..., :256] = local
    # Each conversation's first read is zero; backward halves always zero.
    base = flat[..., 256:1280]
    gap = flat[..., 1280:].reshape(2, 2, 3, 1024)
    base[0] = 0
    gap[0] = 0
    base[..., 512:] = 0
    gap[..., 512:] = 0
    availability = torch.tensor([[[1, 0, 1], [0, 1, 1]], [[1, 0, 1], [0, 0, 0]]])
    view = dict(umask=torch.tensor([[1, 1], [1, 0]]), availability=availability,
        qmask=torch.tensor([[0, 1], [1, 0]]), labels=torch.tensor([[0, 1], [2, 0]]),
        conversation_ids=['a', 'b'])
    capture = SimpleNamespace(local=local, flat_input=flat)
    cache = module.pack_capture(view, capture, torch.randn(2, 2, 4), classes=4)
    assert cache['prediction'].shape == (3, 4)
    assert cache['base'].shape == (3, 512)
    assert cache['gap'].shape == (3, 3, 512)
    assert cache['conversation_ids'].tolist() == ['a', 'a', 'b']
    assert cache['utterance_indices'].tolist() == [0, 1, 0]
    assert not np.any(cache['gap'][cache['availability'].astype(bool)])


def test_exact_identity_check_rejects_reordered_rows():
    module = impl()
    data = dict(labels=np.array([0, 1]), prediction=np.eye(4)[[0, 1]],
        availability=np.array([[1, 0, 1], [0, 1, 1]]),
        conversation_ids=np.array(['a', 'b']), utterance_indices=np.array([0, 0]),
        speaker=np.array([0, 1]))
    saved = dict(labels=data['labels'], logits=data['prediction'], predictions=np.array([0, 1]),
        availability=data['availability'], conversation_ids=np.array(['b', 'a']),
        utterance_indices=data['utterance_indices'], speakers=data['speaker'])
    with pytest.raises(AssertionError):
        module.assert_exact_replay(data, saved)


@pytest.mark.parametrize('classes', [4, 6, 7])
def test_small_probe_uses_requested_class_count_and_saves_predictions(tmp_path, classes):
    from experiments.osram_meld_information_availability_20261014.probe import fit_probe
    rng = np.random.RandomState(66)
    x = rng.randn(16, 9).astype(np.float32)
    y = np.arange(16) % classes
    # Default seven-class behavior must remain available without a new argument.
    kwargs = {} if classes == 7 else {'classes': classes}
    fit_probe(x, y, x, y, seed=66, epochs=2, destination=tmp_path/'probe',
              batch_size=8, device='cpu', **kwargs)
    with np.load(tmp_path/'probe/predictions.npz') as saved:
        assert saved['logits'].shape == (16, classes)
        np.testing.assert_array_equal(saved['labels'], y)
        np.testing.assert_array_equal(saved['predictions'], saved['logits'].argmax(1))


def test_source_activation_is_explicit_and_does_not_use_the_worktree_numerics():
    module = impl()
    assert module.TRAINING_SHA == '785a01242ef1a82b97d5337acbb33ba963df5537'
    assert len(module.PROBE_SEEDS) == 3
    assert module.BACKBONE_SEED == 66
