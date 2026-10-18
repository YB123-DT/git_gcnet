"""The finite TASR adapter protocol must not modify baseline mask RNG."""
import importlib
import math
import random

import numpy as np
import pytest
import torch


def protocol():
    return importlib.import_module('experiments.osram_tasr_20261018.protocol')


def test_fixed_protected_sources_use_one_checkpoint_per_case():
    cases = protocol().build_cases()
    assert set(cases) == {'MELD', 'IEMOCAPSix'}
    for dataset, case in cases.items():
        assert case['dataset'] == dataset
        assert case['seed'] == 66
        assert case['fold'] == (1 if dataset == 'MELD' else 5)
        assert case['anchor_rate'] == '0.3'
        assert case['checkpoint'].endswith('/best_miss_0p3.pt')
        assert len(case['checkpoint_sha256']) == 64
        assert case['expected_config']['missing_mask_protocol'] == 'gcnet-random-mask-v1'
        assert case['expected_config']['training_objective'] == 'emotion-only'
        assert case['expected_config']['osram_write_step'] == .6
        assert case['expected_config']['osram_readout_fusion'] == 'flat'
        assert set(case['evaluation_references']) == {f'{i/10:.1f}' for i in range(8)}
        assert case['checkpoint_sha256'] == case['evaluation_references']['0.3']['checkpoint_sha256']
        assert 'test-oracle' in case['source_selection'].lower()
    assert cases['MELD']['expected_config']['learning_rate'] == 1e-5
    assert cases['MELD']['expected_config']['weight_decay'] == 1e-5
    assert cases['MELD']['checkpoint_sha256'] == '6f031b515723b441129e16a2e6217410e0eceee2c468e1cbe7b7c3d8276bdb7d'
    assert cases['IEMOCAPSix']['checkpoint_sha256'] == '6a7968ade42513ee868e6dc82f306274a2e7bd53d1943c05f00c1a75af530fd0'


def test_persistent_view_uses_six_patterns_and_keeps_padding_zero():
    umask = torch.ones(6000, 4)
    umask[:100, 2:] = 0
    availability, ids = protocol().persistent_availability(umask, seed=66, epoch=2, batch_index=3)
    assert availability.shape == (4, 6000, 3)
    assert ids.shape == (6000,)
    assert availability[2:, :100].count_nonzero() == 0
    assert torch.all(availability.sum(-1).T[umask.bool()] >= 1)
    assert torch.all(availability.sum(-1) <= 2)
    for b in (0, 99, 100, 5999):
        assert torch.equal(availability[:, b][umask[b].bool()], availability[0, b].expand(int(umask[b].sum()), -1))
    counts = torch.bincount(ids, minlength=6)
    assert all(800 < int(n) < 1200 for n in counts)


def test_persistent_view_is_deterministic_without_global_rng_mutation():
    torch.manual_seed(31)
    np.random.seed(32)
    random.seed(33)
    states = torch.get_rng_state().clone(), np.random.get_state(), random.getstate()
    args = dict(seed=66, epoch=1, batch_index=4)
    first = protocol().persistent_availability(torch.ones(100, 5), **args)
    second = protocol().persistent_availability(torch.ones(100, 5), **args)
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    assert torch.equal(states[0], torch.get_rng_state())
    actual_np = np.random.get_state()
    assert states[1][0] == actual_np[0] and np.array_equal(states[1][1], actual_np[1]) and states[1][2:] == actual_np[2:]
    assert states[2] == random.getstate()
    other = protocol().persistent_availability(torch.ones(100, 5), **dict(args, epoch=2))
    assert not torch.equal(first[1], other[1])


@pytest.mark.parametrize('bad', [torch.ones(3), torch.full((2, 3), .5), torch.tensor([[1., float('nan')]])])
def test_persistent_rejects_invalid_padding_masks(bad):
    with pytest.raises(ValueError):
        protocol().persistent_availability(bad, seed=66, epoch=1, batch_index=0)


def test_adapter_validation_is_a_deterministic_conversation_subset_only():
    vids = [f'Ses0{i//10+1}F_impro{i:02}' for i in range(50)]
    indices = tuple(range(40))  # held-out session 5 cannot enter either side.
    train, val, meta = protocol().adapter_split_indices(indices, vids)
    assert len(val) == math.ceil(len(indices)*.1)
    assert set(train).isdisjoint(val)
    assert set(train) | set(val) == set(indices)
    assert not (set(train) | set(val)).intersection(range(40, 50))
    assert (train, val, meta) == protocol().adapter_split_indices(tuple(reversed(indices)), vids)
    assert meta['backbone_previously_saw_adapter_validation'] is True
    assert meta['uses_labels'] is False
    assert meta['validation_conversations'] == [vids[i] for i in val]


@pytest.mark.parametrize('indices,vids,fraction', [([0], ['one'], .1), ([0, 0], ['one'], .1), ([0, 1], ['one', 'one'], .1), ([0, 1], ['one', 'two'], 0), ([0, 2], ['one', 'two'], .1)])
def test_adapter_split_rejects_degenerate_inputs(indices, vids, fraction):
    with pytest.raises(ValueError):
        protocol().adapter_split_indices(indices, vids, fraction=fraction)


def test_one_validation_checkpoint_uses_random_acc_then_ce_only():
    rates = {f'{r/10:.1f}': {'ACC': 50+r, 'CE': 2-r/10} for r in range(8)}
    assert protocol().validation_key(rates) == pytest.approx((53.5, -1.65))
    lower_ce = {r: dict(m, CE=m['CE']-.1) for r,m in rates.items()}
    assert protocol().validation_key(lower_ce) > protocol().validation_key(rates)
    bad = dict(rates); bad.pop('0.7')
    with pytest.raises(ValueError): protocol().validation_key(bad)
    bad = dict(rates); bad['0.7'] = {'ACC': float('nan'), 'CE': 1.}
    with pytest.raises(ValueError): protocol().validation_key(bad)


def test_declared_training_budget_is_fixed_and_no_test_selection():
    config = protocol().TRAINING_PROTOCOL
    assert config['epochs'] == 30
    assert config['learning_rate'] == 3e-4
    assert config['weight_decay'] == 1e-5
    assert config['batch_size'] == 32
    assert config['random_weight'] == .8 and config['persistent_weight'] == .2
    assert config['checkpoint_selection'] == 'validation-random8-ACC-then-CE'
