import importlib

import pytest
import torch


def module():
    name = 'experiments.osram_meld_persistent_unimodal_20261018.run'
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        pytest.fail('Current-baseline persistent unimodal evaluator is missing')


@pytest.mark.parametrize('pattern,expected', [('A', [1, 0, 0]), ('T', [0, 1, 0]), ('V', [0, 0, 1])])
def test_fixed_pattern_is_constant_over_conversations_and_history_with_zero_padding(pattern, expected):
    fn = module().fixed_availability
    umask = torch.tensor([[1., 1., 1.], [1., 1., 0.]])
    before = torch.get_rng_state().clone()
    result = fn(umask, pattern)
    assert result.shape == (3, 2, 3)
    assert torch.equal(result[umask.T.bool()], torch.tensor(expected).float().expand(5, -1))
    assert torch.count_nonzero(result[~umask.T.bool()]) == 0
    assert torch.equal(before, torch.get_rng_state())


def test_invalid_pattern_and_padding_mask_rejected():
    fn = module().fixed_availability
    with pytest.raises(ValueError):
        fn(torch.ones(2, 3), 'AT')
    with pytest.raises(ValueError):
        fn(torch.ones(2, 3, 1), 'A')
    with pytest.raises(ValueError):
        fn(torch.tensor([[1., float('nan')]]), 'A')
