"""Complete the old fixed-rate NeuMiss screen without changing its mathematics."""
import pytest


def test_only_add_missing_seeds_and_preserve_old_budget():
    from experiments.osram_neumiss_stage2_3seed_20261025.run import validate_seed, learning_rate, epoch_budget
    for seed in (67,68):
        validate_seed(seed)
    for seed in (66,69,True):
        with pytest.raises(ValueError):validate_seed(seed)
    assert epoch_budget('0.0') == 0
    assert epoch_budget('0.7') == 150
    assert epoch_budget('0.7',smoke=True) == 2
    assert learning_rate(99) == 3e-4
    assert learning_rate(100) == 3e-4/10


def test_full_observation_neumiss_remains_hard_zero_even_after_learning():
    import torch
    from types import SimpleNamespace
    from osram_four_block_wiring import NeuMissBlock
    block=NeuMissBlock()
    with torch.no_grad():
        for p in block.parameters():p.fill_(.1)
    frame=SimpleNamespace(latents=torch.randn(2,1,3,256),context=torch.randn(2,1,2304),
        availability=torch.ones(2,1,3,dtype=torch.bool),valid=torch.ones(2,1,dtype=torch.bool))
    assert torch.count_nonzero(block(frame)) == 0
