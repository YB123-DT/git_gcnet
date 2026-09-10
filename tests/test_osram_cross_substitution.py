import importlib.util
from pathlib import Path
import torch


def test_five_patches_only_change_eligible_slots_without_mutation():
    path = Path(__file__).resolve().parents[1] / 'experiments/osram_cross_substitution_20260910/run.py'
    assert path.exists(), 'cross-substitution implementation missing'
    spec = importlib.util.spec_from_file_location('cross_patch', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    local = torch.arange(8.).reshape(4, 1, 2)
    base = torch.arange(12.).reshape(4, 1, 3) + 10
    gap = torch.arange(36.).reshape(4, 1, 3, 3) + 100
    availability = torch.tensor([[[1, 0, 1]], [[0, 1, 0]], [[1, 1, 1]], [[1, 1, 0]]])
    valid = torch.tensor([[True], [True], [True], [False]])
    snapshots = [x.clone() for x in (local, base, gap)]
    results = module.patched_inputs(local, base, gap, availability, valid)
    assert set(results) == {'normal', 'no_gap', 'no_base', 'base_to_gap', 'gap_to_base',
                            'move_base_to_gap', 'move_gap_to_base', 'swap'}
    original = torch.cat([local, base, (gap * (1-availability)[...,None]).flatten(-2)], -1)
    assert torch.equal(results['normal'], original)
    for result in results.values():
        assert torch.equal(result[1:], original[1:])
        assert torch.equal(result[..., :2], local)
    assert torch.count_nonzero(results['no_gap'][0,0,5:]) == 0
    assert torch.count_nonzero(results['no_base'][0,0,2:5]) == 0
    assert torch.equal(results['base_to_gap'][0,0,8:11], base[0,0])
    assert torch.count_nonzero(results['base_to_gap'][0,0,5:8]) == 0
    assert torch.count_nonzero(results['base_to_gap'][0,0,11:]) == 0
    assert torch.equal(results['gap_to_base'][0,0,2:5], gap[0,0,1])
    assert torch.equal(results['gap_to_base'][...,5:], original[...,5:])
    assert torch.count_nonzero(results['move_base_to_gap'][0,0,2:5]) == 0
    assert torch.equal(results['move_base_to_gap'][0,0,5:], results['base_to_gap'][0,0,5:])
    assert torch.count_nonzero(results['move_gap_to_base'][0,0,5:]) == 0
    assert torch.equal(results['move_gap_to_base'][0,0,2:5], gap[0,0,1])
    assert torch.equal(results['swap'][0,0,2:5], gap[0,0,1])
    assert torch.equal(results['swap'][0,0,5:], results['base_to_gap'][0,0,5:])
    assert all(torch.equal(a,b) for a,b in zip(snapshots,(local,base,gap)))
