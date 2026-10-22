from argparse import Namespace
from pathlib import Path

import pytest


def test_only_two_supplement_commands_never_repeat_seed66():
    from experiments.osram_mosi_v3_capacity_3seed_20261022.run import worker_command
    args = Namespace(output=Path('/test'), source_commit='abc', gpu_uuid='GPU-test')
    for seed in (67, 68):
        cmd = worker_command(args, seed, 'train')
        assert cmd[cmd.index('--seed') + 1] == str(seed)
        assert cmd[cmd.index('--mode') + 1] == 'f512_v128'
        assert cmd[cmd.index('--output') + 1] == f'/test/runs/seed_{seed}'
    with pytest.raises(ValueError):
        worker_command(args, 66, 'train')


def test_three_seed_aggregate_requires_three_complete_candidates():
    from experiments.osram_mosi_v3_capacity_3seed_20261022.report import three_seed_stats
    assert three_seed_stats({66: {'mean': {'WF1': 79.}}}) is None
    assert three_seed_stats({66: {}, 67: {}}) is None
    runs = {seed: {'mean': {'ACC': score, 'WF1': score}, 'high': {'ACC': score, 'WF1': score}}
            for seed, score in ((66, 79.), (67, 80.), (68, 81.))}
    s = three_seed_stats(runs)
    assert s['mean']['WF1'] == {'mean': 80., 'std': 1.}
