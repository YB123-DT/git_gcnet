from argparse import Namespace
import inspect
from pathlib import Path

import pytest


def test_only_four_new_grid_commands_no_completed_duplicate():
    from experiments.osram_mosei_v3_grid_20261022.run import worker_command, MODES
    args = Namespace(output=Path('/test'), source_commit='abc')
    assert set(MODES) == {'f512_plain', 'f512_v128', 'f512_v256', 'f1024_v128'}
    for mode in MODES:
        cmd = worker_command(args, mode, 'train', 'GPU-test')
        assert cmd[cmd.index('--seed') + 1] == '66'
        assert cmd[cmd.index('--mode') + 1] == mode
        assert cmd[cmd.index('--output') + 1] == '/test/runs/' + mode
    for old in ('small_flat', 'wide_v3', 'f1024_plain', 'f1024_v256'):
        with pytest.raises(ValueError):
            worker_command(args, old, 'train', 'GPU-test')


def test_shared_queue_default_scope_is_unchanged():
    from experiments.osram_mosi_v3_capacity_20261022.run import main
    from experiments.osram_mosi_v3_capacity_20261022.worker import MODES
    s = inspect.signature(main)
    assert s.parameters['modes'].default == MODES
    assert s.parameters['default_per_gpu'].default == 3
    assert s.parameters['max_per_gpu'].default == 3


def test_incomplete_new_grid_arm_is_not_scored(tmp_path):
    import json
    from experiments.osram_mosei_v3_grid_20261022.report import summarize
    d = tmp_path / 'runs' / 'f512_v128'; d.mkdir(parents=True)
    (d / 'SUMMARY.json').write_text(json.dumps({'status': 'running', 'epoch': 1}))
    result = summarize(tmp_path, include_reused=False)
    assert result['per_rate'] == [] and result['aggregate'] == []
    assert result['states']['f512_v128']['status'] == 'running'
