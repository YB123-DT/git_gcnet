from argparse import Namespace
from pathlib import Path

import pytest


def test_worker_commands_cover_only_six_approved_seed66_arms():
    from experiments.osram_mosi_v3_capacity_20261022.run import worker_command
    from experiments.osram_mosi_v3_capacity_20261022.worker import MODES
    args = Namespace(output=Path('/test'), source_commit='abc')
    assert len(MODES) == 6
    for mode in MODES:
        cmd = worker_command(args, mode, 'train', 'GPU-healthy')
        assert cmd[cmd.index('--seed') + 1] == '66'
        assert cmd[cmd.index('--mode') + 1] == mode
        assert cmd[cmd.index('--gpu-uuid') + 1] == 'GPU-healthy'
        assert cmd[cmd.index('--output') + 1] == '/test/runs/' + mode
    with pytest.raises(ValueError):
        worker_command(args, 'extra_hpo', 'train', 'GPU-healthy')


def test_reservation_does_not_double_count_resident_peer():
    from experiments.osram_mosi_v3_capacity_20261022.run import admission_required
    assert admission_required(1000, []) == 3500
    # A peer already occupying its reserved memory does not reserve it twice.
    assert admission_required(1000, [(3000, 4500)]) == 3500
    assert admission_required(1000, [(3000, 1000)]) == 7000
    assert admission_required(1000, [(3000, 1000), (1000, 500)]) == 8000


def test_incomplete_runs_are_not_reported_as_performance(tmp_path):
    import json
    from experiments.osram_mosi_v3_capacity_20261022.report import summarize
    d = tmp_path / 'runs' / 'f512_v256'
    d.mkdir(parents=True)
    (d / 'SUMMARY.json').write_text(json.dumps({'status': 'running', 'epoch': 1}))
    result = summarize(tmp_path)
    assert not result['per_rate'] and not result['aggregate']
    assert result['states']['f512_v256']['status'] == 'running'
