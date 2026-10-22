from pathlib import Path
from types import SimpleNamespace

import pytest


def test_direct_controller_has_only_smoke_and_single_seed66_train():
    from experiments.osram_mosi_v3_direct_20261022.run import worker_command
    args = SimpleNamespace(output=Path('/tmp/direct-only'), gpu_uuid='GPU-healthy', source_commit='sha')
    for phase in ('smoke', 'train'):
        cmd = worker_command(args, phase)
        assert cmd[cmd.index('--mode')+1] == 'f512_v128'
        assert cmd[cmd.index('--seed')+1] == '66'
        assert cmd[cmd.index('--phase')+1] == phase
        assert 'experiments.osram_mosi_v3_direct_20261022.worker' in cmd
    with pytest.raises(ValueError):
        worker_command(args, 'more_seeds')


def test_historical_selection_uses_first100_and_earliest_ties_only():
    from experiments.osram_mosi_v3_direct_20261022.report import select_history
    rates = [f'{i/10:.1f}' for i in range(8)]
    rows = [dict(epoch=i+1, test={r: {'router': dict(ACC=50., WF1=50., MSE=1.)}
                                  for r in rates}) for i in range(150)]
    rows[100]['test']['0.0']['router']['WF1'] = 99.
    rows[50]['test']['0.1']['router']['WF1'] = 51.
    best = select_history(rows, 100)
    assert best['0.0']['epoch'] == 1
    assert best['0.0']['metrics']['WF1'] == 50.
    assert best['0.1']['epoch'] == 51
    with pytest.raises(ValueError):
        select_history(rows[:99], 100)
