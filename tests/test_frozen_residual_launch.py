import importlib


def test_cpu_launch_builds_disjoint_rate_shards():
    launcher = importlib.import_module('experiments.osram_frozen_residual_20261009.launch')
    commands = launcher.commands('/tmp/run', '/tmp/audit', '/usr/bin/python')
    assert len(commands) == 2
    covered = []
    for name, command in commands:
        assert command[command.index('--device') + 1] == 'cpu'
        assert command[command.index('--epochs') + 1] == '100'
        assert command[command.index('--output') + 1] == '/tmp/run/runs/' + name
        covered.extend(float(x) for x in command[command.index('--rates') + 1:])
    assert sorted(covered) == [i / 10 for i in range(8)]
