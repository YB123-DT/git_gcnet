from pathlib import Path

import pytest


def test_resource_gate_uses_available_not_free_ram(tmp_path):
    from experiments.osram_text_path_router_3seed_20261019.resource_gate import available_gib
    path=tmp_path/'meminfo'
    path.write_text('MemTotal: 526385152 kB\nMemFree: 1048576 kB\nMemAvailable: 67108864 kB\n')
    assert available_gib(path)==64.


def test_resource_gate_missing_available_fails_closed(tmp_path):
    from experiments.osram_text_path_router_3seed_20261019.resource_gate import available_gib
    path=tmp_path/'meminfo';path.write_text('MemFree: 100 kB\n')
    with pytest.raises(ValueError,match='MemAvailable'):
        available_gib(path)


def test_resource_gate_refuses_duplicate_submission(tmp_path):
    from experiments.osram_text_path_router_3seed_20261019.resource_gate import validate_plan
    output=tmp_path/'old';output.mkdir()
    plan=dict(minimum_available_gib=64,source_directory=str(tmp_path/'source'),
              jobs=[dict(output=str(output),command=['python','-m','experiment'])])
    with pytest.raises(FileExistsError):validate_plan(plan)
