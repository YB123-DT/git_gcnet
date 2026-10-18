import importlib
import types
from pathlib import Path

import pytest


def module():
    try:
        return importlib.import_module('experiments.osram_meld_wd_seed66_20261018.resources')
    except ModuleNotFoundError:
        pytest.fail('Explicit scheduling-only resource adapter missing')


def test_only_healthy_zero_and_two_are_allowed():
    r = module()
    assert set(r.GPU_UUIDS) == {0,2}
    assert 4 not in r.GPU_UUIDS
    assert r.GPU_UUIDS[0] == 'GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45'


def test_worker_command_preserves_frozen_identity_and_config(tmp_path):
    r = module()
    args = types.SimpleNamespace(frozen_source=tmp_path/'source',output=tmp_path/'runs',
        wrapper_commit='resource-only-commit')
    scheduler = r.ResourceScheduler(tmp_path/'queue',args,python='/fixed/python')
    job = dict(id='S66_WD1',output=str(tmp_path/'runs/S66_WD1'))
    cmd = scheduler.command(job,tmp_path/'unchanged.json',r.GPU_UUIDS[0])
    assert cmd[cmd.index('--driver-commit')+1] == r.ORIGINAL_COMMIT
    assert cmd[cmd.index('--config')+1] == str(tmp_path/'unchanged.json')
    assert cmd[cmd.index('--wrapper-commit')+1] == 'resource-only-commit'
    assert scheduler.capacity == {0:4,2:4}


def test_resource_adapter_loads_only_expected_frozen_driver():
    r = module()
    assert r.FROZEN_DRIVER_SHA == '03c4d962586dd346d61008d5f14896ec11b9f32e680391937cfec055ed411d17'
    with pytest.raises(ValueError):
        r.load_frozen(Path('/not/a/frozen/source'))
