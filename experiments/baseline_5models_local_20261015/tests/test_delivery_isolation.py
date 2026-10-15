"""Delivery defaults must not attach a fresh checkout to active local runs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_default_run_root_and_data_cache_are_checkout_local(tmp_path):
    env = dict(os.environ)
    env.pop('BASELINE_RUN_ROOT', None)
    env['PYTHONPATH'] = str(ROOT)
    script = "import runtime,common_data,json; print(json.dumps([str(runtime.RUN_ROOT),str(common_data.CACHE_ROOT)]))"
    paths = json.loads(subprocess.check_output([sys.executable,'-c',script],cwd=tmp_path,env=env,text=True))
    expected = ROOT.parents[1] / 'runs' / ROOT.name
    assert paths == [str(expected),str(expected / 'data_cache')]
    assert str(expected) != '/data2/yb/paper/runs/baseline_5models_local_20261015'


def test_original_equivalence_reference_is_complete_and_verbatim():
    reference = ROOT / 'tests/reference_momke'
    info = json.loads((reference/'SOURCE.json').read_text())
    assert info['commit'] == '2dceb57f9e6f4bc84c2fc8f4eb7c761ea64c9efc'
    for relative, expected in info['files'].items():
        assert hashlib.sha256((reference/relative).read_bytes()).hexdigest() == expected


def test_frozen_defaults_match_executable_model_configuration():
    from configure import config
    frozen = json.loads((ROOT/'configs/frozen_defaults.json').read_text())
    assert len(frozen['templates']) == 21
    for key, expected in frozen['templates'].items():
        dataset,model,version = key.split('/')
        actual = config(model,version,dataset,66,.3,0 if dataset=='MELD' else 1)
        actual = {k:v for k,v in actual.items() if k not in ('seed','missing_rate','fold')}
        assert actual == expected
