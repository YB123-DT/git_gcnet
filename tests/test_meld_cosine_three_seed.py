import copy
import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def run():
    try:
        return importlib.import_module('experiments.osram_meld_cosine_step03_3seed_20261017.run')
    except ModuleNotFoundError:
        pytest.fail('Finite three-seed cosine replication is not implemented')


def base():
    return json.loads((ROOT/'experiments/osram_meld_memory_sensitivity_20261017/FROZEN_SUMMARY.json').read_text())['source_config']


def test_five_jobs_reuse_completed_cosine66_and_change_only_seed_schedule_step():
    source = base(); original = copy.deepcopy(source)
    expected = {'C67':(67,.6),'C68':(68,.6),'S66':(66,.3),'S67':(67,.3),'S68':(68,.3)}
    assert run().JOB_SPECS == expected
    for name,(seed,step) in expected.items():
        cfg = run().make_config(source,name)
        assert cfg == dict(source,seed=seed,lr_schedule='cosine',osram_write_step=step)
    assert source == original
    with pytest.raises(ValueError):
        run().make_config(source,'C66')


def test_control_references_are_seed_matched(tmp_path):
    root, reused = tmp_path/'new', tmp_path/'completed66'
    controls = {name:run().control_for(name,root,reused) for name in run().JOB_SPECS}
    assert controls == {'C67':None,'C68':None,'S66':reused,'S67':root/'C67','S68':root/'C68'}


def test_second_stage_waits_and_propagates_failed_replication(tmp_path,monkeypatch):
    module = run()
    scheduler = module.ReplicationScheduler(tmp_path/'queue',source=tmp_path/'b4',reused=tmp_path/'C0')
    assert scheduler.gpus == (5,) and scheduler.capacity == {5:4}
    monkeypatch.setattr(module.Scheduler,'admit',lambda self,row:(5,module.GPU_UUID))
    first = {'id':'C67','method':'Flat'}
    assert scheduler.admit(first)==(5,module.GPU_UUID)
    second = dict(id='S66',method='Flat',status='pending',failures=[])
    assert scheduler.admit(second) is None
    scheduler.state['jobs'].update(C67={'status':'complete'},C68={'status':'complete'})
    assert scheduler.admit(second)==(5,module.GPU_UUID)
    scheduler.state['jobs']['C68']['status']='failed'
    assert scheduler.admit(second) is None and second['status']=='failed'


def test_unpaired_replication_does_not_claim_seed66_alignment(tmp_path,monkeypatch):
    cosine = importlib.import_module('experiments.osram_meld_cosine_20261017.run')
    calls=[]
    def parent(base,capture,output,variant,control):
        calls.append(variant)
        return object
    monkeypatch.setattr(cosine,'_original_state',parent)
    cosine.state_factory(base(),paired_control=False)(object,object(),tmp_path,'C67',tmp_path/'C0')
    assert calls==['W0']  # Skip historical seed66 identity comparison ONLY for unpaired controls.
    cosine.state_factory(base())(object,object(),tmp_path,'S67',tmp_path/'C67')
    assert calls==['W0','S67']  # Default paired behavior stays unchanged.


def test_worker_command_has_independent_output_and_uuid(tmp_path):
    module = run()
    scheduler = module.ReplicationScheduler(tmp_path/'runs/queue',source=tmp_path/'b4',reused=tmp_path/'C0')
    cmd = scheduler.command(dict(id='S68',output=str(tmp_path/'runs/S68')),tmp_path/'cfg.json',module.GPU_UUID)
    assert cmd[cmd.index('--job')+1]=='S68'
    assert cmd[cmd.index('--gpu-uuid')+1]==module.GPU_UUID
    assert cmd[cmd.index('--reused')+1]==str(tmp_path/'C0')
