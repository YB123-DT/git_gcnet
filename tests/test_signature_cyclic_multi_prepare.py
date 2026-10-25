"""Frozen parents and exact masks cannot drift when extending cyclic stage two."""
from dataclasses import asdict
import json
from pathlib import Path

import pytest


def test_meld_environment_is_bound_before_first_config_import(tmp_path):
    import os
    import subprocess
    import sys
    script = '''
import sys
from pathlib import Path
from experiments.osram_signature_cyclic_multi_20261025 import prepare as p
assert 'config' not in sys.modules
p.configure_environment('MELD', Path(sys.argv[1]))
import config
assert config.PATH_TO_LABEL['MELD'] == str(p.MELD_FEATURE)
'''
    subprocess.run([sys.executable,'-c',script,str(tmp_path)],check=True,env=os.environ.copy())


def test_parent_paths_cover_only_authorized_completed_parents():
    from experiments.osram_signature_cyclic_multi_20261025 import prepare as p
    assert p.parent_path('CMUMOSI',66) == p.previous.MOSI66
    assert str(p.parent_path('CMUMOSI',67)).endswith('CMUMOSI/seed_67/flat')
    assert p.parent_path('CMUMOSEI',66) == p.previous.MOSEI66
    for seed in (67,68):
        assert str(p.parent_path('CMUMOSEI',seed)).endswith(f'CMUMOSEI/seed_{seed}/flat')
    assert str(p.parent_path('MELD',66)).endswith('runs_v1/S66_WD1')
    for dataset,seed in [('CMUMOSI',68),('MELD',67),('IEMOCAPSix',66)]:
        with pytest.raises(ValueError):
            p.parent_path(dataset,seed)


def test_mosei_masks_use_explicit_common_bank_but_predictions_use_own_parent():
    from experiments.osram_signature_cyclic_multi_20261025 import prepare as p
    for seed in (66,67,68):
        assert p.mask_path('CMUMOSEI',seed) == p.previous.UNIFIED/'CMUMOSEI'/'seed_66'/'flat'
        if seed != 66:
            assert p.mask_path('CMUMOSEI',seed) != p.parent_path('CMUMOSEI',seed)


def test_meld_registry_pins_every_original_checkpoint_and_exact_mask():
    from experiments.osram_signature_cyclic_multi_20261025 import prepare as p
    case=p.meld_case()
    assert case['expected_config']['learning_rate'] == 1e-5
    assert case['expected_config']['weight_decay'] == 1e-5
    assert case['expected_config']['missing_mask_protocol'] == 'gcnet-random-mask-v1'
    assert case['expected_config']['osram_write_step'] == .6
    assert set(case['evaluation_references']) == set(p.RATES)
    assert case['dataset_sha256'] == 'a0349614b2b58e8b2db768f0144aa36b96fb2348ca436a6879ee2623cc2d2f80'
    for rate,ref in case['evaluation_references'].items():
        assert str(p.parent_path('MELD',66)) in ref['checkpoint']
        assert f'best_miss_{rate.replace(".","p")}.pt' in ref['checkpoint']
        assert all(len(ref[k]) == 64 for k in ('checkpoint_sha256','selection_sha256','mask_sha256'))


def test_meld_config_accepts_only_reviewed_inactive_added_defaults():
    from experiments.osram_signature_cyclic_multi_20261025 import prepare as p
    raw=p.meld_case()['expected_config']
    cfg,added=p.meld_config(raw)
    assert added == p.MELD_ADDED_DEFAULTS
    assert cfg.osram_disable_local_skip is False
    with pytest.raises(ValueError,match='protocol'):
        p.meld_config(dict(raw,missing_mask_protocol='conversation-mask-v2'))
    with pytest.raises(ValueError,match='protocol'):
        p.meld_config(dict(raw,osram_source_relation=True))


def test_read_selected_meld_array_rejects_changed_artifact(tmp_path,monkeypatch):
    import numpy as np
    from experiments.osram_signature_cyclic_multi_20261025 import prepare as p
    path=tmp_path/'selected.npz'
    np.savez(path,availability=np.ones((2,3),dtype=np.float32),labels=np.array([0,1]),
             logits=np.eye(7,dtype=np.float32)[:2],predictions=np.array([0,1]))
    ref=dict(selection=str(path),selection_sha256=p.h.sha(path),
             mask_sha256=p.hashlib.sha256(np.ones((2,3),dtype=np.float32).tobytes()).hexdigest())
    assert p.selected_meld_arrays(ref)['logits'].shape == (2,7)
    with pytest.raises(ValueError,match='prediction'):
        p.selected_meld_arrays(dict(ref,selection_sha256='0'*64))
    with pytest.raises(ValueError,match='mask'):
        p.selected_meld_arrays(dict(ref,mask_sha256='0'*64))


def test_meld_cache_replays_selected_logits_and_masks_without_extra_parent_pass(monkeypatch):
    import numpy as np
    import torch
    from types import SimpleNamespace
    from experiments.osram_signature_cyclic_multi_20261025 import prepare as p
    from osram_four_block_wiring import FrozenFlatReader
    from tests.test_four_block_reader import Parent
    parent=Parent()
    parent.smax_fc=torch.nn.Linear(1600,7)
    reader=FrozenFlatReader(parent)
    x=torch.randn(2,1,1600)
    a=torch.tensor([[[1.,0.,1.]],[[0.,1.,0.]]])
    u=torch.ones(1,2)
    logits=reader(x,a,None,u,[2]).prediction.transpose(0,1)[u.bool()].numpy()
    parent.scans=0
    meta=dict(labels=np.array([0,1]),speakers=np.array([0.,1.]),
              conversation_ids=np.array(['c','c']),utterance_indices=np.array([0,1]))
    saved=dict(meta,availability=a.transpose(0,1)[u.bool()].numpy(),logits=logits,
               predictions=logits.argmax(-1))
    view=dict(incomplete=x,qmask=None,umask=u,lengths=[2],labels=torch.tensor([[0,1]]),conversation_ids=['c'])
    raw=[None]*8
    raw[7]=u
    class Loader(list):
        sampler=SimpleNamespace(set_epoch=lambda epoch:None)
    trainer=SimpleNamespace(_move_batch=lambda data,device:data,
        _prepare_view_from_primary_masks=lambda *args:view)
    monkeypatch.setattr(p.h,'metadata',lambda data:meta)
    proof=dict(rate='0.7',selected_prediction_sha256='prediction',test_mask_sha256='mask')
    bundle=dict(cfg=SimpleNamespace(dataset='MELD'),parent_identity=proof,
        selected_meld_arrays=saved,trainer=trainer,test_loader=Loader([raw]),dimensions=(1,1,1))
    rows=p.test_cache(bundle,reader,'0.7')
    assert len(rows) == 1 and parent.scans == 1
    assert proof['saved_prediction_replay']['logits_exact']
    assert proof['saved_prediction_replay']['max_absolute_error'] == 0.
    np.testing.assert_array_equal(rows[0]['availability'],saved['availability'])
    assert not rows[0]['frame'].prediction.requires_grad
    with pytest.raises(ValueError,match='match'):
        p.test_cache(bundle,reader,'0.3')
    saved['logits']=logits+1
    with pytest.raises(AssertionError):
        p.test_cache(bundle,reader,'0.7')
