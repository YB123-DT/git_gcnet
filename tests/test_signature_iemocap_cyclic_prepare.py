"""Accepted original IEMOCAP parents, LOSO identity and exact-mask replay."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


def test_environment_is_bound_before_trainer_import(tmp_path):
    script = '''
import os,sys
from experiments.osram_signature_iemocap_cyclic_20261025 import prepare as p
assert 'config' not in sys.modules
p.configure_environment(sys.argv[1])
import config
assert str(p.DATA_ROOT) in config.PATH_TO_LABEL['IEMOCAPSix']
assert 'GCNET_MELD_PICKLE' not in os.environ
'''
    subprocess.run([sys.executable, '-c', script, str(tmp_path)], check=True)


def test_registered_parent_covers_three_seeds_five_folds_eight_rates():
    from experiments.osram_signature_iemocap_cyclic_20261025 import prepare as p
    for seed in (66, 67, 68):
        for fold in range(1, 6):
            case = p.registered_parent(seed, fold)
            assert case['seed'] == seed and case['fold'] == fold
            assert set(case['references']) == set(p.RATES)
            for rate, ref in case['references'].items():
                assert str(Path(case['output'])) in ref['checkpoint']
                assert all(len(ref[k]) == 64 for k in ('checkpoint_sha256', 'selection_sha256', 'mask_sha256'))
    for seed, fold in [(65, 5), (66, 0), (66, 6), (True, 1)]:
        with pytest.raises(ValueError):
            p.registered_parent(seed, fold)


def test_original_parent_configuration_cannot_change_missing_or_skip():
    from experiments.osram_signature_iemocap_cyclic_20261025 import prepare as p
    common = json.loads((p.ROOT/'experiments/osram_persistent_patterns_20261018/SUMMARY.json').read_text())
    raw = dict(common['datasets']['IEMOCAPSix']['common_training_config'], seed=67, fold=3)
    cfg, added = p.parent_config(raw, 67, 3)
    assert cfg.epochs == 100 and cfg.learning_rate == .001
    assert cfg.osram_write_step == .6 and not cfg.osram_disable_local_skip
    for key, value in [('missing_mask_protocol', 'conversation-mask-v2'), ('fold', 2),
                       ('osram_source_role_v3', True), ('osram_disable_local_skip', True)]:
        with pytest.raises(ValueError):
            p.parent_config(dict(raw, **{key: value}), 67, 3)


def test_selected_arrays_reject_mask_or_logit_class_corruption(tmp_path):
    from experiments.osram_signature_iemocap_cyclic_20261025 import prepare as p
    path = tmp_path/'selected.npz'
    a = np.ones((2, 3), dtype=np.float32)
    np.savez(path, availability=a, labels=np.array([0, 1]), logits=np.eye(6, dtype=np.float32)[:2], predictions=np.array([0, 1]))
    ref = dict(selection=str(path), selection_sha256=p.h.sha(path),
               mask_sha256=p.hashlib.sha256(a.tobytes()).hexdigest())
    assert p.selected_arrays(ref)['logits'].shape == (2, 6)
    with pytest.raises(ValueError):
        p.selected_arrays(dict(ref, mask_sha256='0'*64))
    with pytest.raises(ValueError):
        p.selected_arrays(dict(ref, selection_sha256='0'*64))


def test_config_does_not_accept_four_class_or_foreign_seed():
    from experiments.osram_signature_iemocap_cyclic_20261025 import prepare as p
    common = json.loads((p.ROOT/'experiments/osram_persistent_patterns_20261018/SUMMARY.json').read_text())
    raw = dict(common['datasets']['IEMOCAPSix']['common_training_config'], seed=66, fold=5)
    for key,value in [('dataset','IEMOCAPFour'),('seed',68),('learning_rate',1e-5)]:
        with pytest.raises(ValueError):
            p.parent_config(dict(raw, **{key:value}), 66, 5)


def test_selected_cache_preserves_exact_logits_mask_and_one_scan(monkeypatch):
    import torch
    from types import SimpleNamespace
    from osram_four_block_wiring import FrozenFlatReader
    from tests.test_four_block_reader import Parent
    from experiments.osram_signature_iemocap_cyclic_20261025 import prepare as p
    parent = Parent()
    parent.smax_fc = torch.nn.Linear(1600,6)
    reader = FrozenFlatReader(parent)
    x = torch.randn(2,1,1600)
    availability = torch.tensor([[[1.,0.,1.]],[[0.,1.,0.]]])
    umask = torch.ones(1,2)
    logits = reader(x,availability,None,umask,[2]).prediction.transpose(0,1)[umask.bool()].numpy()
    parent.scans = 0
    meta = dict(labels=np.array([0,1]),speakers=np.array([0.,1.]),
        conversation_ids=np.array(['c','c']),utterance_indices=np.array([0,1]))
    saved = dict(meta, availability=availability.transpose(0,1)[umask.bool()].numpy(),
                 logits=logits,predictions=logits.argmax(-1))
    view = dict(incomplete=x,qmask=None,umask=umask,lengths=[2],labels=torch.tensor([[0,1]]),conversation_ids=['c'])
    raw = [None]*8
    raw[7] = umask
    class Loader(list):
        sampler = SimpleNamespace(set_epoch=lambda epoch:None)
    trainer = SimpleNamespace(_move_batch=lambda data,device:data,
        _prepare_view_from_primary_masks=lambda *args:view)
    monkeypatch.setattr(p.h,'metadata',lambda data:meta)
    proof = dict(seed=68,fold=5,rate='0.7',selected_prediction_sha256='prediction',test_mask_sha256='mask')
    bundle = dict(parent_identity=proof,selected_arrays=saved,trainer=trainer,
        test_loader=Loader([raw]),dimensions=(1,1,1))
    rows = p.test_cache(bundle,reader,'0.7')
    assert len(rows) == 1 and parent.scans == 1
    assert proof['saved_prediction_replay']['logits_exact']
    with pytest.raises(ValueError,match='rate'):
        p.test_cache(bundle,reader,'0.6')
    saved['labels'] = np.array([1,0])
    with pytest.raises(AssertionError):
        p.test_cache(bundle,reader,'0.7')


def test_unreviewed_core_drift_is_never_silently_accepted(monkeypatch):
    from types import SimpleNamespace
    from experiments.osram_signature_iemocap_cyclic_20261025 import prepare as p
    cfg = SimpleNamespace(osram_meaningful_block='none')
    hashes = {name:pair['current'] for name,pair in p.DISABLED_MBT_HASHES.items()}
    hashes['gcnet_modality_jepa/gcnet_legacy_random_mask.py'] = '0'*64
    with pytest.raises(ValueError,match='Unreviewed changed core'):
        p.core_differences(hashes,cfg)


def test_seed66_environment_uses_retained_launcher_proof_not_invented_sklearn(tmp_path):
    from experiments.osram_signature_iemocap_cyclic_20261025 import prepare as p
    path = tmp_path/'runs/iemocap6/seed_66/fold_5'
    path.mkdir(parents=True)
    (tmp_path/'MANIFEST.json').write_text(json.dumps(dict(provenance=dict(environment=dict(
        executable='/data2/yb/conda_envs/cmerc-repro/bin/python')))))
    actual = dict(torch='1.13.1+cu117', numpy='1.26.4', sklearn='1.3.2')
    recorded = dict(provenance=dict(environment=dict(torch=actual['torch'],numpy=actual['numpy'])))
    proof = p.parent_environment(recorded,actual,66,path)
    assert proof['unrecorded_historical_versions'] == ['sklearn']
    assert len(proof['launcher_manifest_sha256']) == 64
    with pytest.raises(ValueError):
        p.parent_environment(recorded,actual,67,path)
    with pytest.raises(ValueError):
        p.parent_environment(dict(provenance=dict(environment=dict(torch='wrong',numpy=actual['numpy']))),actual,66,path)
