from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from experiments.osram_signature_stage2_extension_20261025 import prepare
from osram_four_block_wiring import FrozenFlatReader
from tests.test_four_block_reader import Parent


@pytest.mark.parametrize('dataset,seed', [('CMUMOSI',66),('CMUMOSI',67),('CMUMOSI',68),('CMUMOSEI',66)])
def test_explicit_cmerc_parent_paths(dataset, seed):
    parent = prepare.parent_path(dataset, seed)
    if dataset == 'CMUMOSEI':
        assert 'nested_cmerc_tuning' in str(parent)
        assert str(parent).endswith('CMUMOSEI/fold_1/seed_66/none')
        assert 'cmerc_sentiment_3seed' in str(prepare.mask_path(dataset, seed))
    elif seed == 66:
        assert 'osram_mosi_cmerc_20261023/execution_v2' in str(parent)
    else:
        assert 'cmerc_sentiment_3seed' in str(parent)
        assert str(parent).endswith(f'CMUMOSI/seed_{seed}/flat')
    assert 'mid_lr_3seed' not in str(parent)


@pytest.mark.parametrize('dataset,seed', [('CMUMOSI',67),('CMUMOSI',68),('CMUMOSEI',66)])
def test_original_config_can_add_only_reviewed_disabled_defaults(dataset, seed):
    from gcnet_missing_m3.train_gcnet import TrainConfig
    cfg = TrainConfig(dataset=dataset, seed=seed, epochs=150 if dataset == 'CMUMOSI' else 60,
        learning_rate=3e-4 if dataset == 'CMUMOSI' else 1e-4, batch_size=32, weight_decay=1e-5,
        missing_mask_protocol='gcnet-random-mask-v1', checkpoint_selection='test-oracle-per-rate',
        checkpoint_selection_metric='weighted_f1', backbone_type='osram', training_objective='emotion-only',
        completion_path='none', osram_bidirectional=False, disable_unused_aux_modules=True,
        latent_dim=256, osram_output_dim=1600, osram_num_heads=8, osram_key_dim=64, osram_value_dim=64,
        osram_write_step=.6, dropout=.5, projector_dropout=.1)
    raw = asdict(cfg)
    for key in prepare.NEW_DEFAULTS:
        raw.pop(key)
    actual, added = prepare._config(raw, dataset, seed)
    assert asdict(actual) == asdict(cfg) and added == prepare.NEW_DEFAULTS
    with pytest.raises(ValueError, match='protocol'):
        prepare._config(dict(raw, learning_rate=.001), dataset, seed)
    with pytest.raises(ValueError, match='Unreviewed'):
        prepare._config({k:v for k,v in raw.items() if k != 'gradient_clip_norm'}, dataset, seed)


@pytest.mark.parametrize('dataset,seed', [('CMUMOSI',67),('CMUMOSI',68),('CMUMOSEI',66)])
def test_setup_strict_loads_requested_frozen_parent(monkeypatch, dataset, seed):
    rate = '0.6'
    fresh = nn.Linear(2, 1)
    @dataclass
    class Config:
        seed: int
    cfg = Config(seed)
    identity = dict(config=asdict(cfg), source_hashes={'gcnet_missing_m3/model.py':'same'},
                    rng_repair=dict(random_alias_repaired=True) if dataset == 'CMUMOSEI' else None)
    accepted = dict(status='complete', epoch=150 if dataset == 'CMUMOSI' else 60, identity=identity,
        best_by_rate={rate:dict(epoch=21, checkpoint='best06.pt', checkpoint_sha256='checkpoint')})
    monkeypatch.setattr(Path, 'read_text', lambda path: json.dumps(accepted))
    monkeypatch.setattr(prepare.h, 'sha', lambda path: 'checkpoint' if path.suffix == '.pt' else 'same')
    paths = []
    def load(path, **kwargs):
        paths.append(path)
        return dict(model=fresh.state_dict(), identity=identity, epoch=21, selection=dict(rate=rate))
    monkeypatch.setattr(prepare.torch, 'load', load)
    monkeypatch.setattr(prepare, '_build_bundle', lambda *args: dict(model=fresh, cfg=cfg,
        environment=dict(torch='test'), added_default_config_fields={}))
    bundle, reader, proof = prepare.setup(dataset, seed, 'GPU-test', Path('/unused-output'), rate)
    assert paths == [prepare.parent_path(dataset,seed)/'best06.pt']
    assert proof['parent_path'] == str(prepare.parent_path(dataset,seed))
    assert proof['test_rates'] == [rate] and proof['train_rate'] == rate
    assert proof['parent_rng_repair'] == identity['rng_repair']
    assert proof['saved_prediction_replay_required']
    assert reader.parent is fresh and all(not p.requires_grad for p in fresh.parameters())
    assert bundle['parent_identity'] is proof


def test_repaired_mosei_cache_checks_shared_mask_but_own_prediction(monkeypatch):
    torch.set_num_threads(1)
    rate = '0.7'
    parent, mask_root = prepare.parent_path('CMUMOSEI',66), prepare.mask_path('CMUMOSEI',66)
    reader = FrozenFlatReader(Parent())
    x = torch.randn(2,1,1600)
    a = torch.tensor([[[1.,0.,1.]], [[0.,1.,0.]]])
    u = torch.ones(1,2)
    view = dict(incomplete=x, availability=a, qmask=torch.zeros(1,2), umask=u,
                lengths=[2], labels=torch.tensor([[.2,-.4]]), conversation_ids=['c'])
    meta = dict(labels=np.array([.2,-.4], dtype=np.float32), conversation_ids=np.array(['c','c']),
                utterance_indices=np.array([0,1]))
    digest = hashlib.sha256(a.numpy().tobytes()).hexdigest()
    own = dict(identity=dict(mask_reference=dict(hashes={rate:digest})), best_by_rate={rate:dict(
        predictions='own_selected07.npz', prediction_sha256='pred')})
    mask_summary = dict(identity=dict(mask_hashes=dict(test={rate:digest})))
    monkeypatch.setattr(Path, 'read_text', lambda path: json.dumps(own if path.parent == parent else mask_summary))
    loaded = []
    def load_mask(path, **kwargs):
        loaded.append(path)
        return dict(test={rate:[dict(meta,availability=a)]})
    monkeypatch.setattr(prepare.torch, 'load', load_mask)
    monkeypatch.setattr(prepare.h, 'sha', lambda path: 'pred')
    monkeypatch.setattr(prepare.scalar, 'metadata', lambda data: meta)
    expected = reader(x,a,view['qmask'],u,[2]).prediction.transpose(0,1)[u.bool()].numpy()
    saved = dict(meta, prediction=expected, availability=a.transpose(0,1)[u.bool()].numpy())
    predictions_read = []
    def load_prediction(path, **kwargs):
        predictions_read.append(path)
        return nullcontext(saved)
    monkeypatch.setattr(prepare.np, 'load', load_prediction)
    raw = [None]*8
    raw[7] = u
    class Loader(list):
        sampler = SimpleNamespace(set_epoch=lambda epoch: None)
    trainer = SimpleNamespace(_move_batch=lambda raw,device:raw,
        _prepare_view_from_primary_masks=lambda *args:view)
    bundle = dict(parent_identity=dict(rate=rate,parent_path=str(parent),masks_parent_path=str(mask_root)),
                  trainer=trainer,test_loader=Loader([raw]),dimensions=(1,1,1))
    rows = prepare.test_cache(bundle,reader,rate)
    assert loaded == [mask_root/'fixed_masks.pt']
    assert predictions_read == [parent/'own_selected07.npz']
    assert len(rows) == 1 and bundle['parent_identity']['saved_prediction_replay']['classes_exact']
    with pytest.raises(ValueError,match='match'):
        prepare.test_cache(bundle,reader,'0.3')


def test_registry_exception_requires_exact_reviewed_pair_and_disabled_block(monkeypatch):
    pair = prepare.REVIEWED_REGISTRY_PAIR
    cfg = SimpleNamespace(osram_meaningful_block='none')
    monkeypatch.setattr(prepare.h, 'sha', lambda path: pair['current'])
    core = {prepare.REGISTRY_PATH: pair['parent']}
    assert prepare._core_differences(core,cfg) == {prepare.REGISTRY_PATH:pair}
    with pytest.raises(ValueError,match='active meaningful'):
        prepare._core_differences(core,SimpleNamespace(osram_meaningful_block='nested_dim32'))
    with pytest.raises(ValueError,match='registry'):
        prepare._core_differences({prepare.REGISTRY_PATH:'unknown-parent'},cfg)
    monkeypatch.setattr(prepare.h, 'sha', lambda path:'unreviewed-current')
    with pytest.raises(ValueError,match='registry'):
        prepare._core_differences(core,cfg)
    with pytest.raises(ValueError,match='Unreviewed changed core'):
        prepare._core_differences({'gcnet_missing_m3/meaningful_input_new40.py':'old'},cfg)


def test_original_flat_none_never_constructs_meaningful_adapter(monkeypatch):
    from gcnet_missing_m3.meaningful_input import MeaningfulInputAdapter
    from gcnet_missing_m3.model import MissingM3GraphModel
    from tests.test_completion_memory_write import model_kwargs,inputs
    def forbidden(*args,**kwargs):
        raise AssertionError('inactive registry adapter was constructed')
    monkeypatch.setattr(MeaningfulInputAdapter,'__init__',forbidden)
    cfg = model_kwargs()
    cfg.update(completion_path='none',osram_meaningful_block='none')
    model = MissingM3GraphModel(**cfg).eval()
    assert not hasattr(model.osram,'meaningful_block')
    x,a,q,u,lengths = inputs()
    with torch.no_grad():
        prediction = model([x],a,q,u,lengths)[0]
    assert torch.isfinite(prediction).all()
