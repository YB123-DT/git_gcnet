"""Class-aware IEMOCAP extension; existing MOSEI runner must stay untouched."""
from copy import deepcopy
import hashlib
import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def module():
    return importlib.import_module('experiments.cam_unified_20261030.iemocap')


def base(fold=1):
    m = module()
    return dict(m.required_config(fold), osram_emotion_ablation='full',
                osram_gap_read='residual', gap_residual_strength=1.,
                train_missing_rates=[i / 10 for i in range(8)])


def test_variants_are_only_task_path_changes_and_accept_all_five_folds():
    m = module()
    for fold in range(1, 6):
        raw = base(fold)
        for variant, flags in m.p.VARIANTS.items():
            cfg = m.variant_config(raw, variant, fold)
            assert cfg == dict(raw, **flags)
        assert raw == base(fold)


@pytest.mark.parametrize('key,value', [('dataset', 'IEMOCAPFour'), ('epochs', 150),
    ('learning_rate', 1e-4), ('seed', 67), ('checkpoint_selection_metric', 'weighted_f1'),
    ('missing_mask_protocol', 'conversation-mask-v2'), ('osram_source_role_v3', True)])
def test_wrong_protocol_never_inherits_mosei_settings(key, value):
    raw = base()
    raw[key] = value
    with pytest.raises(ValueError, match=key):
        module().variant_config(raw, 'A3', 1)


def test_initialization_uses_legacy_fold5_but_does_not_relabel_actual_fold(monkeypatch):
    m = module()
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Config:
        fold: int = 2

    seen = []
    monkeypatch.setattr(m.r, 'build_model', lambda cfg, dims, device: seen.append(cfg.fold) or object())
    cfg = Config()
    m.build_model(cfg, (512, 1024, 1024), 'cpu')
    assert seen == [5]
    assert cfg.fold == 2


def test_six_class_ua_is_not_acc_and_includes_absent_class_zero_recall():
    m = module()
    labels = np.asarray([0, 0, 0, 1, 2, 3, 4, 5])
    logits = np.zeros((len(labels), 6), dtype=np.float32)
    logits[:, 0] = 4
    metrics = m.classification_metrics(labels, logits)
    assert metrics['ACC'] == 37.5
    assert metrics['UA'] == pytest.approx(100 / 6)
    assert metrics['recall'] == [100., 0., 0., 0., 0., 0.]
    assert metrics['support'] == [3, 1, 1, 1, 1, 1]
    assert np.isfinite(metrics['CE'])
    assert 'MSE' not in metrics


def test_acc_selection_does_not_choose_better_ua_on_tie():
    m = module()
    old = dict(ACC=60., UA=35., WF1=55., CE=1.)
    new = dict(ACC=60., UA=45., WF1=57., CE=.8)
    assert not m.is_improvement(new, old)
    assert m.is_improvement(dict(new, ACC=60.01), old)


def test_bank_reconstructs_padded_rectangle_without_resampling():
    m = module()
    raw = [None] * 10
    raw[6] = torch.tensor([[0., 1., 0.], [1., 0., 0.]])
    raw[7] = torch.tensor([[1., 1., 0.], [1., 0., 0.]])
    raw[8] = torch.tensor([[0., 2., 0.], [5., 0., 0.]])
    raw[-1] = ['first', 'second']
    meta = m.r.metadata(raw)
    availability = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    selected = {rate: dict(meta, availability=availability) for rate in m.p.RATES}

    class Loader:
        sampler = SimpleNamespace(set_epoch=lambda epoch: None)

        def __iter__(self):
            return iter([raw])

    banks = m.build_banks(Loader(), selected)
    for records in banks.values():
        actual = records[0]['availability']
        np.testing.assert_array_equal(actual.transpose(0, 1)[raw[7].bool()], availability)
        assert not actual.transpose(0, 1)[~raw[7].bool()].any()
    broken = deepcopy(selected)
    broken['0.3']['conversation_ids'] = np.asarray(['wrong', 'first', 'second'])
    with pytest.raises(AssertionError):
        m.build_banks(Loader(), broken)


def test_selected_artifact_paths_cannot_escape_parent_and_hashes_are_checked(tmp_path):
    m = module()
    with pytest.raises(ValueError, match='escape'):
        m.artifact_path(tmp_path, {'path': '../outside.npz', 'sha256': 'f' * 64})
    inside = tmp_path / 'selected.npz'
    inside.write_bytes(b'test')
    with pytest.raises(ValueError, match='hash'):
        m.artifact_path(tmp_path, {'path': 'selected.npz', 'sha256': 'f' * 64})
    assert m.artifact_path(tmp_path, {'path': 'selected.npz', 'sha256': hashlib.sha256(b'test').hexdigest()}) == inside


def test_persisted_acc_winner_binds_six_class_recall_and_same_mask(tmp_path):
    m = module()
    model = torch.nn.Linear(3, 6)
    arrays = dict(labels=np.asarray([0, 1]), prediction=np.asarray([0, 0]),
                  logits=np.asarray([[2., 0, 0, 0, 0, 0], [2., 0, 0, 0, 0, 0]], dtype=np.float32),
                  availability=np.ones((2, 3), dtype=np.float32))
    metrics = m.classification_metrics(arrays['labels'], arrays['logits'])
    metrics['mask_sha256'] = hashlib.sha256(arrays['availability'].tobytes()).hexdigest()
    best = {}
    identity = dict(config={}, dimensions=[3], variant='A3')
    m.persist_selected(tmp_path, model, epoch=3, identity=identity, best=best,
                       evaluated={rate: (dict(metrics), dict(arrays)) for rate in m.p.RATES})
    assert len(best) == 8
    row = best['0.3']
    state = torch.load(tmp_path / row['checkpoint'], map_location='cpu', weights_only=False)
    assert state['kind'] == 'weights-only'
    assert state['selection']['metrics']['UA'] == metrics['UA']
    assert state['selection']['metrics']['support'] == [1, 1, 0, 0, 0, 0]
    before = deepcopy(best)
    m.persist_selected(tmp_path, model, epoch=4, identity=identity, best=best,
                       evaluated={rate: (dict(metrics, UA=metrics['UA'] + 1), dict(arrays))
                                  for rate in m.p.RATES})
    assert best == before
