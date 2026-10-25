from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from experiments.osram_neumiss_fixed_rate_stage2_20261025 import prepare
from osram_four_block_wiring import FrozenFlatReader
from tests.test_four_block_reader import Parent


@dataclass
class Config:
    seed: int = 66


@pytest.mark.parametrize('rate', prepare.RATES)
def test_setup_selects_and_freezes_matching_parent(monkeypatch, rate):
    cfg = Config()
    environment = dict(torch='torch', numpy='numpy', sklearn='sklearn')
    identity = dict(config=dict(seed=66))
    best = {r: dict(checkpoint='best_'+r+'.pt', checkpoint_sha256='sha:'+r, epoch=10+i)
            for i, r in enumerate(prepare.RATES)}
    accepted = dict(status='complete', epoch=150, identity=identity,
                    environment=environment, best_by_rate=best)
    monkeypatch.setattr(Path, 'read_text', lambda path: json.dumps(accepted))
    monkeypatch.setattr(prepare.h, 'sha', lambda path: 'sha:'+rate if path.suffix == '.pt' else 'summary')
    fresh = nn.Linear(2, 1)
    loaded = []

    def load(path, **kwargs):
        loaded.append(path.name)
        return dict(model=fresh.state_dict(), epoch=best[rate]['epoch'], identity=identity,
                    selection=dict(rate=rate))

    monkeypatch.setattr(prepare.torch, 'load', load)
    monkeypatch.setattr(prepare.control, 'setup', lambda *args: dict(
        model=fresh, cfg=cfg, environment=environment))
    bundle, reader, proof = prepare.setup('GPU-test', Path('/unused-output'), rate)
    assert loaded == ['best_'+rate+'.pt']
    assert proof['rate'] == proof['train_rate'] == rate
    assert proof['test_rates'] == [rate]
    assert proof['neumiss_zero_rate_bypass'] == (rate == '0.0')
    assert reader.parent is fresh and not fresh.training
    assert all(not p.requires_grad for p in fresh.parameters())
    assert bundle['parent_identity'] == proof


def test_fixed_rate_schedule_and_zero_bypass():
    seen = []
    schedules = {float(r): object() for r in prepare.RATES}
    trainer = SimpleNamespace(_schedules=lambda cfg, split: seen.append(split) or schedules)
    bundle = dict(trainer=trainer, cfg=Config())
    for rate in prepare.RATES:
        assert prepare.train_schedule(bundle, rate) is schedules[float(rate)]
        assert prepare.should_train(rate) == (rate != '0.0')
        assert prepare.should_train(rate, 'signature')
    assert seen == ['train']*8
    for invalid in (-.1, .8, True, 'nan', None):
        with pytest.raises(ValueError):
            prepare.validate_rate(invalid)


def test_cache_reads_only_matching_mask_and_prediction(monkeypatch):
    torch.set_num_threads(1)
    rate = '0.6'
    reader = FrozenFlatReader(Parent())
    x = torch.randn(2, 1, 1600)
    a = torch.tensor([[[1., 0., 1.]], [[0., 1., 0.]]])
    u = torch.ones(1, 2)
    view = dict(incomplete=x, availability=a, qmask=torch.zeros(1, 2), umask=u,
                lengths=[2], labels=torch.tensor([[.2, -.4]]), conversation_ids=['c'])
    meta = dict(labels=np.array([.2, -.4], dtype=np.float32),
                conversation_ids=np.array(['c', 'c']), utterance_indices=np.array([0, 1]))
    mask_hash = hashlib.sha256(a.numpy().tobytes()).hexdigest()
    accepted = dict(identity=dict(mask_hashes=dict(test={rate: mask_hash})),
                    best_by_rate={rate: dict(predictions='selected06.npz', prediction_sha256='pred-sha')})
    monkeypatch.setattr(Path, 'read_text', lambda path: json.dumps(accepted))
    monkeypatch.setattr(prepare.torch, 'load', lambda *args, **kwargs: dict(test={rate: [dict(meta, availability=a)]}))
    monkeypatch.setattr(prepare.h, 'sha', lambda path: 'pred-sha')
    monkeypatch.setattr(prepare.scalar, 'metadata', lambda data: meta)
    expected = reader(x, a, view['qmask'], u, [2]).prediction.transpose(0, 1)[u.bool()].numpy()
    saved = dict(meta, prediction=expected, availability=a.transpose(0, 1)[u.bool()].numpy())
    paths = []

    def load_npz(path, **kwargs):
        paths.append(path.name)
        return nullcontext(saved)

    monkeypatch.setattr(prepare.np, 'load', load_npz)
    raw = [None]*8
    raw[7] = u
    loader = [raw]

    class Loader(list):
        sampler = SimpleNamespace(set_epoch=lambda epoch: None)

    trainer = SimpleNamespace(_move_batch=lambda raw, device: raw,
        _prepare_view_from_primary_masks=lambda data, host, guest, dims: view)
    bundle = dict(parent_identity=dict(rate=rate), trainer=trainer,
                  test_loader=Loader(loader), dimensions=(1, 1, 1))
    rows = prepare.test_cache(bundle, reader, rate)
    assert len(rows) == 1 and paths == ['selected06.npz']
    np.testing.assert_array_equal(rows[0]['frame'].prediction.transpose(0, 1)[u.bool()], expected)
    with pytest.raises(ValueError, match='match'):
        prepare.test_cache(bundle, reader, '0.3')


def test_original_flat_config_accepts_each_fixed_rate():
    from dataclasses import replace
    from gcnet_missing_m3.train_gcnet import TrainConfig, _protocol_rates
    cfg = TrainConfig(dataset='CMUMOSI', backbone_type='osram', training_objective='emotion-only',
                      completion_path='none', osram_bidirectional=False, disable_unused_aux_modules=True)
    for rate in prepare.RATES:
        fixed = replace(cfg, train_rate_mode='fixed', fixed_missing_rate=float(rate),
                        train_missing_rates=(float(rate),))
        assert _protocol_rates(fixed) == (float(rate),)


def test_train_epoch_uses_same_requested_schedule_for_every_batch():
    from experiments.osram_neumiss_fixed_rate_stage2_20261025.run import train_epoch
    rate, requested, seen = '0.6', object(), []
    frame = SimpleNamespace(value=torch.ones(2, 1, 1))

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))

        def forward(self, frame):
            return self.weight.expand_as(frame.value)

    class Loader(list):
        sampler = SimpleNamespace(set_epoch=lambda epoch: None)

    def make_view(data, schedule, epoch, dimensions):
        seen.append((schedule, epoch))
        return dict(incomplete=torch.zeros(2, 1, 1), availability=torch.ones(2, 1, 3),
                    qmask=torch.zeros(1, 2), umask=torch.ones(1, 2), lengths=[2],
                    labels=torch.zeros(1, 2), conversation_ids=[str(data)])

    trainer = SimpleNamespace(_schedules=lambda cfg, split: {.6: requested},
        _move_batch=lambda raw, device: raw, _prepare_view=make_view,
        _task_loss=lambda dataset, prediction, labels, umask, *args: prediction.square().mean())
    reader = SimpleNamespace(parent=nn.Linear(1, 1), decode_delta=lambda frame, delta: frame.value + delta)

    class Reader:
        parent = reader.parent
        decode_delta = staticmethod(reader.decode_delta)

        def __call__(self, *args):
            return frame

    cfg = SimpleNamespace(dataset='CMUMOSI', mosi_task_mode='regression', task_regression_loss='mse',
                          task_smooth_l1_beta=1., gradient_clip_norm=5.)
    bundle = dict(trainer=trainer, cfg=cfg, train_loader=Loader([1, 2, 3]), dimensions=(1, 1, 1))
    block = Block()
    row = train_epoch(bundle, Reader(), block, torch.optim.Adam(block.parameters()), 7, rate, False)
    assert seen == [(requested, 7)]*3
    assert row['training_rate'] == rate and row['optimizer_steps'] == 3
