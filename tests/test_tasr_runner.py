"""CPU checks for the frozen-adapter runner, without feature or GPU access."""
import importlib
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn


def helpers():
    return importlib.import_module('experiments.osram_tasr_20261018.runner_helpers')


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 3)
        self.osram = nn.Module()
        self.osram.tasr = nn.Linear(3, 2)
        self.osram.osram_tasr = True

    def forward(self, values, availability, qmask, umask, lengths, predict_missing=False):
        assert predict_missing is False
        base = self.encoder(values[0])
        return self.osram.tasr(base), None, None, None


def test_only_adapter_updates_and_backbone_stays_eval():
    model = TinyModel().train()
    names = helpers().freeze_for_adapter(model)
    assert names == ['osram.tasr.weight', 'osram.tasr.bias']
    assert not model.training
    assert not model.encoder.weight.requires_grad
    before = model.encoder.weight.detach().clone()
    opt = torch.optim.Adam(model.osram.tasr.parameters(), lr=3e-4)
    opt.zero_grad(set_to_none=True)
    model.osram.tasr(model.encoder(torch.ones(2, 3))).sum().backward()
    report = helpers().check_adapter_gradients(model)
    assert report['nonzero_parameters'] > 0
    opt.step()
    assert torch.equal(model.encoder.weight, before)
    model.encoder.weight.grad = torch.ones_like(model.encoder.weight)
    with pytest.raises(RuntimeError): helpers().check_adapter_gradients(model)


def test_isolated_rng_restores_training_streams_even_after_exception():
    torch.manual_seed(9); np.random.seed(10); random.seed(11)
    before = helpers().rng_state()
    with pytest.raises(RuntimeError):
        with helpers().isolated_rng(999):
            torch.rand(10); np.random.rand(10); random.random()
            raise RuntimeError('diagnostic only')
    after = helpers().rng_state()
    assert torch.equal(before['torch'], after['torch'])
    assert np.array_equal(before['numpy'][1], after['numpy'][1])
    assert before['numpy'][2:] == after['numpy'][2:]
    assert before['python'] == after['python']


def test_adapter_checkpoint_restores_optimizer_progress_and_rng(tmp_path):
    model = TinyModel()
    helpers().freeze_for_adapter(model)
    opt = torch.optim.Adam(model.osram.tasr.parameters(), lr=3e-4)
    model.osram.tasr(torch.ones(1, 3)).sum().backward(); opt.step()
    state = helpers().checkpoint_payload(model, opt, epoch=3, identity={'anchor_sha': 'abc'},
        history=[{'epoch': 3}], best_key=(50., -1.))
    helpers().atomic_torch_save(tmp_path/'last.pt', state)
    loaded = torch.load(tmp_path/'last.pt', weights_only=False)
    assert set(loaded['adapter']) == {'weight', 'bias'}
    assert not any('encoder' in k for k in loaded['adapter'])
    expected = model.osram.tasr.weight.detach().clone()
    model.osram.tasr.weight.data.zero_()
    helpers().restore_checkpoint(model, opt, loaded, identity={'anchor_sha': 'abc'})
    assert torch.equal(expected, model.osram.tasr.weight)
    assert loaded['epoch'] == 3 and loaded['history'] == [{'epoch': 3}]
    with pytest.raises(ValueError):
        helpers().restore_checkpoint(model, opt, loaded, identity={'anchor_sha': 'wrong'})


def test_paired_metrics_compute_corrections_harms_on_identical_cohort():
    original = dict(labels=np.array([0, 1, 0, 1]), predictions=np.array([0, 0, 0, 1]))
    new = dict(labels=original['labels'].copy(), predictions=np.array([1, 1, 0, 1]))
    assert helpers().paired_counts(original, new) == {'corrections': 1, 'harms': 1, 'both_correct': 2, 'both_wrong': 0, 'N': 4}
    new['labels'][0] = 1
    with pytest.raises(AssertionError): helpers().paired_counts(original, new)


def test_actual_selection_path_is_checked_not_invented(tmp_path):
    target = tmp_path/'selection_versions'/'actual_name.npz'
    target.parent.mkdir(); np.savez(target, availability=np.ones((2, 3)))
    expected = helpers().sha(target)
    cp = {'selection_artifact': {'path': 'selection_versions/actual_name.npz', 'sha256': expected}}
    assert helpers().resolve_selection(cp, tmp_path, expected) == target
    cp['selection_artifact']['path'] = '../escape.npz'
    with pytest.raises(ValueError): helpers().resolve_selection(cp, tmp_path, expected)


def test_runner_parser_does_not_allow_scope_expansion():
    runner = importlib.import_module('experiments.osram_tasr_20261018.run')
    args = runner.parser().parse_args(['--phase', 'train', '--dataset', 'IEMOCAPSix',
        '--mode', 'real', '--output', '/tmp/new', '--gpu-uuid', 'GPU-example'])
    assert args.resume is False
    with pytest.raises(SystemExit):
        runner.parser().parse_args(['--phase', 'train', '--dataset', 'IEMOCAPFour',
            '--mode', 'real', '--output', '/tmp/new', '--gpu-uuid', 'GPU-example'])


def test_training_uses_raw_full_second_view_and_one_shared_optimizer_step():
    runner = importlib.import_module('experiments.osram_tasr_20261018.run')
    model = TinyModel()
    helpers().freeze_for_adapter(model)
    raw = [torch.ones(2, 64, 3), None, None, None, None, None,
           torch.ones(64, 2), torch.ones(64, 2), torch.zeros(64, 2,dtype=torch.long),
           [f'dialogue-{i}' for i in range(64)]]
    views=[]
    def prepare(data, host, guest, dimensions):
        assert data is raw
        value=dict(incomplete=data[0]*host,availability=host,qmask=data[6],umask=data[7],
                   labels=data[8],lengths=[2]*64)
        views.append(value)
        return value
    def random_view(data, schedule, epoch, dimensions):
        mask=torch.zeros_like(data[0]); mask[...,0]=1
        return prepare(data,mask,mask,dimensions)
    trainer=SimpleNamespace(_move_batch=lambda data,device:data,
        _schedules=lambda cfg,split:{i/10:None for i in range(8)},
        _prepare_view=random_view,_prepare_view_from_primary_masks=prepare,
        _task_loss=lambda dataset,logits,labels,umask:torch.nn.functional.cross_entropy(
            logits.transpose(0,1).reshape(-1,2),labels.reshape(-1)))
    class Loader:
        sampler=SimpleNamespace(set_epoch=lambda epoch:None)
        def __iter__(self): return iter([raw])
    cfg=SimpleNamespace(train_missing_rates=[i/10 for i in range(8)],gradient_clip_norm=1.)
    bundle=dict(model=model,trainer=trainer,train_loader=Loader(),cfg=cfg,
                dimensions=(1,1,1),case={'dataset':'MELD'})
    opt=torch.optim.Adam(model.osram.tasr.parameters(),lr=3e-4)
    before=model.encoder.weight.detach().clone()
    report=runner.train_epoch(bundle,opt,0)
    assert len(views)==2
    assert not views[0]['incomplete'][...,1:].count_nonzero()
    assert views[1]['incomplete'][...,1:].count_nonzero()>0
    assert report['batches']==report['optimizer_steps']==1
    assert report['loss']==pytest.approx(.8*report['random_CE']+.2*report['persistent_CE'])
    assert sum(report['persistent_pattern_conversations'].values())==64
    assert all(int(state['step'])==1 for state in opt.state.values())
    assert torch.equal(before,model.encoder.weight)
