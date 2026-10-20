"""Original full-four-session LOSO training with per-rate TEST ACC selection."""
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def worker():
    from experiments.osram_iemocap6_gcnet_testbest_20261020 import worker
    return worker


def evaluation(acc, ua=35., wf1=57., marker=1.):
    arrays = dict(labels=np.array([1]), availability=np.array([[1., 0., 0.]], dtype=np.float32),
                  log_probs_router=np.array([[marker, -marker]], dtype=np.float32))
    return dict(router=dict(ACC=acc, UA=ua, WF1=wf1, CE=1.),
                mask_sha256=hashlib.sha256(arrays['availability'].tobytes()).hexdigest()), arrays


@pytest.mark.parametrize('fold', (1, 2, 3, 4, 5))
@pytest.mark.parametrize('seed', (66, 67, 68))
def test_full_original_configuration_all_folds_and_seeds(fold, seed):
    cfg = worker().make_config(fold, seed)
    assert cfg.fold == fold and cfg.seed == seed
    assert cfg.checkpoint_selection == 'test-oracle-per-rate'
    assert cfg.checkpoint_selection_metric == 'accuracy' and cfg.evaluate_test
    assert cfg.evaluation_protocol == 'official'
    assert cfg.epochs == 100 and cfg.batch_size == 32
    assert cfg.learning_rate == .001 and cfg.weight_decay == 1e-5
    assert cfg.lr_schedule == 'constant' and cfg.missing_mask_protocol == 'gcnet-random-mask-v1'
    assert cfg.osram_readout_fusion == 'flat' and cfg.osram_output_dim == 1600


def test_invalid_fold_and_seed_rejected():
    with pytest.raises(ValueError):
        worker().make_config(6, 66)
    with pytest.raises(ValueError):
        worker().make_config(5, 99)


def test_original_loaders_are_returned_without_adapter_holdout():
    train, validation, test = [object() for _ in range(5)], [object() for _ in range(5)], [object() for _ in range(5)]
    for fold in range(1, 6):
        actual = worker().select_fold(train, validation, test, fold)
        assert actual == (train[fold - 1], validation[fold - 1], test[fold - 1])
    original = SimpleNamespace(sampler=SimpleNamespace(indices=[0, 1, 2, 3]),
                               dataset=SimpleNamespace(vids=['Ses01F', 'Ses02F', 'Ses03F', 'Ses04F', 'Ses05F']))
    heldout = SimpleNamespace(sampler=SimpleNamespace(indices=[4]))
    split = worker().full_training_split(original, heldout, 5)
    assert split['train_indices'] == split['original_train_indices'] == [0, 1, 2, 3]
    assert split['test_indices'] == [4]
    assert split['no_extra_training_holdout'] is True
    assert split['validation_indices'] == [4]
    assert split['validation_is_test_diagnostic_only'] is True
    heldout.sampler.indices = [3, 4]
    with pytest.raises(ValueError, match='overlap'):
        worker().full_training_split(original, heldout, 5)


def test_per_rate_acc_selects_same_call_ua_wf1_and_retains_earliest_tie(tmp_path):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters())
    best, identity = {}, dict(config=dict(checkpoint_selection='test-oracle-per-rate'))
    first = {r: evaluation(60. + i, ua=30. + i, marker=10. + i) for i, r in enumerate(worker().RATES)}
    worker().persist_epoch(tmp_path, model, optimizer, epoch=1, identity=identity,
                           history=[dict(epoch=1)], best=best, evaluated=first)
    second = {r: evaluation(50., ua=90., marker=99.) for r in worker().RATES}
    second['0.0'] = evaluation(65., ua=22., wf1=44., marker=21.)
    second['0.2'] = evaluation(62., ua=99., wf1=99., marker=22.)
    worker().persist_epoch(tmp_path, model, optimizer, epoch=2, identity=identity,
                           history=[dict(epoch=1), dict(epoch=2)], best=best, evaluated=second)
    assert best['0.0']['epoch'] == 2
    assert all(best[r]['epoch'] == 1 for r in worker().RATES[1:])
    assert best['0.0']['metrics']['router'] == dict(ACC=65., UA=22., WF1=44., CE=1.)
    assert best['0.2']['metrics']['router']['UA'] == 32.
    published = worker().publish_selected(tmp_path, best)
    for rate in worker().RATES:
        row = best[rate]
        stored = torch.load(tmp_path / row['checkpoint'], weights_only=False)
        assert stored['selection']['metrics'] == row['metrics']
        assert stored['epoch'] == row['epoch']
        output = tmp_path / 'evaluation' / ('random_' + rate.replace('.', 'p') + '.npz')
        assert output.read_bytes() == (tmp_path / row['predictions']).read_bytes()
        assert published[rate]['selected_epoch'] == row['epoch']


def test_full_last_restores_optimizer_rng_ledger_and_model(tmp_path):
    from experiments.osram_text_path_router_joint_20261019.run import restore
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters())
    model(torch.ones(1, 1)).square().mean().backward()
    optimizer.step()
    best, identity = {}, dict(seed=66)
    worker().persist_epoch(tmp_path, model, optimizer, epoch=1, identity=identity,
                           history=[dict(epoch=1)], best=best,
                           evaluated={r: evaluation(60.) for r in worker().RATES})
    state = torch.load(tmp_path / 'last.pt', weights_only=False)
    expected = torch.rand(4)
    with torch.no_grad():
        model.weight.add_(10.)
    restore(model, optimizer, state, identity)
    assert torch.equal(torch.rand(4), expected)
    assert torch.equal(model.weight, state['model']['weight'])
    assert state['optimizer']['state'] and state['best_by_rate'] == best
    worker().validate_selected(tmp_path, best)
    with (tmp_path / best['0.0']['predictions']).open('ab') as stream:
        stream.write(b'tampered')
    with pytest.raises(ValueError, match='integrity'):
        worker().validate_selected(tmp_path, best)


def test_reject_incomplete_nonfinite_and_mismatched_mask_before_persist(tmp_path):
    model = torch.nn.Linear(1, 1)
    kwargs = dict(output=tmp_path, model=model, optimizer=torch.optim.Adam(model.parameters()),
                  epoch=1, identity={}, history=[], best={})
    with pytest.raises(ValueError, match='eight'):
        worker().persist_epoch(**kwargs, evaluated={'0.0': evaluation(60.)})
    rows = {r: evaluation(60.) for r in worker().RATES}
    rows['0.2'][0]['router']['UA'] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        worker().persist_epoch(**kwargs, evaluated=rows)
    rows = {r: evaluation(60.) for r in worker().RATES}
    rows['0.2'][0]['mask_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='mask'):
        worker().persist_epoch(**kwargs, evaluated=rows)
    assert not (tmp_path / 'last.pt').exists()
