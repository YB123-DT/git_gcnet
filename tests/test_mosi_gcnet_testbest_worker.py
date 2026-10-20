"""Selection-only MOSI reruns retain GCNet masks and exact selected predictions."""
import copy
from dataclasses import asdict
import hashlib
import json

import numpy as np
import pytest
import torch

from tests.test_mosi_gcnet_router_worker import historical_config


def worker():
    from experiments.osram_mosi_gcnet_testbest_20261020 import worker
    return worker


def evaluation(wf1, marker=1., mse=1.):
    arrays = dict(prediction=np.array([[marker]], dtype=np.float32),
                  labels=np.array([1.], dtype=np.float32),
                  availability=np.array([[1, 0, 0]], dtype=np.float32))
    metrics = dict(router=dict(ACC=80., WF1=wf1, MSE=mse),
                   mask_sha256=hashlib.sha256(arrays['availability'].tobytes()).hexdigest())
    return metrics, arrays


@pytest.mark.parametrize('seed', (66, 67, 68))
def test_restored_selection_changes_only_checkpoint_policy(seed):
    from experiments.osram_mosi_gcnet_router_3seed_20261019 import worker as old
    _, prior = old.configs(historical_config(), seed)
    current = worker().testbest_config(prior)
    changed = {k for k in asdict(prior) if asdict(prior)[k] != asdict(current)[k]}
    assert changed == {'checkpoint_selection'}
    assert current.checkpoint_selection == 'test-oracle-per-rate'
    assert current.checkpoint_selection_metric == 'weighted_f1'
    assert current.missing_mask_protocol == 'gcnet-random-mask-v1'
    assert current.evaluate_test
    assert current.epochs == 150


def test_per_rate_best_is_strict_wf1_not_global_or_mse_and_arrays_are_saved(tmp_path):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    identity = dict(policy='test-oracle-per-rate')
    best = {}
    first = {rate: evaluation(70. + i, marker=10. + i) for i, rate in enumerate(worker().RATES)}
    worker().persist_epoch(tmp_path, model, optimizer, epoch=1, identity=identity,
                           history=[dict(epoch=1)], best=best, evaluated=first)
    # Rate0 improves while rate.1 gets worse; rate.2 ties WF1 but has better MSE.
    second = {rate: evaluation(60., marker=99.) for rate in worker().RATES}
    second['0.0'] = evaluation(75., marker=21.)
    second['0.2'] = evaluation(72., marker=22., mse=.001)
    worker().persist_epoch(tmp_path, model, optimizer, epoch=2, identity=identity,
                           history=[dict(epoch=1), dict(epoch=2)], best=best, evaluated=second)
    assert best['0.0']['epoch'] == 2
    assert all(best[rate]['epoch'] == 1 for rate in worker().RATES[1:])
    assert best['0.2']['metrics']['router']['MSE'] == 1.
    for rate in worker().RATES:
        record = best[rate]
        checkpoint = torch.load(tmp_path / record['checkpoint'], weights_only=False)
        assert checkpoint['epoch'] == record['epoch']
        assert checkpoint['selection']['metrics'] == record['metrics']
        assert checkpoint['selection']['prediction_sha256'] == record['prediction_sha256']
        with np.load(tmp_path / record['predictions'], allow_pickle=False) as saved:
            expected = second[rate][1] if rate == '0.0' else first[rate][1]
            for key in expected:
                np.testing.assert_array_equal(saved[key], expected[key])
    # No evaluation callback is accepted: final outputs must copy selected bytes.
    final = worker().publish_selected(tmp_path, best)
    assert set(final) == set(worker().RATES)
    for rate in worker().RATES:
        artifact = tmp_path / 'evaluation' / ('random_' + rate.replace('.', 'p') + '.npz')
        assert artifact.read_bytes() == (tmp_path / best[rate]['predictions']).read_bytes()
        assert final[rate]['output_sha256'] == best[rate]['prediction_sha256']


def test_selected_checkpoint_and_last_restore_model_adam_rng_and_best_ledger(tmp_path):
    from experiments.osram_text_path_router_joint_20261019.run import restore
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    model(torch.ones(1, 1)).square().mean().backward()
    optimizer.step()
    best, identity = {}, dict(seed=66)
    worker().persist_epoch(tmp_path, model, optimizer, epoch=1, identity=identity,
                           history=[dict(epoch=1)], best=best,
                           evaluated={r: evaluation(75.) for r in worker().RATES})
    state = torch.load(tmp_path / 'last.pt', weights_only=False)
    expected = torch.rand(4)
    with torch.no_grad():
        model.weight.add_(10.)
    restore(model, optimizer, state, identity)
    assert torch.equal(torch.rand(4), expected)
    assert torch.equal(model.weight, state['model']['weight'])
    assert state['best_by_rate'] == best
    assert state['optimizer']['state']
    assert state['history'] == [dict(epoch=1)]
    worker().validate_selected(tmp_path, state['best_by_rate'])


def test_selected_integrity_detects_prediction_tampering(tmp_path):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters())
    best = {}
    worker().persist_epoch(tmp_path, model, optimizer, epoch=1, identity={}, history=[], best=best,
                           evaluated={r: evaluation(75.) for r in worker().RATES})
    with (tmp_path / best['0.0']['predictions']).open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='prediction'):
        worker().validate_selected(tmp_path, best)


def test_training_parity_requires_exact_masks_and_small_explicit_numeric_tolerance():
    reference = dict(epoch=1, mask_trace_sha256='mask', task_MSE=1.2,
                     optimizer_steps=3, learning_rates=[.001])
    same = dict(reference)
    assert worker().check_training_parity(same, reference)['mask_trace_exact']
    slight = dict(reference, task_MSE=1.2000001)
    proof = worker().check_training_parity(slight, reference)
    assert proof['task_MSE_close'] and proof['task_MSE_abs_error'] > 0
    with pytest.raises(ValueError, match='mask'):
        worker().check_training_parity(dict(reference, mask_trace_sha256='other'), reference)
    with pytest.raises(ValueError, match='MSE'):
        worker().check_training_parity(dict(reference, task_MSE=1.3), reference)


def test_selection_reference_rejects_changed_training_config_or_initialization(tmp_path):
    cfg = dict(checkpoint_selection='test-oracle-per-rate', checkpoint_selection_metric='weighted_f1',
               missing_mask_protocol='gcnet-random-mask-v1', learning_rate=.001)
    identity = dict(config=cfg, source_hashes={'core.py': 'sha'}, split={'train': 'same'},
                    mask_hashes={'test': 'test', 'validation': 'val'}, seed_protocol={'seed': 66})
    old = dict(status='complete', epoch=150, initial_backbone_sha256='initial',
               identity=dict(copy.deepcopy(identity), config=dict(cfg, checkpoint_selection='validation')))
    (tmp_path / 'SUMMARY.json').write_text(json.dumps(old))
    history = [dict(epoch=i+1) for i in range(150)]
    (tmp_path / 'history.json').write_text(json.dumps(history))
    loaded, proof = worker().selection_reference(tmp_path, identity, 'initial')
    assert loaded == history and proof['only_checkpoint_selection_changed']
    with pytest.raises(ValueError, match='initialization'):
        worker().selection_reference(tmp_path, identity, 'different')
    changed = copy.deepcopy(identity)
    changed['config']['learning_rate'] = .002
    with pytest.raises(ValueError, match='config'):
        worker().selection_reference(tmp_path, changed, 'initial')


def test_incomplete_or_nonfinite_epoch_is_not_persisted(tmp_path):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters())
    kwargs = dict(output=tmp_path, model=model, optimizer=optimizer, epoch=1,
                  identity={}, history=[], best={})
    with pytest.raises(ValueError, match='eight'):
        worker().persist_epoch(**kwargs, evaluated={'0.0': evaluation(75.)})
    rows = {r: evaluation(75.) for r in worker().RATES}
    rows['0.1'] = evaluation(float('nan'))
    with pytest.raises(ValueError, match='finite'):
        worker().persist_epoch(**kwargs, evaluated=rows)
    assert not (tmp_path / 'last.pt').exists()


def test_smoke_reaches_missing_batches_not_only_all_observed_batch_zero():
    from gcnet_missing_m3.mixed_rate import BalancedBatchRateSchedule
    limit = worker().SMOKE_BATCHES
    rates = [BalancedBatchRateSchedule().rate_for(0, i) for i in range(limit)]
    assert limit == 3
    assert rates[0] == 0. and any(rate > 0 for rate in rates)
