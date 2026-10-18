"""Finite MOSI150 protocol and recovery tests for recent-history residual."""
from copy import deepcopy
import importlib
import json
from pathlib import Path

import pytest


def runner():
    try:
        return importlib.import_module('experiments.osram_mosi_recent_history_20261018.run')
    except ModuleNotFoundError as exc:
        if exc.name and 'osram_mosi_recent_history_20261018' in exc.name:
            pytest.fail('The finite recent-history runner is not implemented')
        raise


def config():
    return dict(dataset='CMUMOSI', seed=66, epochs=100, learning_rate=.001,
        lr_schedule='constant', batch_size=32, optimizer='adam', weight_decay=1e-5,
        dropout=.5, projector_dropout=.1, latent_dim=256, osram_output_dim=1600,
        osram_num_heads=8, osram_key_dim=64, osram_value_dim=64,
        osram_write_step=.6, osram_bidirectional=False, osram_readout_fusion='flat',
        training_objective='emotion-only', train_rate_mode='cyclic',
        checkpoint_selection='test-oracle-per-rate', emotion_loss_mode='sample-mean',
        task_regression_loss='mse', osram_meaningful_block='none', paired_history_views=False,
        osram_relation_block=False, completion_path='none', osram_ablation='full',
        osram_emotion_ablation='full', evaluate_test=True)


def test_stage_configs_preserve_original_protocol_except_epoch_and_lr():
    source=config(); previous=deepcopy(source)
    a,b=runner().stage_configs(source)
    assert source==previous and a==source
    assert b==dict(source, epochs=150, learning_rate=.0001)


@pytest.mark.parametrize('change', [dict(dataset='MELD'),dict(seed=67),dict(epochs=150),
    dict(lr_schedule='cosine'),dict(batch_size=64),dict(osram_meaningful_block='nested'),
    dict(paired_history_views=True),dict(osram_write_step=.3),dict(completion_path='jepa'),
    dict(training_objective='joint'),dict(osram_relation_block=True),
    dict(missing_mask_protocol='gcnet-random-mask-v1')])
def test_wrong_source_protocol_cannot_silently_enter(change):
    with pytest.raises(ValueError):
        runner().stage_configs(dict(config(),**change))


def state():
    return dict(next_epoch=100,history=[{'epoch':i} for i in range(100)],
        optimizer=dict(state={1:{'step':200}},param_groups=[dict(lr=.001,params=[1])]),
        scheduler=None,scaler=None,rng={'torch':'state'},
        best_references={str(i/10):{} for i in range(8)},
        schedule_identity='original-mask-identity',identity={'config':'original'})


def test_continuation_only_changes_optimizer_lr_and_explicit_run_identity():
    old=state(); saved=deepcopy(old)
    result=runner().continuation_state(old,{'config':'continued'})
    expected=deepcopy(saved)
    expected['identity']={'config':'continued'}
    expected['optimizer']['param_groups'][0]['lr']=.0001
    assert result==expected and old==saved


@pytest.mark.parametrize('change',[dict(next_epoch=99),dict(history=[]),dict(rng={}),
    dict(best_references={}),dict(scheduler={'step':1})])
def test_incomplete_training_state_rejected(change):
    with pytest.raises(ValueError):
        runner().continuation_state(dict(state(),**change),{'config':'continued'})


def test_metrics_use_per_rate_wf1_and_verify_original_masks():
    test={str(i/10):dict(accuracy=.7+i*.01,weighted_f1=.6+i*.01) for i in range(8)}
    metrics=dict(test=test,mask_sha256={str(i/10):str(i) for i in range(8)},selection_metric='weighted_f1')
    result=runner().summarize(metrics,deepcopy(metrics))
    assert result['mean_wf1']==pytest.approx(63.5)
    assert result['high_wf1']==pytest.approx(66.)
    assert result['mean_acc']==pytest.approx(73.5)
    assert result['delta_mean_wf1']==0
    broken=deepcopy(metrics);broken['mask_sha256']['0.7']='different'
    with pytest.raises(ValueError): runner().summarize(broken,metrics)


def test_snapshot_hashes_and_bad_gpu_are_verified(tmp_path):
    run=runner(); source=tmp_path/'source';source.mkdir()
    f=source/'gcnet_missing_m3/model.py';f.parent.mkdir();f.write_text('original')
    (source/'SNAPSHOT.json').write_text(json.dumps(dict(code_commit=run.HISTORICAL_COMMIT,
        source_sha256={'gcnet_missing_m3/model.py':run.sha(f)})))
    assert run.verify_source(source)['commit']==run.HISTORICAL_COMMIT
    f.write_text('changed')
    with pytest.raises(ValueError):run.verify_source(source)
    with pytest.raises(ValueError):run.verify_gpu('bad', [(4,'bad')],visible='bad')
    assert run.verify_gpu('healthy',[(3,'healthy')],visible='healthy')==3
    with pytest.raises(ValueError):run.verify_gpu('healthy',[(3,'healthy')],visible='other')
