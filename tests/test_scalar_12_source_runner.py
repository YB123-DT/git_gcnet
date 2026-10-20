from pathlib import Path
from types import SimpleNamespace
import importlib

import numpy as np
import pytest
import torch
from torch import nn


def api():
    return importlib.import_module('experiments.osram_scalar_12_source_audit_20261020.run')


def test_scalar_audit_anchor_must_be_completed_original_flat_gcnet():
    summary=dict(status='complete',phase='train',dataset='CMUMOSI',seed=66,mode='flat',epoch=150,
        identity=dict(config=dict(dataset='CMUMOSI',seed=66,epochs=150,
            missing_mask_protocol='gcnet-random-mask-v1',checkpoint_selection='test-oracle-per-rate',
            checkpoint_selection_metric='weighted_f1',mosi_task_mode='regression',
            osram_readout_fusion='flat',osram_output_dim=1600,osram_bidirectional=False,
            training_objective='emotion-only',completion_path='none',osram_tasr=False,
            task_regression_loss='mse',osram_num_heads=8,osram_key_dim=64,osram_value_dim=64,
            latent_dim=256,osram_write_step=.6)))
    assert api().validate_anchor(summary,'CMUMOSI')['epochs']==150
    summary['mode']='joint'
    with pytest.raises(ValueError,match='Flat'):
        api().validate_anchor(summary,'CMUMOSI')
    summary['mode']='flat';summary['identity']['config']['missing_mask_protocol']='conversation-mask-v2'
    with pytest.raises(ValueError,match='GCNet'):
        api().validate_anchor(summary,'CMUMOSI')


def test_only_two_accepted_scalar_dataset_anchors_are_used():
    assert api().reference_root('CMUMOSI').name=='flat'
    assert 'execution_v2' in str(api().reference_root('CMUMOSI'))
    assert 'execution_v1' in str(api().reference_root('CMUMOSEI'))
    with pytest.raises(ValueError): api().reference_root('MELD')


def test_existing_readout_replay_handles_scalar_head_without_classification():
    from test_source_evidence_runner import _fixture
    from experiments.osram_12_source_evidence_audit_20261019.readout import replay_readouts
    osram,_,local,inputs,sources,umask,av=_fixture()
    head=nn.Linear(10,1).eval().requires_grad_(False)
    with torch.no_grad():
        full=head(osram.emotion_norm(osram.local_skip(local)+osram.emotion_adapter(inputs))*umask.T[...,None])
        out=replay_readouts(osram,head,local,inputs,sources,umask,av,full)
    assert out['logits_removed'].shape==(4,2,4,3,1)
    assert torch.equal(out['logits_coalitions'][...,7,:],full)
    assert full.unique().numel()>1


def test_conditions_include_eight_random_and_seven_persistent_not_eight_checkpoints():
    conditions=api().conditions()
    assert len(conditions)==15
    assert [(g,n) for g,n,_ in conditions[:8]]==[('random',f'{i/10:.1f}') for i in range(8)]
    assert {p for g,n,p in conditions[8:]}=={'A','T','V','AT','AV','TV','ATV'}


def test_launcher_is_finite_two_scalar_frozen_jobs_with_original_runtime(tmp_path):
    from experiments.osram_scalar_12_source_audit_20261020.launch import build_jobs
    jobs=build_jobs(tmp_path,'sha','GPU-healthy')
    assert len(jobs)==2
    assert {j['dataset'] for j in jobs}=={'CMUMOSI','CMUMOSEI'}
    assert all('envs/s0/bin/python' in j['command'][0] for j in jobs)
    assert all(j['command'][3]=='experiments.osram_scalar_12_source_audit_20261020.run' for j in jobs)
