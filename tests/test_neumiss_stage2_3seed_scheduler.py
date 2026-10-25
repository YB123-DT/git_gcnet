import json

import pytest


def test_queue_adds_only_missing_neumiss_seeds():
    from experiments.osram_neumiss_stage2_3seed_20261025.scheduler import jobs_for
    jobs = jobs_for('train')
    assert len(jobs) == 16
    assert {(j['seed'], j['rate']) for j in jobs} == {
        (s, f'{r/10:.1f}') for s in (67, 68) for r in range(8)}
    assert all(j['mode'] == 'neumiss' for j in jobs)
    assert [(j['seed'], j['rate']) for j in jobs_for('smoke')] == [(67, '0.7')]


def test_gpu_map_has_no_accidental_gpu3_or7():
    from experiments.osram_neumiss_stage2_3seed_20261025.scheduler import parse_gpu_map
    assert parse_gpu_map('0=GPU-a,2=GPU-b') == {0:'GPU-a',2:'GPU-b'}
    for text in ('3=GPU-a', '4=GPU-a', '7=GPU-a', '0=GPU-a,1=GPU-a', '0=GPU-a,0=GPU-b'):
        with pytest.raises(ValueError):
            parse_gpu_map(text)


def test_admission_reserves_current_peers_and_never_ignores_memory():
    from experiments.osram_neumiss_stage2_3seed_20261025.scheduler import can_admit
    snapshot = {'ram_available_gib':40, 'disk_free_gib':20,
                'gpus':{'GPU-a':{'free_mib':10000}}}
    assert can_admit(snapshot, 'GPU-a', 1, 1000, 4)
    assert not can_admit(snapshot, 'GPU-a', 4, 1000, 4)
    assert not can_admit(snapshot, 'GPU-a', 1, 4000, 4)
    assert not can_admit(dict(snapshot, ram_available_gib=10), 'GPU-a', 0, 1000, 4)


def test_aggregates_use_seed_means_and_sample_sd():
    from experiments.osram_neumiss_stage2_3seed_20261025.summarize import aggregate_rows
    rows = [dict(seed=s, rate=f'{r/10:.1f}', ACC=float(s), WF1=float(s), MSE=1.,
                 parent_ACC=66., parent_WF1=66., parent_MSE=1.,
                 delta_ACC=float(s-66), delta_WF1=float(s-66), delta_MSE=0.)
            for s in (66,67,68) for r in range(8)]
    seeds, groups = aggregate_rows(rows)
    assert len(seeds) == 6
    average = next(g for g in groups if g['rate'] == 'mean')
    assert average['metrics']['WF1'] == {'mean':67., 'sample_sd':1.}
    with pytest.raises(ValueError, match='24'):
        aggregate_rows(rows[:-1])


def test_epoch_zero_requires_noop_and_baseline_equality():
    from experiments.osram_neumiss_stage2_3seed_20261025.summarize import check_epoch
    summary = dict(status='complete', epoch=0, no_op=True,
        baseline={'ACC':70., 'WF1':70., 'MSE':1.},
        best=dict(epoch=0, no_op=True, metrics={'ACC':70., 'WF1':70., 'MSE':1.}))
    check_epoch(summary, '0.0', [])
    summary['best']['metrics']['WF1'] = 71.
    with pytest.raises(ValueError, match='no-op'):
        check_epoch(summary, '0.0', [])


def test_normal_rate_requires_contiguous_150_and_fixed_rate_history():
    from experiments.osram_neumiss_stage2_3seed_20261025.summarize import check_epoch
    history = [dict(epoch=e, training_rate='0.3', optimizer_steps=2) for e in range(1,151)]
    summary = dict(status='complete', epoch=150)
    check_epoch(summary, '0.3', history)
    history[30]['training_rate'] = '0.4'
    with pytest.raises(ValueError, match='fixed'):
        check_epoch(summary, '0.3', history)


def test_recovered_pid_must_match_recorded_command(monkeypatch, tmp_path):
    from experiments.osram_neumiss_stage2_3seed_20261025.scheduler import process_alive
    monkeypatch.setattr('experiments.osram_neumiss_stage2_3seed_20261025.scheduler.PROC',tmp_path)
    directory = tmp_path/'123'
    directory.mkdir()
    (directory/'cmdline').write_bytes(b'python\0-m\0wrong_module\0')
    job = dict(pid=123, command=['python','-m','expected_module'])
    with pytest.raises(RuntimeError, match='PID'):
        process_alive(job)
    (directory/'cmdline').write_bytes(b'python\0-m\0expected_module\0')
    assert process_alive(job)
    assert not process_alive(dict(pid=456,command=job['command']))


def test_saved_epoch_zero_predictions_are_independently_verified(tmp_path):
    import hashlib
    import numpy as np
    from experiments.osram_neumiss_stage2_3seed_20261025.summarize import validated_row
    from experiments.osram_neumiss_fixed_rate_stage2_20261025.scheduler import digest
    parent_dir,run_dir = tmp_path/'parent',tmp_path/'run'
    parent_dir.mkdir()
    run_dir.mkdir()
    checkpoint = parent_dir/'best.pt'
    checkpoint.write_bytes(b'frozen parent')
    parent_hash = digest(checkpoint)
    y = np.array([1.,-1.,0.],dtype=np.float32)
    p = np.array([.5,-.5,.1],dtype=np.float32)
    a = np.ones((3,3),dtype=np.bool_)
    for directory in (parent_dir,run_dir):
        np.savez(directory/'prediction.npz',labels=y,prediction=p,base_prediction=p,availability=a)
    metrics = dict(ACC=100.,WF1=100.,MSE=float(np.mean((y-p)**2)))
    (parent_dir/'SUMMARY.json').write_text(json.dumps(dict(best_by_rate={'0.0':dict(
        checkpoint='best.pt',checkpoint_sha256=parent_hash,predictions='prediction.npz')})))
    checks = dict.fromkeys(('parent_unchanged','parent_grad_none','baseline_replay_exact',
                           'only_requested_train_rate','only_requested_test_rate'),True)
    summary = dict(status='complete',epoch=0,no_op=True,parent_id=parent_hash,checks=checks,
        baseline=metrics,best=dict(epoch=0,no_op=True,metrics=metrics,predictions='prediction.npz',
        prediction_sha256=digest(run_dir/'prediction.npz')),
        identity=dict(seed=66,mode='neumiss',training_rate='0.0',evaluation_rate='0.0',
            cyclic_training=False,mask_protocol='gcnet-random-mask-v1',source_commit='fixture',
            test_mask_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
            source_hashes={'osram_four_block_wiring/blocks.py':'same-block'},
            parent=dict(checkpoint=str(checkpoint),checkpoint_sha256=parent_hash,config={'seed':66})))
    (run_dir/'history.json').write_text('[]')
    path = run_dir/'SUMMARY.json'
    path.write_text(json.dumps(summary))
    row,proof = validated_row(path,66,'0.0')
    assert row['delta_WF1'] == 0 and row['selected_epoch'] == 0
    assert proof['independent_prediction_metrics'] is True
    summary['identity']['test_mask_sha256'] = 'wrong'
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError,match='mask digest'):
        validated_row(path,66,'0.0')
