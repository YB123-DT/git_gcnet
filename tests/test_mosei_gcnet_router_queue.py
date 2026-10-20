from copy import deepcopy

import pytest


def test_queue_adds_only_seed66_mosei_flat_router_on_one_healthy_card(tmp_path):
    from experiments.osram_mosei_gcnet_router_20261020.run import plan_jobs
    jobs = plan_jobs(tmp_path, 'sha', ['GPU-healthy'])
    assert len(jobs) == 2
    assert {j['mode'] for j in jobs} == {'flat', 'joint'}
    assert all(j['seed'] == 66 and j['dataset'] == 'CMUMOSEI' for j in jobs)
    assert all(j['gpu_uuid'] == 'GPU-healthy' for j in jobs)
    assert all(j['mask_protocol'] == 'gcnet-random-mask-v1' for j in jobs)
    assert all(j['checkpoint_selection'] == 'test-oracle-per-rate' for j in jobs)
    assert all(j['command'][2] == 'experiments.osram_mosei_gcnet_router_20261020.worker' for j in jobs)
    with pytest.raises(ValueError):
        plan_jobs(tmp_path, 'sha', ['GPU-a', 'GPU-b'])


def test_report_accepts_only_complete60epoch_mosei_wf1_policy():
    from experiments.osram_mosei_gcnet_router_20261020.report import validate_policy
    row = dict(dataset='CMUMOSEI', status='complete', phase='train', epoch=60,
        selected_epoch_by_rate={str(i/10): i+1 for i in range(8)},
        identity=dict(config=dict(dataset='CMUMOSEI', epochs=60,
            checkpoint_selection='test-oracle-per-rate', checkpoint_selection_metric='weighted_f1',
            missing_mask_protocol='gcnet-random-mask-v1')))
    assert validate_policy(row) == row['selected_epoch_by_rate']
    for field, value in [('checkpoint_selection', 'validation'),
                         ('checkpoint_selection_metric', 'accuracy'),
                         ('missing_mask_protocol', 'conversation-mask-v2')]:
        bad = deepcopy(row)
        bad['identity']['config'][field] = value
        with pytest.raises(ValueError):
            validate_policy(bad)
    bad = deepcopy(row)
    bad['selected_epoch_by_rate']['0.7'] = 61
    with pytest.raises(ValueError):
        validate_policy(bad)


def test_incomplete_queue_does_not_publish_a_winner(tmp_path):
    from experiments.osram_mosei_gcnet_router_20261020.report import write_report
    with pytest.raises(ValueError, match='two'):
        write_report(tmp_path, dict(jobs=[]))
