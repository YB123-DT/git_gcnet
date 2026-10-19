import importlib
import json


def test_finite_f2_sweep_only_changes_adapter_learning_rate(tmp_path):
    module = importlib.import_module('experiments.osram_tasr_lr_20261019.launch')
    from experiments.osram_tasr_20261018.protocol import TRAINING_PROTOCOL
    jobs = module.build_jobs(tmp_path)
    assert len(jobs) == 6
    assert len({job['output'] for job in jobs}) == 6
    assert {job['config']['dataset'] for job in jobs} == {'MELD', 'IEMOCAPSix'}
    for job in jobs:
        config = job['config']
        assert config['mode'] == 'real'
        rate = config['training']['learning_rate']
        assert rate in (3e-6, 1e-5, 3e-5)
        assert config['training'] == dict(TRAINING_PROTOCOL, learning_rate=rate)
    assert len({(j['config']['dataset'], j['config']['training']['learning_rate']) for j in jobs}) == 6


def test_summary_never_counts_incomplete_runs_as_results(tmp_path):
    module = importlib.import_module('experiments.osram_tasr_lr_20261019.launch')
    result = module.summarize(tmp_path)
    assert result['complete'] == 0
    assert result['planned'] == 6
    assert result['results'] == []


def test_summary_renders_a_completed_validation_selected_job(tmp_path):
    module = importlib.import_module('experiments.osram_tasr_lr_20261019.launch')
    job = module.build_jobs(tmp_path)[0]
    scores = dict(ACC=60., UA=35., WF1=57., CE=1.)
    summary = dict(status='complete', dataset=job['config']['dataset'], selected_epoch=2,
        backbone_unchanged=True, identity=dict(training=job['config']['training'], mode='real',
            case=dict(checkpoint_sha256=job['config']['anchor_sha256'])),
        evaluation=dict(random=dict(conditions={}, adapter=dict(mean=scores, high=scores)),
                        persistent=dict(conditions={}, six=dict(adapter=scores))))
    path = tmp_path/job['id']; path.mkdir()
    (path/'SUMMARY.json').write_text(json.dumps(summary))
    report = module.summarize(tmp_path)
    assert report['complete'] == 1
    assert '60.0000' in (tmp_path/'RESULT.md').read_text()
