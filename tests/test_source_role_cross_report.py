"""Complete-only, fold-first reporting for the approved cross-dataset screen."""
import copy
import importlib
import json
from pathlib import Path

import pytest


MODULE = 'experiments.osram_source_role_direct_cross_20261020.report'
RATES = tuple(f'{i / 10:.1f}' for i in range(8))


def fixture_state(tmp_path):
    jobs = []
    for dataset, epochs, folds in [('CMUMOSEI', 60, (1,)), ('MELD', 50, (1,)),
                                   ('IEMOCAPSix', 100, range(1, 6))]:
        for seed in (66, 67, 68):
            for fold in folds:
                path = tmp_path / dataset / str(seed) / str(fold)
                path.mkdir(parents=True)
                evaluation = {}
                for index, rate in enumerate(RATES):
                    metric = dict(ACC=40.0 + seed - 66 + fold + index,
                                  WF1=38.0 + seed - 66 + fold + index)
                    metric.update(dict(MSE=0.5) if dataset == 'CMUMOSEI'
                                  else dict(UA=30.0 + seed - 66 + fold + index, CE=1.0))
                    evaluation[rate] = dict(router=metric, selected_epoch=3 + index)
                summary = dict(status='complete', phase='train', epoch=epochs,
                               source_commit='a' * 40,
                               config=dict(epochs=epochs, osram_source_role=True, osram_source_role_mode='direct',
                                           missing_mask_protocol='gcnet-random-mask-v1',
                                           checkpoint_selection='test-oracle-per-rate',
                                           checkpoint_selection_metric=('weighted_f1' if dataset == 'CMUMOSEI'
                                                                        else 'accuracy')),
                               evaluation=evaluation)
                (path / 'SUMMARY.json').write_text(json.dumps(summary))
                jobs.append(dict(dataset=dataset, seed=seed, fold=fold, output=str(path), status='complete'))
    return dict(jobs=jobs)


def test_cross_report_implementation_exists():
    from importlib.util import find_spec
    try:
        found = find_spec(MODULE)
    except ModuleNotFoundError:
        found = None
    assert found is not None, 'Cross-dataset complete-only report is not implemented'


def test_complete_report_fold_then_rate_then_seed_aggregation(tmp_path):
    state = fixture_state(tmp_path)
    output = tmp_path / 'report'
    report = importlib.import_module(MODULE).write_report(output, state)
    saved = json.loads((output / 'REPORT.json').read_text())
    assert report == saved
    assert report['status'] == 'complete'
    iemo = report['datasets']['IEMOCAPSix']
    # Five folds averaged first: seed66 rate0 ACC=(41+42+43+44+45)/5=43.
    assert iemo['per_seed']['66']['by_rate']['0.0']['ACC'] == pytest.approx(43)
    assert iemo['by_rate']['0.0']['ACC'] == pytest.approx(dict(mean=44, std=1))
    assert iemo['mean']['ACC'] == pytest.approx(dict(mean=47.5, std=1))
    assert iemo['high']['ACC'] == pytest.approx(dict(mean=50, std=1))
    assert len(report['rows']) == 168
    assert len((output / 'METRICS.csv').read_text().splitlines()) == 169
    text = (output / 'RESULT.md').read_text()
    assert 'INTERNAL TEST-ORACLE SCREEN' in text
    assert 'not pooled utterance' in text
    assert 'independent' in text
    assert 'MSE' in text and 'UA' in text


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'unfinished', 'failed', 'wrong_epoch',
                                   'wrong_protocol', 'residual', 'wrong_selection', 'nan',
                                   'missing_metric', 'missing_ua', 'mixed_source', 'invalid_selected_epoch',
                                   'wrong_config_epoch', 'wrong_phase', 'disabled', 'missing_rate'])
def test_reject_invalid_or_incomplete_runs_before_writing(tmp_path, change):
    state = fixture_state(tmp_path)
    if change == 'missing':
        state['jobs'].pop()
    elif change == 'duplicate':
        state['jobs'][-1] = copy.deepcopy(state['jobs'][-2])
    elif change == 'unfinished':
        state['jobs'][0]['status'] = 'running'
    elif change == 'failed':
        state['jobs'][0]['status'] = 'failed'
    else:
        job = next(job for job in state['jobs'] if job['dataset'] == 'MELD') if change == 'missing_ua' else state['jobs'][0]
        path = Path(job['output']) / 'SUMMARY.json'
        summary = json.loads(path.read_text())
        if change == 'wrong_epoch': summary['epoch'] = 59
        elif change == 'wrong_protocol': summary['config']['missing_mask_protocol'] = 'conversation-mask-v2'
        elif change == 'residual': summary['config']['osram_source_role_mode'] = 'residual'
        elif change == 'wrong_selection': summary['config']['checkpoint_selection_metric'] = 'accuracy'
        elif change == 'nan': summary['evaluation']['0.0']['router']['ACC'] = float('nan')
        elif change == 'missing_metric': del summary['evaluation']['0.0']['router']['MSE']
        elif change == 'mixed_source': summary['source_commit'] = 'b' * 40
        elif change == 'invalid_selected_epoch': summary['evaluation']['0.0']['selected_epoch'] = 61
        elif change == 'wrong_config_epoch': summary['config']['epochs'] = 59
        elif change == 'wrong_phase': summary['phase'] = 'smoke'
        elif change == 'disabled': summary['config']['osram_source_role'] = False
        elif change == 'missing_rate': del summary['evaluation']['0.7']
        elif change == 'missing_ua': del summary['evaluation']['0.0']['router']['UA']
        path.write_text(json.dumps(summary))
    output = tmp_path / 'report'
    with pytest.raises((ValueError, FileNotFoundError)):
        importlib.import_module(MODULE).write_report(output, state)
    assert not (output / 'RESULT.md').exists()
    assert not (output / 'REPORT.json').exists()


def test_selected_epoch_mapping_is_supported(tmp_path):
    state = fixture_state(tmp_path)
    for job in state['jobs']:
        path = Path(job['output']) / 'SUMMARY.json'
        summary = json.loads(path.read_text())
        summary['selected_epoch_by_rate'] = {
            rate: summary['evaluation'][rate].pop('selected_epoch') for rate in RATES}
        path.write_text(json.dumps(summary))
    report = importlib.import_module(MODULE).write_report(tmp_path / 'report', state)
    assert report['rows'][0]['selected_epoch'] == 3
