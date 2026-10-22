"""CPU-only checks for the finite MOSI memory queue and paired reporting."""
from argparse import Namespace
from copy import deepcopy
import importlib
import json
from pathlib import Path

import pytest


PACKAGE = 'experiments.osram_mosi_memory_grid_20261022'
MODES = ('baseline', 'step_0p2', 'step_0p4', 'step_0p8', 'step_1p0',
         'head_32', 'head_48', 'head_96', 'head_128')
RATES = tuple(f'{n / 10:.1f}' for n in range(8))


def module(name):
    path = Path(__file__).resolve().parents[1] / PACKAGE.replace('.', '/') / (name + '.py')
    assert path.exists(), f'Missing memory-grid {name} implementation'
    return importlib.import_module(PACKAGE + '.' + name)


def summary(score=70, status='complete', epoch=150):
    return dict(status=status, epoch=epoch, seconds=20,
        identity=dict(source_commit='abc', source_hashes={'worker.py': 'sha'},
            mask_hashes={'test': 'exact'}, split={'train': ['a'], 'test': ['b']},
            seed_protocol={'seed': 66}, data_files={'features': 'sha'},
            environment={'python': '3.11'}, config={'osram_write_step': .6},
            mode='baseline', initial_model_sha256='varies', initialization_proof={'mode': 'baseline'}),
        parameters=dict(total=100, flat_adapter=20, memory_state_elements_per_conversation=512),
        best_by_rate={r: {'epoch': i + 1} for i, r in enumerate(RATES)},
        evaluation={r: {'router': {'ACC': score + i, 'WF1': score - i, 'MSE': i / 10 + .1}}
                    for i, r in enumerate(RATES)})


def write_summary(root, mode, value):
    folder = root / 'runs' / mode
    folder.mkdir(parents=True)
    (folder / 'SUMMARY.json').write_text(json.dumps(value))


def test_worker_commands_cover_nine_independent_seed66_arms():
    run = module('run')
    args = Namespace(output=Path('/test'), source_commit='abc')
    assert tuple(run.MODES) == MODES
    for mode in MODES:
        for phase, folder in (('train', 'runs'), ('smoke', 'smoke')):
            cmd = run.worker_command(args, mode, phase, 'GPU-seven')
            assert cmd[cmd.index('-m') + 1] == PACKAGE + '.worker'
            for flag, value in (('--mode', mode), ('--phase', phase), ('--seed', '66'),
                                ('--gpu-uuid', 'GPU-seven'), ('--source-commit', 'abc'),
                                ('--output', f'/test/{folder}/{mode}')):
                assert cmd[cmd.index(flag) + 1] == value
    for mode, phase in (('extra_hpo', 'train'), ('baseline', 'evaluate')):
        with pytest.raises(ValueError):
            run.worker_command(args, mode, phase, 'GPU-seven')


def test_queue_reuses_measured_admission_with_five_process_cap(monkeypatch):
    run = module('run')
    recorded = {}
    def queue_main(argv, **kwargs):
        recorded.update(kwargs)
        return 'delegated'
    monkeypatch.setattr(run.queue, 'main', queue_main)
    assert run.main(['--unused']) == 'delegated'
    assert recorded['modes'] is run.MODES
    assert recorded['command_factory'] is run.worker_command
    assert recorded['report_factory'] is run.summarize
    assert recorded['default_per_gpu'] == recorded['max_per_gpu'] == 5
    args = Namespace(gpu_uuids=['GPU-seven', 'GPU-zero'])
    monkeypatch.setattr(run.queue, 'gpu_free', lambda uuid: (7 if uuid == 'GPU-seven' else 0, 40000))
    assigned = [recorded['assignment_factory'](args, mode, i) for i, mode in enumerate(MODES)]
    assert assigned.count('GPU-seven') == 5
    assert assigned.count('GPU-zero') == 4


@pytest.mark.parametrize('physical', [4, 1, 2])
def test_gpu_assignment_rejects_unapproved_physical_devices(monkeypatch, physical):
    run = module('run')
    monkeypatch.setattr(run.queue, 'gpu_free', lambda uuid: (physical, 40000))
    with pytest.raises(ValueError, match='GPU7|GPU0|GPU 7|GPU 0'):
        run.assignment(Namespace(gpu_uuids=['GPU-wrong', 'GPU-zero']), 'baseline', 0)


def test_running_and_missing_arms_have_progress_without_ranked_scores(tmp_path):
    report = module('report')
    write_summary(tmp_path, 'step_0p2', summary(status='running', epoch=72))
    result = report.summarize(tmp_path)
    assert result['label'] == 'INTERNAL TEST-ORACLE SCREEN'
    assert result['states']['step_0p2']['epoch'] == 72
    assert result['states']['baseline']['status'] == 'not_started'
    assert result['aggregate'] == result['per_rate'] == []
    assert all(not family['aggregate'] for family in result['families'].values())


def test_two_families_share_current_baseline_and_compute_paired_deltas(tmp_path):
    report = module('report')
    write_summary(tmp_path, 'baseline', summary())
    for mode in MODES[1:]:
        current = summary(72)
        current['identity'].update(mode=mode, config={'variant': mode},
                                   initial_model_sha256=mode, initialization_proof={'mode': mode})
        current['parameters']['total'] = 101
        write_summary(tmp_path, mode, current)
    result = report.summarize(tmp_path)
    assert len(result['aggregate']) == 18
    assert len(result['per_rate']) == 72
    assert len(result['families']) == 2
    for family in result['families'].values():
        assert len(family['modes']) == 5 and family['modes'].count('baseline') == 1
        assert len(family['aggregate']) == 10 and len(family['per_rate']) == 40
        assert [row['model'] for row in family['aggregate'] if row['scope'] == 'mean'] == family['modes']
    mean = next(row for row in result['aggregate'] if row['model'] == 'step_0p2' and row['scope'] == 'mean')
    high = next(row for row in result['aggregate'] if row['model'] == 'step_0p2' and row['scope'] == 'high')
    assert mean['ACC'] == 75.5 and mean['WF1'] == 68.5
    assert high['ACC'] == 78 and high['WF1'] == 66
    assert mean['delta'] == {'ACC': 2, 'WF1': 2, 'MSE': 0}
    assert result['per_rate'][7]['epoch'] == 8
    assert result['states']['head_128']['parameters']['total'] == 101
    assert 'INTERNAL TEST-ORACLE' in (tmp_path / 'RESULT.md').read_text()
    assert (tmp_path / 'PER_RATE.csv').exists()


def test_unfinished_baseline_means_null_deltas(tmp_path):
    report = module('report')
    write_summary(tmp_path, 'baseline', summary(status='running', epoch=149))
    write_summary(tmp_path, 'head_32', summary(72))
    result = report.summarize(tmp_path)
    assert len(result['aggregate']) == 2
    assert all(row['delta'] is None for row in result['aggregate'] + result['per_rate'])


@pytest.mark.parametrize('change', ['epoch', 'evaluation', 'best_by_rate', 'nan', 'missing_metric'])
def test_incomplete_or_invalid_completed_runs_are_rejected(tmp_path, change):
    report = module('report')
    current = summary()
    if change == 'epoch':
        current['epoch'] = 149
    elif change in ('evaluation', 'best_by_rate'):
        current[change].pop('0.7')
    elif change == 'nan':
        current['evaluation']['0.7']['router']['WF1'] = float('nan')
    else:
        current['evaluation']['0.7']['router'].pop('MSE')
    write_summary(tmp_path, 'baseline', current)
    with pytest.raises(ValueError, match='baseline'):
        report.summarize(tmp_path)


@pytest.mark.parametrize('key', ['source_commit', 'source_hashes', 'mask_hashes', 'split',
                                'seed_protocol', 'data_files', 'environment'])
def test_paired_reports_reject_protocol_or_provenance_mismatch(tmp_path, key):
    report = module('report')
    first = summary()
    write_summary(tmp_path, 'baseline', first)
    second = deepcopy(first)
    second['identity'][key] = 'different'
    write_summary(tmp_path, 'step_0p4', second)
    with pytest.raises(ValueError, match=key):
        report.summarize(tmp_path)


def test_missing_required_identity_is_rejected_even_for_one_completed_arm(tmp_path):
    report = module('report')
    current = summary()
    current['identity'].pop('mask_hashes')
    write_summary(tmp_path, 'head_48', current)
    with pytest.raises(ValueError, match='mask_hashes'):
        report.summarize(tmp_path)
