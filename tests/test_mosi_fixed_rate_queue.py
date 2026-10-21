import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.osram_mosi_fixed_rate_seed66_20261021.run import (
    RATES, worker_command, capacity, run_directory,
)
from experiments.osram_mosi_fixed_rate_seed66_20261021.report import summarize


def test_exact_eight_unique_fixed_rate_jobs(tmp_path):
    args = SimpleNamespace(output=tmp_path, source_commit='sha', gpu_uuid='GPU-ok')
    commands = [worker_command(args, rate) for rate in RATES]
    assert len(commands) == len({tuple(c) for c in commands}) == 8
    assert commands[0][commands[0].index('--rate') + 1] == '0.0'
    assert commands[-1][commands[-1].index('--rate') + 1] == '0.7'
    assert run_directory(tmp_path, '0.3') == tmp_path/'runs/rate_0p3/seed_66'
    assert all('--layout' not in c and '--phase' in c for c in commands)
    with pytest.raises(ValueError): worker_command(args, '0.8')


def test_capacity_preserves_two_gib_and_four_job_limit():
    assert capacity(32768, 0, 1024) == 4
    assert capacity(32768, 3, 1024) == 1
    assert capacity(6143, 0, 1024) == 0
    assert capacity(6144, 0, 1024) == 1
    assert capacity(32768, 4, 1024) == 0
    assert capacity(16384, 0, 10000) == 1


def test_pending_unmeasured_workers_reserve_their_memory():
    assert capacity(16384, 2, 1024, unmeasured=2) == 1
    assert capacity(10000, 2, 1024, unmeasured=2) == 0


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture_run(root, rate, reference, status='complete'):
    state = dict(status=status, epoch=150 if status == 'complete' else 3,
        fixed_rate=float(rate), config={'epochs': 150}, total_parameters=123,
        identity={'cyclic_reference': {'path': str(reference)}},
        testbest={'epoch': 20, 'metrics': {'ACC': 81., 'WF1': 80., 'MSE': 1.},
                  'checkpoint': 'best.pt', 'checkpoint_sha256': 'weights',
                  'predictions': 'predictions_best.npz', 'prediction_sha256': 'predictions'})
    write(run_directory(root, rate)/'SUMMARY.json', state)


def test_report_only_completed_and_no_fake_eight_model_mean(tmp_path):
    reference = tmp_path/'cyclic'
    write(reference/'SUMMARY.json', {'evaluation': {r: {'router': {'ACC': 80., 'WF1': 79., 'MSE': 1.1}} for r in RATES}})
    fixture_run(tmp_path, '0.0', reference)
    fixture_run(tmp_path, '0.1', reference, status='running')
    result = summarize(tmp_path)
    assert len(result['per_rate']) == 1
    assert result['per_rate'][0]['delta_WF1_pp'] == 1.
    assert result['aggregate'] == []
    assert result['states']['0.1']['status'] == 'running'


def test_eight_model_mean_and_high_missing_are_explicit(tmp_path):
    reference = tmp_path/'cyclic'
    write(reference/'SUMMARY.json', {'evaluation': {r: {'router': {'ACC': 80., 'WF1': 79., 'MSE': 1.1}} for r in RATES}})
    for rate in RATES: fixture_run(tmp_path, rate, reference)
    result = summarize(tmp_path)
    assert len(result['per_rate']) == 8
    assert {row['scope'] for row in result['aggregate']} == {'mean', 'high'}
    assert 'eight independently trained models' in (tmp_path/'RESULT.md').read_text()
