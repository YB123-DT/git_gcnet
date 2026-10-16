"""Finite B3 screen, fixed original Flat protocol, and real Optuna state tests."""
import importlib
import json
from pathlib import Path

import pytest


PACKAGE = 'experiments.osram_meld_b3_structure_hpo_20261016.run'
ROOT = Path(__file__).resolve().parents[3]


def module():
    return importlib.import_module(PACKAGE)


def test_b3_only_changes_the_two_structure_parameters():
    m = module()
    base = json.loads(m.BASE.read_text())
    from experiments.osram_meld_block2_auto_20261014.methods import build_config
    expected = build_config(base, 'B3', 'unused')
    actual = m.make_config(8, 3)
    assert actual == dict(expected, osram_mbt_bottlenecks=8, osram_mbt_rounds=3)
    assert (actual['learning_rate'], actual['weight_decay'], actual['dropout']) == (.001, 1e-5, .5)
    assert actual['osram_readout_fusion'] == 'flat'
    assert not actual.get('osram_disable_local_skip', False)
    assert actual['osram_output_dim'] == 1600
    assert actual['missing_mask_protocol'] == 'gcnet-random-mask-v1'
    for args in ((0, 2), (9, 2), (4, 0), (4, 5), (True, 2), (4, 2.0)):
        with pytest.raises(ValueError):
            m.make_config(*args)


def test_exact_five_initial_configurations_and_gpu2_limit():
    m = module()
    assert m.INITIAL_CONFIGS == (
        {'bottlenecks': 4, 'rounds': 2}, {'bottlenecks': 2, 'rounds': 2},
        {'bottlenecks': 8, 'rounds': 2}, {'bottlenecks': 4, 'rounds': 1},
        {'bottlenecks': 4, 'rounds': 3})
    a = m.parse_args(['--gpus', '2', '--per-gpu', '5', '--trials', '20'])
    assert a.gpus == (2,) and a.per_gpu == 5 and a.trials == 20
    for args in (['--gpus', '4'], ['--gpus', '2,3'], ['--per-gpu', '6'], ['--trials', '25']):
        with pytest.raises(SystemExit):
            m.parse_args(args)


def test_full_b3_search_is_unique_initial_barrier_then_adaptive_and_resumable(tmp_path):
    optuna = pytest.importorskip('optuna')
    m = module()

    class CPUResults:
        def __init__(self):
            self.jobs = {}
            self.batches = []

        def run_jobs(self, jobs, return_on_first_completion=False):
            self.batches.append((len(jobs), return_on_first_completion))
            rows = []
            for job in jobs:
                assert job['method'] == 'B3' and job['id'].startswith('b3_struct_')
                self.jobs[job['id']] = job
                config = job['config']
                tokens = config['osram_mbt_bottlenecks']
                rounds = config['osram_mbt_rounds']
                assert type(tokens) is int and type(rounds) is int
                metrics = {'ACC': 60. + tokens / 10, 'UA': 35. + rounds / 10, 'WF1': 56., 'CE': 1.2}
                full = {'mean': metrics, 'high': metrics, 'rates': {str(i / 10): metrics for i in range(8)}}
                rows.append(dict(job, status='complete', summary={
                    'status': 'complete', 'paired': {'on': full, 'mask_digest': 'fixed'},
                    'testbest': full, 'parameters': 123, 'block_parameters': 20,
                    'training_seconds': 1., 'peakmemorybytes': 1}))
            return rows

    results = CPUResults()
    flat = {'on': {'mean': {'ACC': 60., 'UA': 35.}}, 'mask_digest': 'fixed'}
    # Fill an interrupted ask/suggest from SQLite without a duplicated worker.
    study = optuna.create_study(study_name=m.STUDY_NAME,
        storage='sqlite:///' + str(tmp_path / 'STRUCTURE.sqlite3'), direction='maximize',
        sampler=m.unique_tpe_sampler())
    for config in m.INITIAL_CONFIGS:
        study.enqueue_trial(config)
    study.ask().suggest_int('bottlenecks', 1, 8)
    report = m.run_search(tmp_path, flat, results)
    assert report['status'] == 'complete' and len(report['trials']) == 20
    pairs = [(j['config']['osram_mbt_bottlenecks'], j['config']['osram_mbt_rounds'])
             for j in results.jobs.values()]
    assert len(pairs) == len(set(pairs)) == 20
    assert pairs[:5] == [(4, 2), (2, 2), (8, 2), (4, 1), (4, 3)]
    assert results.batches[0] == (5, False)
    assert all(width <= 5 and adaptive for width, adaptive in results.batches[1:])
    count = len(results.batches)
    m.run_search(tmp_path, flat, results)
    assert len(results.batches) == count
    assert report['scope'].startswith('B3')
