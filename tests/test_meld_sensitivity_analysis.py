"""Synthetic artifact contracts for the frozen sensitivity postprocessor."""
import gzip
import importlib
import json

import numpy as np
import pytest


def analyzer():
    try:
        return importlib.import_module('experiments.osram_meld_memory_sensitivity_20261017.analyze')
    except ModuleNotFoundError:
        pytest.fail('sensitivity analysis has not been implemented')


def write_rows(path, rows):
    with gzip.open(path, 'wt') as stream:
        for row in rows:
            stream.write(json.dumps(row) + '\n')


def fixture(tmp_path, status='complete'):
    metric = dict(N=2, accuracy=.5, unweighted_accuracy=1/7, weighted_f1=.5,
                  cross_entropy=1., corrections=0, harms=0, prediction_changes=0,
                  class_recall=[1., 0., 0., 0., 0., 0., 0.], class_support=[1, 1, 0, 0, 0, 0, 0])
    scores = [dict(condition='baseline', grouping='all', group='all', **metric),
              dict(condition='remove_base_head_0', grouping='all', group='all',
                   **{**metric, 'accuracy': 0., 'unweighted_accuracy': 0.,
                      'class_recall': [0.] * 7, 'harms': 1, 'cross_entropy': 2.})]
    (tmp_path / 'SUMMARY.json').write_text(json.dumps(dict(status=status, rates={'0.4':
        dict(full_coverage=True, metrics=scores, lagged_metrics=[
            dict(alpha='learned', distance=2, cohort='paired', **metric),
            dict(alpha='0.95', distance=2, cohort='paired', **{**metric, 'cross_entropy': 1.25})])})))
    write_rows(tmp_path / 'miss_0p4_utterances.jsonl.gz', [dict(condition='baseline',
        conversation_id='c', utterance_index=t, prior_valid_utterances=t) for t in (0, 1)])
    write_rows(tmp_path / 'miss_0p4_state_changes.jsonl.gz', [dict(condition='beta_0.05',
        conversation_id='c', utterance_index=t, prestate_relative_change=[v, v],
        poststate_relative_change=[v, v], base_relative_change=[v, v],
        gap_relative_change=[v, v], active_gap=[True, False, False]) for t, v in ((0, 1000.), (1, .5))])
    write_rows(tmp_path / 'miss_0p4_direct_beta.jsonl.gz', [dict(beta=.05,
        relative_write_delta_change=[.1, .3], absolute_poststate_change=[.2, .4])])
    write_rows(tmp_path / 'miss_0p4_spectra.jsonl.gz', [dict(head=0, active_modalities=[0, 1],
        gram_eigenvalues=[.1, 1.], weighted_gram_eigenvalues=[.0005, .002],
        weighted_eigenvalues_below_ridge=1, weighted_min_eigenvalue_over_ridge=.5,
        weighted_max_eigenvalue_over_ridge=2., effective_ridge_diagonal=[.002, .004])])
    return tmp_path


def test_refuses_running_or_failed_outputs_by_default(tmp_path):
    source = fixture(tmp_path, 'running')
    with pytest.raises(ValueError, match='complete'):
        analyzer().analyze(source)
    report = analyzer().analyze(source, allow_incomplete=True)
    assert report['source_status'] == 'running'


def test_paired_analysis_excludes_no_history_and_keeps_all_presets(tmp_path):
    report = analyzer().analyze(fixture(tmp_path))
    assert report['state_changes'][0]['N'] == 1
    assert report['state_changes'][0]['prestate_relative_change']['mean'] == pytest.approx(.5)
    assert report['state_changes'][0]['excluded_first'] == 1
    head = report['head_ablation'][0]
    assert head['full_minus_removed_accuracy'] == pytest.approx(.5)
    assert head['full_minus_removed_unweighted_accuracy'] == pytest.approx(1/7)
    assert head['class_recall_delta_full_minus_removed'] == [1., 0., 0., 0., 0., 0., 0.]
    assert report['lagged_alpha'][1]['delta_ce_vs_learned'] == pytest.approx(.25)
    assert len(report['performance']) == 2


def test_spectral_and_direct_aggregates_preserve_scale(tmp_path):
    report = analyzer().analyze(fixture(tmp_path))
    spectral = report['spectra'][0]
    assert spectral['fraction_eigenvalues_below_ridge'] == pytest.approx(.5)
    assert spectral['minimum_eigenvalue_over_ridge']['median'] == pytest.approx(.5)
    assert spectral['effective_ridge_min'] == pytest.approx(.002)
    direct = report['direct_beta'][0]
    assert direct['relative_write_delta_change']['mean'] == pytest.approx(.2)
    assert direct['absolute_poststate_change']['mean'] == pytest.approx(.3)


def test_effect_correlations_compare_ce_changes_and_report_zero_variance():
    labels = np.arange(24) % 2
    baseline = np.zeros((24, 7))
    removed = np.repeat(baseline[:, None, :], 8, axis=1)
    removed[:, :7, 0] = np.linspace(-2., 2., 24)[:, None]
    report = analyzer().head_effect_profiles(labels, baseline, removed, np.array(['011'] * 24))
    assert report[0]['correlation_matrix'][0][1] == pytest.approx(1.)
    assert report[0]['correlation_matrix'][0][7] is None
    assert report[0]['off_diagonal_mean'] == pytest.approx(1.)
    assert report[1]['availability'] == '011'
    assert report[1]['N'] == 24
