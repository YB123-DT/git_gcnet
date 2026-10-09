import importlib.util
import csv
import json
from pathlib import Path

import numpy as np
import pytest


PATH = Path(__file__).resolve().parents[1] / 'experiments/osram_probe_direction_20261009/analyze.py'


def module():
    assert PATH.exists(), 'direction analysis implementation is missing'
    spec = importlib.util.spec_from_file_location('direction_analysis', PATH)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def row(**changes):
    result = dict(rate=0., probe_seed=66, conversation_id='a', utterance_index=1,
                  arm='delete', pattern=7, lag=1, y=1., s_true=1.,
                  y_base=1., y_full=9., y_hist=4., y_other=4.,
                  s_base=1., s_full=3., s_hist=2., s_other=2.,
                  delta_r_norm=2**.5, hist_norm=1., other_norm=1.,
                  gradient_norm=1., valid_gradient=True, local_error=0., full_replay_error=0.)
    result.update(changes)
    return result


def test_nonlinear_outputs_are_not_assumed_additive():
    stats = module().summarize_group([row()])
    assert stats['nonadditivity_y_mean'] == pytest.approx(2.)
    assert stats['full_delta_mse_y'] == pytest.approx(64.)
    assert stats['hist_delta_mse_s'] == pytest.approx(1.)


def test_partial_correlation_removes_shared_factor():
    rng = np.random.default_rng(5)
    z = rng.normal(size=4000)
    x, y = 10*z+rng.normal(size=4000), 10*z+rng.normal(size=4000)
    m = module()
    assert m.correlation(x, y) > .98
    assert abs(m.partial_correlation(x, y, z[:, None])) < .06


def test_neutral_targets_excluded_only_from_classification():
    stats = module().summarize_group([row(y=0., y_base=-5., y_full=5.), row(y=1., y_base=-1., y_full=1.)])
    assert stats['base_acc_nonzero'] == 0.
    assert stats['full_acc_nonzero'] == 1.
    assert stats['full_corrections'] == 1
    assert stats['full_harms'] == 0
    assert stats['base_mse_y'] == 14.5


def test_macro_rates_give_equal_weight_to_different_sample_counts():
    values = [dict(rate=0., probe_seed=66, arm='delete', pattern='all', n=100, full_delta_mse_y=1.),
              dict(rate=.1, probe_seed=66, arm='delete', pattern='all', n=1, full_delta_mse_y=9.)]
    result = module().macro_rates(values)
    assert result[0]['full_delta_mse_y'] == 5.
    assert result[0]['rates_n'] == 2


def test_zero_gradient_and_constant_deltas_are_explicit():
    r = row(valid_gradient=False, gradient_norm=0., delta_r_norm=0., hist_norm=0., other_norm=0.)
    r.update({f'{kind}_{part}': 1. for kind in ('y', 's') for part in ('base', 'full', 'hist', 'other')})
    m = module()
    stats = m.summarize_group([r])
    assert stats['negligible_gradient_n'] == 1
    assert stats['full_delta_mse_y'] == 0.
    assert m.correlation([0., 0.], [1., 1.]) is None
    assert m.partial_correlation([0., 0.], [1., 1.], np.zeros((2, 1))) is None


def test_report_keeps_probe_seeds_separate_and_records_partial_coverage(tmp_path):
    rows = [row(probe_seed=seed, conversation_id=str(i), utterance_index=i,
                y_full=1.+i, s_full=1.+i/2.) for seed in (66, 67, 68) for i in range(4)]
    folder = tmp_path/'rate_0p0'
    folder.mkdir()
    with (folder/'rows.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (tmp_path/'STATUS.json').write_text(json.dumps(dict(status='running', rates=[0.], completed=[0.])))
    result = module().analyze(tmp_path, bootstrap=10)
    assert not result['complete_expected_groups']
    assert len(result['macro_rates']) == 6  # all and pattern7 for each seed
    assert result['descriptive_probe_seeds'][0]['complete_three_probe_seeds']
    assert {r['probe_seed'] for r in result['cluster_bootstrap']} == {66}
    assert result['source_status']['status'] == 'running'
    assert len(result['quadrants']) == 12
    assert (tmp_path/'summary/RESULT.md').exists()
    assert json.loads((tmp_path/'summary/SUMMARY.json').read_text())['row_count'] == 12


def test_duplicate_rows_are_rejected(tmp_path):
    folder = tmp_path/'rate_0p0'
    folder.mkdir()
    with (folder/'rows.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row()))
        writer.writeheader()
        writer.writerows([row(), row()])
    with pytest.raises(ValueError, match='Duplicate'):
        module().load_rows(tmp_path)


def test_numerical_decomposition_diagnostics_are_reported():
    stats = module().summarize_group([row(reconstruction_error=2e-4, orthogonality_error=0., probe_parity_error=3e-5)])
    assert stats['reconstruction_failure_n'] == 1
    assert stats['orthogonality_failure_n'] == 0
    assert stats['probe_parity_failure_n'] == 1
