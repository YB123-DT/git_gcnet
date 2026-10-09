import importlib.util
import json
from pathlib import Path

import pytest


PATH = Path(__file__).parents[1] / 'experiments/osram_frozen_residual_20261009/analyze.py'


def module():
    assert PATH.exists(), 'Frozen residual analyzer has not been implemented'
    spec = importlib.util.spec_from_file_location('residual_analysis', PATH)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


GROUPS = ['A_local', 'A_donor', 'A_real', 'B_local_history', 'B_donor_history',
          'B_real_history', 'B_gold_history']


def runs():
    rows = []
    for seed in (66, 67, 68):
        for rate in range(8):
            for group in GROUPS:
                base = dict(weighted_f1_nonzero=(80 - rate)/100, acc_nonzero=(81-rate)/100,
                            mse=1.0, n=686, nonneutral_n=656, corrections=0, harms=0)
                score = dict(base, weighted_f1_nonzero=base['weighted_f1_nonzero'] + (seed-65)/100,
                             corrections=seed-65, harms=1)
                rows.append(dict(seed=seed, rate=rate/10, group=group, epoch=100,
                                 selection='fixed_last_epoch', parameter_count=100,
                                 train=score, test=score, baseline_train=base, baseline_test=base))
    return rows


def test_complete_equal_rate_then_seed_macro():
    a = module()
    result = a.summarize(runs())
    assert result['complete']
    real = next(r for r in result['aggregate'] if r['group'] == 'A_real')
    assert real['mean8_wf1_mean'] == pytest.approx(.785)
    assert real['mean8_wf1_std'] == pytest.approx(.01)
    assert real['high_wf1_mean'] == pytest.approx(.76)
    paired = [r for r in result['comparisons'] if r['model'] == 'A_real' and r['reference'] == 'Original' and r['scope'] == 'mean8']
    assert [r['wf1_delta_pp'] for r in paired] == pytest.approx([1, 2, 3])


def test_partial_has_no_fake_eight_rate_mean():
    result = module().summarize(runs()[:7])
    assert not result['complete']
    assert len(result['missing']) == 161
    assert all(r['mean8_wf1'] is None for r in result['per_seed'])


def test_duplicate_or_baseline_mismatch_rejected():
    a = module()
    rows = runs()
    with pytest.raises(ValueError, match='Duplicate'):
        a.summarize(rows + rows[:1])
    rows[1]['baseline_test'] = dict(rows[1]['baseline_test'], mse=2)
    with pytest.raises(ValueError, match='baseline'):
        a.summarize(rows)


def test_input_merge_and_report(tmp_path):
    a = module()
    rows = runs()
    for i in range(2):
        (tmp_path / f's{i}.json').write_text(json.dumps({'runs': rows[i::2], 'protocol': {'epochs': 100}}))
    loaded, protocols = a.load_inputs([tmp_path/'s0.json', tmp_path/'s1.json'])
    result = a.summarize(loaded)
    a.write_report(result, tmp_path/'report', protocols)
    text = (tmp_path/'report/RESULT.md').read_text()
    assert 'INTERNAL DIAGNOSTIC ONLY' in text
    assert 'residual-head' in text
    assert 'Gold-history' in text
    assert 'per_rate.csv' in text
    assert (tmp_path/'report/comparisons.csv').exists()
