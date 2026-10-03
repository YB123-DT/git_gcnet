import importlib
import csv
import json

import pytest


def api():
    return importlib.import_module('experiments.osram_gap_increment_audit_20261003.analyze')


def row(i, y=1, base=-1, full=1, value=1, rate=0.1):
    return dict(seed=66, rate=rate, artifact_row=i, utterance_id=f'u{i}',
                conversation_id='c', utterance_index=i, label=y,
                pred_local=0, pred_base=base, pred_full=full,
                availability='AV', obs_local_margin=value)


def test_polarity_filter_zero_threshold_and_delta():
    a = api()
    rows = a.prepare_rows([row(0), row(1, base=1, full=0),
                           row(2, y=0), row(3, y=-1, base=0, full=-1)])
    assert [r['polarity_category'] for r in rows] == ['rescue', 'harm', 'neutral', 'both_correct']
    assert rows[0]['delta_gap'] == 4
    score = a.score_rows(rows)
    assert score['n'] == 3
    assert score['rescue'] == score['harm'] == 1
    assert score['base_wf1'] == pytest.approx(score['full_wf1'])


def test_auroc_raw_direction_and_undefined():
    a = api()
    rows = a.prepare_rows([row(0, value=0.1), row(1, base=1, full=-1, value=0.9),
                           row(2, value='nan'), row(3, base=1, full=-1, value='')])
    out = a.observable_rows(rows, ['obs_local_margin'])[0]
    assert out['auc_rescue_positive'] == 0
    assert out['rescue_n'] == out['harm_n'] == 1
    assert out['rescue_missing'] == out['harm_missing'] == 1
    assert out['difference_rescue_minus_harm'] == pytest.approx(-0.8)
    one_class = a.observable_rows(rows[:1], ['obs_local_margin'])[0]
    assert one_class['auc_rescue_positive'] is None


def test_macro_equal_rates_and_missing_auc():
    a = api()
    rows = [dict(seed=66, n=1, full_wf1=0, rescue=1),
            dict(seed=66, n=99, full_wf1=100, rescue=9)]
    out = a.macro(rows, ('full_wf1',), ('n', 'rescue'))
    assert out['full_wf1'] == 50
    assert out['n'] == 100 and out['rescue'] == 10
    assert a.macro([dict(seed=66, auc=None)], ('auc',), ())['auc'] is None


def test_duplicate_and_saved_delta_mismatch_rejected():
    a = api()
    with pytest.raises(ValueError, match='Duplicate'):
        a.prepare_rows([row(0), row(0)])
    bad = row(0)
    bad['delta_gap'] = -4
    with pytest.raises(ValueError, match='delta_gap'):
        a.prepare_rows([bad])


def test_same_availability_strata_and_nonempty_macro(tmp_path):
    a = api()
    rows = [row(0, value=0.1), row(1, base=1, full=-1, value=.9),
            row(0, rate=.2, value=.8), row(1, rate=.2, base=1, full=-1, value=.2)]
    rows[2]['availability'] = rows[3]['availability'] = 'A'
    result = a.analyze(a.prepare_rows(rows))
    assert result['summary']['overall']['n'] == 4
    overall = [r for r in result['observables_macro'] if r['availability'] == 'ALL'][0]
    assert overall['auc_rescue_positive'] == .5
    assert overall['auc_rescue_positive_nonempty_cells'] == 2
    av = [r for r in result['observables_macro'] if r['availability'] == 'AV'][0]
    assert av['auc_rescue_positive'] == 0
    assert av['auc_rescue_positive_nonempty_cells'] == 1


def test_tied_scores_are_chance_not_perfect_separation():
    a = api()
    rows = a.prepare_rows([row(0, value=0), row(1, base=1, full=-1, value=0)])
    assert a.observable_rows(rows, ['obs_local_margin'])[0]['auc_rescue_positive'] == .5


def test_cli_preserves_input_and_writes_complete_outputs(tmp_path, monkeypatch):
    a = api()
    rows = [row(i, rate=rate / 10, base=(-1 if i == 0 else 1),
                full=(1 if i == 0 else -1)) for rate in range(8) for i in range(2)]
    source = tmp_path / 'input.csv'
    with source.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    before = source.read_bytes()
    output = tmp_path / 'analysis'
    monkeypatch.setattr('sys.argv', ['analyze.py', '--input', str(source), '--output', str(output)])
    a.main()
    assert source.read_bytes() == before
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['overall']['n'] == 16
    assert summary['high_missing']['n'] == 6
    assert 'INTERNAL DIAGNOSTIC ONLY' in (output / 'RESULT.md').read_text()
    assert (output / 'observables_per_rate.csv').is_file()
