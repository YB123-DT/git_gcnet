import importlib
import csv

import pytest


def api():
    return importlib.import_module('experiments.osram_gap_increment_audit_20261003.head_analyze')


def row(i, y=1, full=1, masked=-1, rate=.1, head=0, kind='base', pattern='AV'):
    return dict(seed=66, rate=rate, utterance_id=f'u{i}', label=y, availability=pattern,
                pred_full=full, pred_masked=masked, head=head, intervention=kind)


def test_contribution_direction_neutral_filter_and_changes():
    a = api()
    rows = a.prepare_rows([row(0), row(1, y=-1, full=-1, masked=0), row(2, y=0)])
    score = a.score_rows(rows)
    assert score['full_wf1'] == 100
    assert score['contribution'] > 0
    assert score['mask_harms'] == 1 and score['mask_corrections'] == 0
    assert score['n'] == 2


def test_mismatch_full_prediction_and_duplicate_rejected():
    a = api()
    with pytest.raises(ValueError, match='Full reference'):
        a.prepare_rows([row(0), row(0, head=1, full=-1)])
    with pytest.raises(ValueError, match='Duplicate'):
        a.prepare_rows([row(0), row(0)])


def test_macro_equal_rates_and_separate_base_gap_patterns():
    a = api()
    rows = [row(0), row(1, y=-1, full=-1, masked=-1),
            row(0, rate=.2, masked=1), row(1, rate=.2, y=-1, full=-1, masked=-1),
            row(0, kind='gap', masked=1), row(1, kind='gap', y=-1, full=-1, masked=-1)]
    result = a.analyze(a.prepare_rows(rows))
    base = [r for r in result['matrix'] if r['intervention'] == 'base'][0]
    gap = [r for r in result['matrix'] if r['intervention'] == 'gap'][0]
    assert base['AV'] > 0 and base['ALL'] == base['AV']
    assert gap['AV'] == 0
    assert base['ATV'] is None


def test_gap_atv_sanity_rejects_changed_prediction():
    a = api()
    with pytest.raises(ValueError, match='ATV'):
        a.prepare_rows([row(0, kind='gap', pattern='ATV')])


def test_completeness_missing_head_rejected():
    a = api()
    with pytest.raises(ValueError, match='16'):
        a.validate_complete(a.prepare_rows([row(0)]))


def test_full_cli_all_heads_all_rates(tmp_path, monkeypatch):
    a = api()
    rows = [row(i, y=(1 if i == 0 else -1), full=(1 if i == 0 else -1),
                masked=(1 if i == 0 else -1), rate=r / 10, head=h, kind=k)
            for r in range(8) for h in range(8) for k in ('base', 'gap') for i in range(2)]
    source = tmp_path / 'heads.csv'
    with source.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    before = source.read_bytes()
    output = tmp_path / 'analysis'
    monkeypatch.setattr('sys.argv', ['head_analyze.py', '--input', str(source), '--output', str(output)])
    a.main()
    assert source.read_bytes() == before
    with (output / 'matrix.csv').open() as stream:
        matrix = list(csv.DictReader(stream))
    assert len(matrix) == 32
    assert all(float(r['ALL']) == 0 for r in matrix)
    assert 'INTERNAL DIAGNOSTIC ONLY' in (output / 'RESULT.md').read_text()
