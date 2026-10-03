import importlib
import csv

import pytest


def api():
    return importlib.import_module('experiments.osram_gap_increment_audit_20261003.query_analyze')


def row(i=0, rate=.1, head=0, modality='T', pattern='AV', q=.2, read=.8):
    return dict(seed=66, rate=rate, utterance_id=f'u{i}', availability=pattern,
                modality=modality, head=head, query_base_gap_cos=q,
                query_base_residual_gap_cos=0, query_gap_residual_cos=.3,
                read_base_gap_cos=read, rho=.6, eta=.2)


def test_undefined_and_paired_similarity_change():
    a = api()
    rows = a.prepare_rows([row(), row(1, q='nan', read=.9)])
    result = a.summarize(rows)
    assert result['query_base_gap_cos_n'] == 1
    assert result['query_base_gap_cos_missing'] == 1
    assert result['read_minus_raw_query_cos_mean'] == pytest.approx(.6)
    assert result['read_minus_raw_query_cos_n'] == 1
    assert result['read_minus_actual_query_cos_mean'] == pytest.approx(.85)


def test_invalid_observed_modality_duplicate_and_cosine_rejected():
    a = api()
    with pytest.raises(ValueError, match='active'):
        a.prepare_rows([row(modality='A')])
    with pytest.raises(ValueError, match='Duplicate'):
        a.prepare_rows([row(), row()])
    with pytest.raises(ValueError, match='cosine'):
        a.prepare_rows([row(q=1.5)])


def test_quantiles_and_equal_rate_macro_not_pooled():
    a = api()
    rows = a.prepare_rows([row(q=0, read=0), row(1, q=1, read=1), row(rate=.2, q=.2, read=.4)])
    out = a.analyze(rows)
    chosen = [r for r in out['macro'] if r['head'] == 0 and r['modality'] == 'T'][0]
    assert chosen['query_base_gap_cos_mean'] == pytest.approx(.35)
    assert chosen['query_base_gap_cos_p50'] == pytest.approx(.35)
    assert chosen['query_base_gap_cos_mean_nonempty_cells'] == 2
    assert chosen['query_base_gap_cos_n'] == 3


def test_join_category_same_identity_label_and_availability():
    a = api()
    rows = a.prepare_rows([dict(row(), label=1)])
    audit = [dict(seed=66, rate=.1, utterance_id='u0', availability='AV', label=1, polarity_category='rescue')]
    assert a.join_categories(rows, audit)[0]['polarity_category'] == 'rescue'
    audit[0]['availability'] = 'A'
    with pytest.raises(ValueError, match='availability'):
        a.join_categories(rows, audit)


def test_runner_schema_and_cli(tmp_path, monkeypatch):
    a = api()
    rows = []
    for head in range(8):
        r = row(head=head)
        for target, source in a.ALIASES.items():
            r[target] = r.pop(source)
        rows.append(r)
    source = tmp_path / 'queries.csv'
    with source.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    before = source.read_bytes()
    output = tmp_path / 'analysis'
    monkeypatch.setattr('sys.argv', ['query_analyze.py', '--input', str(source), '--output', str(output)])
    a.main()
    assert source.read_bytes() == before
    with (output / 'macro.csv').open() as stream:
        macro = list(csv.DictReader(stream))
    assert len(macro) == 10
    assert 'BEFORE residual addressing' in (output / 'RESULT.md').read_text()
