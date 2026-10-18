import copy
import importlib

import pytest


PATTERNS = ('A', 'T', 'V', 'AT', 'AV', 'TV', 'ATV')


def api():
    try:
        return importlib.import_module('experiments.osram_persistent_patterns_20261018.analysis')
    except ModuleNotFoundError:
        pytest.fail('Persistent pattern aggregation is missing')


def rows(folds=(1, 2)):
    return [
        {'seed': seed, 'fold': fold, 'patterns': {
            pattern: {'metrics': {
                'ACC': (seed - 66) * 10 + (fold - 1) * 100 + index,
                'UA': (seed - 66) * 10 + (fold - 1) * 100 + index,
                'WF1': (seed - 66) * 10 + (fold - 1) * 100 + index,
                'CE': (seed - 66) * 10 + (fold - 1) * 100 + index,
                'N': 1 if fold == 1 else 99, 'recall': [0., 100.], 'support': [1, 2],
            }} for index, pattern in enumerate(PATTERNS)}}
        for seed in (66, 67, 68) for fold in folds
    ]


def test_equal_fold_then_seed_average_and_sample_sd_without_fake_n():
    source = rows()
    before = copy.deepcopy(source)
    result = api().aggregate_runs(source, folds=(1, 2))
    assert source == before
    assert result['per_seed']['66']['A']['ACC'] == 50
    assert result['patterns']['A']['mean']['ACC'] == 60
    assert result['patterns']['A']['sample_std']['ACC'] == 10
    assert result['macro_six']['per_seed']['66']['ACC'] == 52.5
    assert result['macro_six']['mean']['ACC'] == 62.5
    assert result['macro_seven']['mean']['ACC'] == 63
    assert result['no_text']['mean']['ACC'] == 62
    assert result['macro_six']['sample_std']['ACC'] == 10
    assert set(result['per_seed']['66']['A']) == {'ACC', 'UA', 'WF1', 'CE'}


@pytest.mark.parametrize('change,match', [
    ('missing_run', 'Missing'), ('duplicate_run', 'Duplicate'),
    ('extra_run', 'Unexpected'), ('missing_pattern', 'pattern'),
    ('missing_metric', 'ACC'), ('nonfinite', 'finite'),
    ('nonfinite_recall', 'finite'),
])
def test_incomplete_or_invalid_inputs_are_rejected(change, match):
    source = rows()
    if change == 'missing_run':
        source.pop()
    elif change == 'duplicate_run':
        source.append(copy.deepcopy(source[0]))
    elif change == 'extra_run':
        source.append(dict(source[0], seed=69))
    elif change == 'missing_pattern':
        del source[0]['patterns']['ATV']
    elif change == 'missing_metric':
        del source[0]['patterns']['A']['metrics']['ACC']
    elif change == 'nonfinite':
        source[0]['patterns']['A']['metrics']['ACC'] = float('nan')
    else:
        source[0]['patterns']['A']['metrics']['recall'][0] = float('inf')
    with pytest.raises(ValueError, match=match):
        api().aggregate_runs(source, folds=(1, 2))


def test_single_fold_and_compact_report_include_protocol_and_all_patterns():
    aggregate = api().aggregate_runs(rows((1,)), folds=(1,))
    report = api().render_report({'MELD': {'aggregation': aggregate}})
    for text in ('INTERNAL DIAGNOSTIC ONLY', 'Flat', 'Nested', 'frozen',
                 'T = Text', '0.7', '0.3', '0.0', 'BEST', '66', '67', '68',
                 'five folds', 'ACC', 'UA', 'WF1', 'macro6', 'full7'):
        assert text in report
    for pattern in PATTERNS:
        assert f'| {pattern} |' in report
    assert '10.000 ± 10.000' in report
