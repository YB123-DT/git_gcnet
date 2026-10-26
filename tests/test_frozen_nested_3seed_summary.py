import importlib

MODULE = 'experiments.osram_frozen_nested_3seed_20261026.summarize'


def sample(dataset, seed, fold, rate, value):
    return dict(dataset=dataset, seed=seed, fold=fold, rate=str(rate/10),
        ACC=value, Flat_ACC=value-1, delta_Flat_ACC=1,
        WF1=value-2, Flat_WF1=value-3, delta_Flat_WF1=1)


def test_no_seed_mean_until_all_eight_rates_are_present():
    m = importlib.import_module(MODULE)
    rows = [sample('MELD', 67, 1, r, 60) for r in range(7)]
    assert m.aggregate_rows(rows) == []


def test_no_fivefold_headline_until_all_folds_are_complete():
    m = importlib.import_module(MODULE)
    rows = [sample('IEMOCAPSix', 68, f, r, 60+f) for f in range(1, 5) for r in range(8)]
    aggregate = m.aggregate_rows(rows)
    assert not any(r['fold'] == 'all' for r in aggregate)
    assert len(aggregate) == 8


def test_three_seed_mean_and_std_do_not_pool_rates_or_folds():
    m = importlib.import_module(MODULE)
    rows = [sample('IEMOCAPSix', s, f, r, 60+s-66+f)
        for s in (66, 67, 68) for f in range(1, 6) for r in range(8)]
    aggregate = m.aggregate_rows(rows)
    mean = next(r for r in aggregate if r['seed'] == '66/67/68' and r['rate'] == 'mean')
    assert mean['ACC'] == 64
    assert mean['ACC_std'] == 1
    assert mean['delta_Flat_ACC'] == 1
