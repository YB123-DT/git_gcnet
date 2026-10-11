import numpy as np
from experiments.osram_flat_base_gap_interaction_20261011.run import interaction, adjacent, summarize


def test_algebra_and_polarity_pairing():
    f00 = np.array([1., -1.])
    b, g = np.array([2., 3.]), np.array([4., 5.])
    assert np.array_equal(interaction(f00, f00+b, f00+g, f00+b+g), np.zeros(2))
    assert np.array_equal(interaction(f00, f00+b, f00+g, f00+b+g+7), np.full(2, 7.))
    assert adjacent(1., -1.) == 'opposite'
    assert adjacent(-1., -2.) == 'same'
    assert adjacent(0., 1.) == adjacent(None, 1.) == 'excluded'


def test_summary_uses_nonzero_labels_and_paired_counts():
    rows = [dict(label=y, f00=0., f10=a, f01=0., f11=f,
                 interaction=f-a) for y,a,f in [(1.,-1.,1.),(-1.,-1.,1.),(0.,1.,-1.)]]
    s = summarize(rows)
    assert s['n'] == 3 and s['nonneutral_n'] == 2
    assert s['corrections'] == s['harms'] == 1
    assert summarize([])['full_wf1'] is None
