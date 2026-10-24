import numpy as np


def test_probe_covers_largest_padding_rectangle_and_most_valid_tokens():
    from experiments.osram_mosei_flat128_nested_20261024.probe import choose_probe_batches
    records = []
    for mask in (np.ones((2,3)), np.array([[1]*8,[1]+[0]*7]), np.ones((3,5))):
        records.append([None]*7+[mask])
    assert choose_probe_batches(records) == [1,2]
