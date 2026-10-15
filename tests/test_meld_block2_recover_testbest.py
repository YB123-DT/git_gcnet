"""Recover exact metrics without applying another seed's positional masks."""
import importlib
import importlib.util

import numpy as np
import pytest

NAME = 'experiments.osram_meld_block2_auto_20261014.recover_testbest'


def module():
    assert importlib.util.find_spec(NAME) is not None, 'identity-safe metric recovery is missing'
    return importlib.import_module(NAME)


def example():
    return {'conversation_ids': np.array(['a', 'a', 'b']),
            'utterance_indices': np.array([0, 1, 0]),
            'labels': np.array([0, 2, 1]), 'speakers': np.array([0, 1, 2])}


def test_permuted_test_loader_order_is_not_a_label_change():
    source = example()
    shuffled = {key: value[[2, 0, 1]] for key, value in source.items()}
    module().verify_identity(shuffled, source)


def test_correct_labels_at_wrong_ids_are_rejected():
    check = module().verify_identity
    source = example()
    wrong = {key: value.copy() for key, value in source.items()}
    wrong['labels'] = np.array([1, 0, 2])
    with pytest.raises((AssertionError, ValueError)):
        check(wrong, source)


def test_duplicate_utterance_keys_are_rejected():
    source = example()
    wrong = {key: value.copy() for key, value in source.items()}
    wrong['utterance_indices'][1] = 0
    with pytest.raises(ValueError):
        module().verify_identity(wrong, source)
