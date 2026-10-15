import sys
from pathlib import Path

import numpy as np
import pytest
import os

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common_data import fixed_masks, random_masks, load_dataset


def test_masks_are_identity_based_and_nonempty():
    uids = ['u' + str(i) for i in range(50)]
    whole = fixed_masks('MELD', 'dialogue', uids, .3)
    np.testing.assert_array_equal(whole, fixed_masks('meld', 'dialogue', uids, .3))
    np.testing.assert_array_equal(whole[::-1], fixed_masks('MELD', 'dialogue', uids[::-1], .3))
    np.testing.assert_array_equal(whole[:11], fixed_masks('MELD', 'dialogue', uids[:11], .3))
    assert np.all(whole.sum(1) >= 1)
    assert not np.array_equal(whole, fixed_masks('MELD', 'other', uids, .3))


@pytest.mark.parametrize('mr', [0, .1, .3, .6, .7])
def test_mask_expected_rates_and_caps(mr):
    masks = random_masks(100000, mr, np.random.default_rng(66))
    assert masks.dtype == np.float32
    assert np.all(masks.sum(1) >= 1)
    assert abs(1 - masks.mean() - min(mr, 2 / 3)) < .003
    if mr == .7:
        assert np.all(masks.sum(1) == 1)


def test_fixed_mask_rate():
    masks = fixed_masks('MELD', 'd', list(map(str, range(15000))), .4)
    assert abs(1 - masks.mean() - .4) < .01


@pytest.mark.parametrize('name,utterances,dims,classes', [
    ('MELD', 13708, [300, 1024, 342], 7),
    ('IEMOCAPFour', 5531, [512, 1024, 1024], 4),
    ('IEMOCAPSix', 7433, [512, 1024, 1024], 6),
])
@pytest.mark.skipif(os.environ.get('BASELINE_TEST_DATA') != '1', reason='Use verify_cpu.py --with-data with local features installed')
def test_local_data_integrity(name, utterances, dims, classes):
    data = load_dataset(name)
    assert data['metadata']['utterance_count'] == utterances
    assert data['metadata']['dimensions'] == dims
    assert data['metadata']['num_classes'] == classes
    records = data['records']
    assert len({r['id'] for r in records}) == len(records)
    if name == 'MELD':
        assert {r['split'] for r in records} == {'train', 'valid', 'test'}
        assert max(int(r['speakers'].max()) for r in records) == 8
    else:
        assert {r['session'] for r in records} == set(range(1, 6))
