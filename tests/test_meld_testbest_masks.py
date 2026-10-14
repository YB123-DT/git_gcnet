from types import SimpleNamespace

import numpy as np
from gcnet_modality_jepa.gcnet_legacy_random_mask import random_mask
from gcnet_modality_jepa.protocol import EpochSeededSubsetSampler, SeedBundle
from torch.utils.data import BatchSampler

from experiments.osram_meld_testbest_ua_20261014.recover import recover_masks


def toy_loaders():
    dataset = SimpleNamespace(vids=['a', 'b', 'c'],
                              videoLabelsNew={'a': [0, 1], 'b': [1], 'c': [2, 1, 0]},
                              videoSpeakersNew={'a': np.array([0, 1]), 'b': np.array([2]),
                                                'c': np.array([0, 3, 1])})
    loaders = []
    for indices in ([0, 1], [2]):
        sampler = EpochSeededSubsetSampler(indices, 123)
        loaders.append(SimpleNamespace(dataset=dataset, sampler=sampler,
                                       batch_sampler=BatchSampler(sampler, 2, False)))
    return loaders


def test_recovery_uses_original_draw_order_without_touching_global_numpy():
    train, test = toy_loaders()
    config = SimpleNamespace(seed=66, epochs=2, train_missing_rates=[0., .7])
    state = np.random.get_state()
    selected, final = recover_masks(config, train, test, {'0.7': 1}, rates=(0., .7))
    assert np.array_equal(state[1], np.random.get_state()[1])
    rng = np.random.RandomState(SeedBundle(66).derive('missing_m3_model_init:fold:5'))
    expected = None
    for epoch in range(2):
        train.sampler.set_epoch(epoch)
        for batch_index, batch in enumerate(train.batch_sampler):
            length = max(len(train.dataset.videoLabelsNew[train.dataset.vids[i]]) for i in batch)
            rate = config.train_missing_rates[(epoch + batch_index) % 2]
            random_mask(3, length * len(batch), rate, rng)
            random_mask(3, length * len(batch), rate, rng)
        for rate in (0., .7):
            host = random_mask(3, 3, rate, rng)
            guest = random_mask(3, 3, rate, rng)
            if epoch == 0 and rate == .7:
                expected = np.stack([host[0], guest[1], guest[2]])
    np.testing.assert_array_equal(selected['0.7'], expected)
    assert final['0.0'].shape == final['0.7'].shape == (3, 3)
