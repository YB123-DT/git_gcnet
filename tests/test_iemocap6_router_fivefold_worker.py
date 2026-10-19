from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import pytest
import torch

from experiments.osram_iemocap6_router_fivefold_20261019 import worker


def test_only_remaining_folds_can_be_selected():
    for fold in (1, 2, 3, 4):
        cfg = worker.make_config(fold, 67)
        assert (cfg.fold, cfg.seed, cfg.epochs) == (fold, 67, 100)
        assert cfg.learning_rate == .001 and cfg.lr_schedule == 'constant'
        assert cfg.missing_mask_protocol == 'gcnet-random-mask-v1'
        assert cfg.checkpoint_selection == 'validation'
        assert cfg.osram_readout_fusion == 'flat' and cfg.osram_output_dim == 1600
    with pytest.raises(ValueError, match='fold5'):
        worker.make_config(5, 66)


def test_holdout_uses_only_chosen_fold_training_conversations():
    vids = [f'conversation_{i}' for i in range(12)]
    train, validation, split = worker.partition_fold(tuple(range(10)), (10, 11), vids, 3)
    assert set(train) | set(validation) == set(range(10))
    assert not set(train) & set(validation)
    assert not (set(train) | set(validation)) & {10, 11}
    assert split['fold'] == 3 and split['seed'] == 66
    assert split['backbone_previously_saw_adapter_validation'] is False
    with pytest.raises(ValueError, match='overlap'):
        worker.partition_fold(tuple(range(10)), (9, 10), vids, 3)


def test_fold_selection_never_silently_returns_fold5():
    loaders = list(range(5))
    assert worker.select_fold(loaders, loaders, loaders, 2) == (1, 1, 1)
    with pytest.raises(ValueError, match='five'):
        worker.select_fold(loaders[:4], loaders, loaders, 2)


def test_parser_contains_full_recovery_and_provenance_interface():
    args = worker.parser().parse_args(['--fold', '3', '--seed', '68', '--mode', 'joint',
        '--phase', 'train', '--source-commit', 'source', '--gpu-uuid', 'GPU-test',
        '--output', '/tmp/example', '--resume'])
    assert args.fold == 3 and args.seed == 68 and args.resume


def test_frozen_bank_is_common_between_seed_pairs_and_restores_rng():
    from gcnet_modality_jepa.gcnet_legacy_random_mask import GCNetLegacyRandomMaskSchedule

    class Loader(list):
        sampler = SimpleNamespace(set_epoch=lambda epoch: None)

    class Trainer:
        _move_batch = staticmethod(lambda raw, device: raw)

        @staticmethod
        def _schedules(cfg, split):
            assert cfg.seed == 66
            return {i / 10: GCNetLegacyRandomMaskSchedule('IEMOCAPSix', split, cfg.fold, i / 10, 66)
                    for i in range(8)}

        @staticmethod
        def _prepare_view(data, schedule, epoch, dimensions):
            host, guest = schedule.generate_batch(data[-1], data[7], epoch)
            speaker = data[6].T[..., None].bool()
            return {'availability': torch.where(speaker, guest, host)}

    raw = [None] * 10
    raw[6] = torch.tensor([[0, 1, 0], [1, 0, 0]])
    raw[7] = torch.tensor([[1., 1., 1.], [1., 1., 0.]])
    raw[8] = torch.zeros(2, 3, dtype=torch.long)
    raw[-1] = ['first', 'second']
    loader = Loader([raw])
    bundle = dict(trainer=Trainer, cfg=worker.make_config(2, 67),
        test_loader=loader, validation_loader=loader, dimensions=(1, 1, 1),
        model=torch.nn.Linear(1, 1))
    state = np.random.get_state()
    first = worker.build_mask_bank(bundle, 'test')
    np.testing.assert_array_equal(state[1], np.random.get_state()[1])
    bundle['cfg'] = replace(bundle['cfg'], seed=68)
    second = worker.build_mask_bank(bundle, 'test')
    for rate in first:
        torch.testing.assert_close(first[rate][0]['availability'], second[rate][0]['availability'], rtol=0, atol=0)
        availability = first[rate][0]['availability']
        assert availability[:, 1][2].eq(0).all()
        counts = availability.sum(-1)[raw[7].T.bool()]
        assert (counts >= 1).all()
        if rate == '0.7':
            assert counts.eq(1).all()
    selected, references = worker.selected_from_bank(first)
    assert selected['0.0']['availability'].shape == (5, 3)
    assert selected['0.0']['availability'].sum() == 15
    assert all(len(row['mask_sha256']) == 64 for row in references.values())
    assert 'checkpoint' not in references['0.3']


def test_bank_uses_real_trainer_host_guest_visibility_and_original_schedule():
    from gcnet_missing_m3 import train_gcnet as trainer
    from gcnet_modality_jepa.gcnet_legacy_random_mask import GCNetLegacyRandomMaskSchedule
    from experiments.osram_tasr_20261018 import runner_helpers as h

    class Loader(list):
        sampler = SimpleNamespace(set_epoch=lambda epoch: None)

    raw = [torch.ones(3, 2, 1) for _ in range(6)]
    raw += [torch.tensor([[0, 1, 0], [1, 0, 0]]),
        torch.tensor([[1., 1., 1.], [1., 1., 0.]]), torch.zeros(2, 3, dtype=torch.long),
        ['first', 'second']]
    cfg = worker.make_config(1, 67)
    bundle = dict(trainer=trainer, cfg=cfg, dimensions=(1, 1, 1),
        model=torch.nn.Linear(1, 1), test_loader=Loader([raw]))
    bank = worker.build_mask_bank(bundle, 'test')
    with h.isolated_rng(66+76194):
        schedules = trainer._schedules(replace(cfg, seed=66), 'test')
        for rate, schedule in schedules.items():
            assert isinstance(schedule, GCNetLegacyRandomMaskSchedule)
            assert schedule.fold == 1
            host, guest = schedule.generate_batch(raw[-1], raw[7], 0)
            # Production generate_inputs selects host when speaker==0, guest==1.
            expected = torch.where(raw[6].T[..., None].bool(), guest, host).float()
            torch.testing.assert_close(bank[f'{rate:.1f}'][0]['availability'], expected, rtol=0, atol=0)
