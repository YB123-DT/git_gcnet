"""Frozen text-availability interventions and reversible gradient diagnostics."""
import json

import numpy as np
import pytest
import torch


def core():
    from experiments.osram_text_condition_optimization_audit_20261022 import core as impl
    return impl


def masks():
    availability = torch.tensor([[[1, 1, 0], [1, 0, 1]],
                                 [[1, 0, 0], [1, 1, 0]],
                                 [[0, 0, 1], [1, 0, 1]],
                                 [[1, 1, 1], [1, 0, 0]]], dtype=torch.float32)
    umask = torch.tensor([[1, 0, 1, 1], [1, 1, 1, 0.]])
    return availability, umask


def test_history_seen_is_strictly_past_and_does_not_count_padding():
    availability, umask = masks()
    actual = core().history_text_seen(availability, umask)
    assert torch.equal(actual, torch.tensor([[0, 0], [0, 0], [1, 1], [1, 0]], dtype=torch.bool))
    altered = availability.clone()
    altered[1, 0, 1] = 1  # padding cannot change history
    assert torch.equal(actual, core().history_text_seen(altered, umask))


def test_paired_masks_changes_only_target_text_and_leaves_original_untouched():
    availability, umask = masks()
    before = availability.clone()
    visible, missing = core().paired_masks(availability, umask, 2, 0)
    assert visible[2, 0].tolist() == [0, 1, 1]
    assert missing[2, 0].tolist() == [0, 0, 1]
    assert torch.equal(availability, before)
    difference = visible != missing
    assert difference.sum() == 1 and difference[2, 0, 1]


@pytest.mark.parametrize('target', [(0, 0), (1, 0), (3, 1), (4, 0)])
def test_invalid_pair_targets_rejected(target):
    with pytest.raises(ValueError):
        core().paired_masks(*masks(), *target)


def parameters():
    return [('observed_set.projectors.audio.weight', torch.nn.Parameter(torch.tensor([1., 2.]))),
            ('osram.key_projectors.text.weight', torch.nn.Parameter(torch.tensor([3.]))),
            ('osram.value_projectors.text.bias', torch.nn.Parameter(torch.tensor([4.]))),
            ('osram.query_projector.weight', torch.nn.Parameter(torch.tensor([5.]))),
            ('osram.alpha_logits', torch.nn.Parameter(torch.tensor([6.]))),
            ('osram.emotion_adapter.1.weight', torch.nn.Parameter(torch.tensor([7.]))),
            ('osram.speaker_embedding.weight', torch.nn.Parameter(torch.tensor([8.])))]


def test_gradient_geometry_disjoint_partition_none_and_antiparallel():
    params = parameters()
    visible = [torch.ones_like(p) for _, p in params]
    missing = [-g for g in visible]
    missing[2] = None
    result = core().gradient_geometry(params, visible, missing)
    assert result['groups']['ObservedEncoder']['cosine'] == pytest.approx(-1)
    assert result['groups']['WriteValue']['cosine'] is None
    assert result['groups']['WriteValue']['missing_norm'] == 0
    assert result['groups']['Other']['parameter_count'] == 1
    assert result['overall']['dot'] == pytest.approx(-7)
    assert sum(len(v) for v in result['partition'].values()) == len(params)
    json.dumps(result, allow_nan=False)


def test_gradient_geometry_rejects_nonfinite_and_mismatched_gradients():
    params = parameters()
    grads = [torch.ones_like(p) for _, p in params]
    grads[0][0] = float('nan')
    with pytest.raises(ValueError):
        core().gradient_geometry(params, grads, grads)
    with pytest.raises(ValueError):
        core().gradient_geometry(params, [], [])


def test_temporary_sgd_restores_every_value_on_exception_without_touching_grad():
    params = parameters()
    snapshots = [p.detach().clone() for _, p in params]
    for _, p in params:
        p.grad = torch.full_like(p, 5)
    grads = [torch.ones_like(p) for _, p in params]
    with pytest.raises(RuntimeError, match='forced'):
        with core().temporary_sgd(params, grads, learning_rate=.01) as stats:
            assert stats['gradient_norm'] == pytest.approx(np.sqrt(8))
            assert stats['update_norm'] > 0
            assert stats['relative_update'] > 0
            assert all(torch.allclose(p, old-.01) for (_, p), old in zip(params, snapshots))
            raise RuntimeError('forced')
    assert all(torch.equal(p, old) for (_, p), old in zip(params, snapshots))
    assert all(torch.equal(p.grad, torch.full_like(p, 5)) for _, p in params)


def test_temporary_sgd_none_gradient_and_mapping_supported():
    params = parameters()
    with core().temporary_sgd(params, {name: None for name, _ in params}) as stats:
        assert stats['update_norm'] == 0
        assert stats['relative_update'] == 0


def test_summary_corrects_harms_and_neutral_only_filtered_from_polarity():
    labels = np.array([1., -1., 0., -1.])
    full = np.array([1., 1., 2., -1.])
    local = np.array([-1., -1., 1., -1.])
    availability = np.array([[1, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1]])
    seen = np.array([0, 1, 0, 0], bool)
    result = core().group_summary(labels, full, local, availability, seen)
    all_row = result['overall']
    assert all_row['N'] == 4 and all_row['polarity_N'] == 3
    assert all_row['corrections'] == 1 and all_row['harms'] == 1
    assert all_row['full']['ACC'] == pytest.approx(200/3)
    assert all_row['full']['MSE'] == pytest.approx(2.)
    assert all_row['local_minus_full_MSE'] == pytest.approx(-.75)
    assert result['Textmissing_never']['N'] == 2
    json.dumps(result, allow_nan=False)
