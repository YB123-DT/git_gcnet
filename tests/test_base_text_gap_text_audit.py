import importlib

import numpy as np
import pytest
import torch

from test_source_evidence_runner import _fixture


def api():
    try:
        return importlib.import_module('experiments.osram_meld_base_text_gap_text_20261019.audit')
    except ModuleNotFoundError as error:
        pytest.fail('Missing paired Base-text/Gap-text implementation: '+str(error))


def test_eligibility_uses_current_missing_and_strict_past_ever_only():
    module = api()
    av = np.array([[1, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    trajectory = np.zeros((4, 3, 4))
    trajectory[1:, 1] = [1, .5, 1, 1]
    assert module.eligible_mask(av, trajectory).tolist() == [False, True, False, True]
    av[:, 1] = 1
    assert not module.eligible_mask(av, trajectory).any()


def test_four_replays_change_only_two_source_slots_and_keep_full_exact():
    module = api()
    osram, head, local, inputs, sources, umask, av = _fixture()
    selected = umask.T.bool() & ~av[..., 1].bool()
    selected[0] = False
    captured = []
    hook = osram.emotion_adapter.register_forward_pre_hook(lambda m, a: captured.append(a[0].clone()))
    with torch.no_grad():
        full = head(osram.emotion_norm(osram.local_skip(local)+osram.emotion_adapter(inputs))
                    * umask.T[..., None])
    captured.clear()
    before = torch.get_rng_state().clone()
    original = inputs.clone()
    result = module.replay_four(osram, head, local, inputs, sources, umask, selected, full)
    hook.remove()
    assert torch.equal(result['full'], full)
    assert len(captured) == 4
    fd, ld = 6, 8
    for index, name in enumerate(('full','without_base_text','without_gap_text','without_both')):
        expected = inputs.clone()
        for row in ((0,) if index == 1 else (2,) if index == 2 else (0, 2) if index == 3 else ()):
            expected[..., ld+row*2*fd:ld+row*2*fd+fd] -= torch.where(
                selected[..., None], sources[..., row, 1, :], 0.)
        assert torch.equal(captured[index], expected)
        assert torch.equal(result[name][~selected], full[~selected])
    assert torch.equal(inputs, original) and torch.equal(before, torch.get_rng_state())


def test_empty_cohort_has_na_scores_not_zero_and_identical_four_cohorts():
    module = api()
    logits = {name:np.zeros((3, 7)) for name in module.CONDITIONS}
    arrays = dict(labels=np.array([0, 1, 2]), availability=np.ones((3,3)),
                  trajectory=np.zeros((3,3,4)), **logits)
    result = module.summarize_arrays(arrays)
    assert result['N'] == 0
    assert all(v['ACC'] is None and v['N']==0 for v in result['metrics'].values())
    assert result['interaction_CE'] is None


def test_conditional_ce_effects_and_finite_difference_use_same_cohort():
    module = api()
    arrays = dict(labels=np.array([0,1]), availability=np.array([[1,0,0],[0,0,1]]),
                  trajectory=np.tile([1., .5, 1., 1.], (2,3,1)))
    arrays.update({name:np.zeros((2,7)) for name in module.CONDITIONS})
    arrays['full'][0,0] = 3
    arrays['without_base_text'][0,0] = 2
    arrays['without_gap_text'][0,0] = 1
    result = module.summarize_arrays(arrays)
    assert result['N']==2
    m=result['metrics']
    assert result['conditional_effects']['gap_with_base']['CE_improvement'] == pytest.approx(m['without_gap_text']['CE']-m['full']['CE'])
    assert result['conditional_effects']['gap_without_base']['CE_improvement'] == pytest.approx(m['without_both']['CE']-m['without_base_text']['CE'])
    assert result['interaction_CE'] == pytest.approx(m['without_both']['CE']-m['without_base_text']['CE']-m['without_gap_text']['CE']+m['full']['CE'])
    assert result['paired_vs_full']['without_gap_text']['harms']==0


def test_invalid_alltext_or_wrong_shapes_are_rejected():
    module=api()
    with pytest.raises(ValueError):
        module.eligible_mask(np.zeros((2,2)), np.zeros((2,3,4)))


def test_report_preserves_four_equal_cohorts_and_empty_na(tmp_path):
    import csv
    module=api()
    from experiments.osram_meld_base_text_gap_text_20261019.run import write_report
    arrays=dict(labels=np.array([0,1]),availability=np.array([[1,0,0],[0,0,1]]),
                trajectory=np.tile([1.,.5,1.,1.],(2,3,1)))
    arrays.update({name:np.zeros((2,7)) for name in module.CONDITIONS})
    result=module.summarize_arrays(arrays)
    write_report(tmp_path,dict(source_commit='test',rates={'0.3':result}))
    rows=list(csv.DictReader((tmp_path/'METRICS.csv').open()))
    assert len(rows)==4 and all(row['N']=='2' for row in rows)
    assert '同样本' in (tmp_path/'RESULT.md').read_text()
