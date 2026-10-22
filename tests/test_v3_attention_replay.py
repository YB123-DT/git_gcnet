import numpy as np
import pytest


def test_attention_audit_scope_is_original_seed66_not_current_direct_or_capacity():
    from experiments.osram_v3_source_attention_audit_20261022.run import source_runs
    assert len(source_runs('CMUMOSI')) == len(source_runs('CMUMOSEI')) == 1
    for d in ('IEMOCAPFour', 'IEMOCAPSix'):
        runs = source_runs(d)
        assert [f for f, _ in runs] == list(range(1, 6))
        assert all('seed_66' in str(p) for _, p in runs)
    with pytest.raises(ValueError):
        source_runs('MELD')


def test_prediction_replay_matches_same_cohort_mask_and_saved_logits():
    from experiments.osram_v3_source_attention_audit_20261022.run import verify_replay
    arrays = dict(labels=np.array([1., -1.]), availability=np.array([[1, 0, 0], [0, 1, 1]]),
                  conversation_ids=np.array(['a', 'b']), utterance_indices=np.array([1, 2]),
                  speakers=np.array([0, 1]), prediction=np.array([[.1], [-.2]]))
    saved = dict(arrays, prediction=arrays['prediction'].ravel())
    assert verify_replay(arrays, saved)['prediction_max_abs_error'] == 0.
    with pytest.raises(AssertionError):
        verify_replay(dict(arrays, availability=np.ones((2, 3))), saved)
    with pytest.raises(AssertionError):
        verify_replay(dict(arrays, prediction=np.array([[-.1], [-.2]])), saved)
