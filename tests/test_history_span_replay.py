"""Frozen history windows, independently checked against original scans."""
import copy
import torch

from gcnet_missing_m3.osram import MODALITIES, OSRAMBackbone
from experiments.osram_iemocap_history_span_20261026.replay import (
    ScanCapture, replay_reads,
)


def fixture():
    torch.manual_seed(19)
    osram = OSRAMBackbone(latent_dim=8, output_dim=10, num_heads=2,
        key_dim=3, value_dim=4, n_speakers=2, dropout=0., bidirectional=False).eval()
    valid = torch.tensor([[1,1],[1,1],[0,1],[1,1],[1,0],[1,0]]).bool()
    a = torch.randint(0,2,(6,2,3)).float()
    a[...,0] = 1
    a *= valid[...,None]
    keys = {m: torch.randn(6,2,2,3) for m in MODALITIES}
    values = {m: torch.randn(6,2,2,4) for m in MODALITIES}
    q = torch.randn(6,2,4,2,3)
    with torch.no_grad(), ScanCapture(osram) as trace:
        expected = osram._scan(keys, values, q, a, valid, False)
    return osram, trace, expected


def test_full_matches_original_and_zero_has_no_history():
    osram, trace, expected = fixture()
    rng = torch.random.get_rng_state().clone()
    for result, gold in zip(replay_reads(osram,trace,None),expected[:2]):
        torch.testing.assert_close(result,gold,rtol=0,atol=0)
    for result in replay_reads(osram,trace,0):
        assert torch.count_nonzero(result) == 0
    assert torch.equal(rng,torch.random.get_rng_state())


def test_windows_equal_independent_trimmed_original_scans():
    osram, trace, _ = fixture()
    for window in (1,2,4,8):
        actual = replay_reads(osram,trace,window,chunk_size=3)
        for b in range(2):
            times = trace.valid[:,b].nonzero().flatten().tolist()
            for rank,t in enumerate(times):
                chosen = times[max(0,rank-window):rank+1]
                with torch.no_grad():
                    gold = osram._scan(
                        {m:v[chosen,b:b+1] for m,v in trace.keys.items()},
                        {m:v[chosen,b:b+1] for m,v in trace.values.items()},
                        trace.queries[chosen,b:b+1],trace.availability[chosen,b:b+1],
                        trace.valid[chosen,b:b+1],False)
                for output,ref in zip(actual,gold[:2]):
                    torch.testing.assert_close(output[t,b],ref[-1,0],rtol=2e-5,atol=2e-6)


def test_padding_dialogue_isolation_and_future_causality():
    osram, trace, _ = fixture()
    expected = replay_reads(osram,trace,2)
    changed = copy.copy(trace)
    changed.keys = {m:v.clone() for m,v in trace.keys.items()}
    changed.values = {m:v.clone() for m,v in trace.values.items()}
    for m in MODALITIES:
        changed.keys[m][4:] += 10
        changed.values[m][4:] -= 7
        changed.values[m][:,1] += 30
        changed.values[m][2,0] += 99  # interior padding is not a past turn
    for actual,gold in zip(replay_reads(osram,changed,2),expected):
        torch.testing.assert_close(actual[:4,0],gold[:4,0],rtol=0,atol=0)
        assert torch.count_nonzero(actual[~trace.valid]) == 0
        assert torch.count_nonzero(actual[0]) == 0


def test_capture_restores_scan_and_does_not_change_forward_or_queries():
    osram, trace, expected = fixture()
    assert '_scan' not in osram.__dict__
    queries = trace.queries.clone()
    local_rng = torch.random.get_rng_state().clone()
    replay_reads(osram,trace,2)
    torch.testing.assert_close(trace.queries,queries,rtol=0,atol=0)
    assert torch.equal(local_rng,torch.random.get_rng_state())
    with torch.no_grad():
        original = osram._scan(trace.keys,trace.values,trace.queries,
            trace.availability,trace.valid,False)
    for gold,ref in zip(expected[:2],original[:2]):
        torch.testing.assert_close(gold,ref,rtol=0,atol=0)


def test_inactive_gap_is_always_zero_and_invalid_window_rejected():
    import pytest
    osram,trace,_ = fixture()
    for window in (0,1,2,None):
        _,gap = replay_reads(osram,trace,window)
        assert torch.count_nonzero(gap[trace.availability.bool()]) == 0
    with pytest.raises(ValueError):
        replay_reads(osram,trace,-1)
