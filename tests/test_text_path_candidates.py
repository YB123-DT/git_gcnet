"""Two extra readouts share the one unchanged causal Memory scan."""
from types import SimpleNamespace

import pytest
import torch

from experiments.osram_meld_base_text_gap_text_20261019.audit import replay_four
from experiments.osram_text_path_router_20261019.candidates import candidate_readouts, pack_view
from tests.test_source_evidence_runner import _fixture


def _case():
    osram, head, local, inputs, sources, umask, av = _fixture()
    full = head(osram.emotion_norm(osram.local_skip(local) + osram.emotion_adapter(inputs))
                * umask.T[..., None])
    eligible = umask.T.bool() & ~av[..., 1].bool()
    return osram, head, local, inputs, sources, umask, av, full, eligible


def test_two_readouts_match_old_four_replay_and_do_not_touch_other_rows():
    osram, head, local, inputs, sources, umask, av, full, eligible = _case()
    expected = replay_four(osram, head, local, inputs, sources, umask, eligible, full)
    before = [x.clone() for x in (local, inputs, sources, full)]
    calls = []
    hook = head.register_forward_pre_hook(lambda module, values: calls.append(values[0].clone()))
    rng = torch.get_rng_state().clone()
    result = candidate_readouts(osram, head, local, inputs, sources[..., 0, 1, :],
                                sources[..., 2, 1, :], umask, eligible, full)
    hook.remove()
    assert len(calls) == 2
    assert result.shape == (4, 2, 3, 4)
    assert torch.equal(result[..., 0, :], full)
    assert torch.equal(result[..., 1, :], expected['without_gap_text'])
    assert torch.equal(result[..., 2, :], expected['without_base_text'])
    assert torch.equal(result[~eligible], full[~eligible][:, None, :].expand(-1, 3, -1))
    assert not result.requires_grad
    assert torch.equal(rng, torch.get_rng_state())
    for old, current in zip(before, (local, inputs, sources, full)):
        assert torch.equal(old, current)


@pytest.mark.parametrize('fault', ['backward', 'shape', 'padding', 'train'])
def test_invalid_or_training_readout_is_rejected(fault):
    osram, head, local, inputs, sources, umask, av, full, eligible = _case()
    base = sources[..., 0, 1, :]
    if fault == 'backward': inputs[..., 8 + 6] = .2
    if fault == 'shape': base = base[..., :-1]
    if fault == 'padding': eligible[-1, -1] = True
    if fault == 'train': head.train()
    with pytest.raises(ValueError):
        candidate_readouts(osram, head, local, inputs, base, sources[..., 2, 1, :],
                           umask, eligible, full)


def test_pack_view_scans_once_uses_actual_skip_input_and_flattens_batch_major(monkeypatch):
    from experiments.osram_tasr_20261018 import runner_helpers as helper
    osram, head, local, inputs, sources, umask, av, full, _ = _case()
    trajectory = torch.zeros(4, 2, 3, 4)
    trajectory[1:, :, 1] = torch.tensor([1., .5, 1., 1.])
    osram.last_source_trace = dict(source_base=sources[..., 0, :, :],
        source_gap=sources[..., 1:, :, :], trajectory=trajectory)
    model = SimpleNamespace(osram=osram, smax_fc=head, training=False)
    calls = []

    def forward(actual, view):
        assert actual is model
        calls.append(1)
        return head(osram.emotion_norm(osram.local_skip(local) + osram.emotion_adapter(inputs))
                    * umask.T[..., None])

    monkeypatch.setattr(helper, 'forward', forward)
    labels = torch.arange(8).reshape(2, 4)
    view = dict(umask=umask, availability=av, labels=labels)
    result = pack_view(dict(model=model), view)
    valid = umask.bool()
    assert len(calls) == 1
    assert torch.equal(result['local'], local.transpose(0, 1)[valid])
    assert torch.equal(result['base_text'], sources[..., 0, 1, :].transpose(0, 1)[valid])
    assert torch.equal(result['gap_text'], sources[..., 2, 1, :].transpose(0, 1)[valid])
    assert torch.equal(result['labels'], labels[valid])
    expected = (~av[..., 1].bool() & trajectory[..., 1, 3].bool()).transpose(0, 1)[valid]
    assert torch.equal(result['eligible'], expected)
    assert all(not value.requires_grad for value in result.values())
def test_candidate_head_layout_matches_original_multiplicative_padding():
    from test_source_evidence_runner import _fixture
    from experiments.osram_text_path_router_20261019.candidates import candidate_readouts
    osram,head,local,inputs,sources,umask,av=_fixture()
    with torch.no_grad():
        hidden=osram.emotion_norm(osram.local_skip(local)+osram.emotion_adapter(inputs))
        reference=hidden*umask.T.bool()[...,None].to(hidden.dtype)
        full=head(reference)
    seen=[]
    hook=head.register_forward_pre_hook(lambda module,args:seen.append(args[0].stride()))
    selected=umask.T.bool()&~av[...,1].bool()
    candidate_readouts(osram,head,local,inputs,sources[...,0,1,:],sources[...,2,1,:],umask,selected,full)
    hook.remove()
    assert seen==[reference.stride(),reference.stride()]
