import importlib
import numpy as np
import torch


def module():
    return importlib.import_module('experiments.osram_gap_increment_audit_20261003.run')


def test_reconstruct_scan_diagnostics_token_order():
    valid = torch.tensor([[True, True], [True, False]])
    diagnostics = {name: {metric: list(range(6)) for metric in ('rho', 'eta', 'cosine')}
                   for name in ('audio', 'text', 'visual')}
    result = module().unpack_diagnostics(diagnostics, valid, 2)
    assert result.shape == (2, 2, 3, 3)
    assert result[0, 1, 0, 0] == 2.5
    assert result[1, 0, 0, 0] == 4.5
    assert np.isnan(result[1, 1]).all()


def test_observables_ignore_zero_half_and_inactive_gaps():
    base = torch.tensor([1., 0., 900., 900.])
    gap = torch.tensor([[float('nan')]*4, [0., 2., 900., 900.], [float('nan')]*4])
    obs = module().observables(base, gap, torch.tensor([1, 0, 1]), 2,
                               np.ones((3, 3)), .2)
    assert obs['obs_base_norm'] == 1.
    assert obs['obs_gap_T_norm'] == 2.
    assert obs['obs_gap_A_norm'] == 0.
    assert obs['obs_gap_base_ratio'] == 2.
    assert obs['obs_base_gap_cos_mean'] == 0.
    assert np.isnan(obs['obs_base_gap_A_cos'])


def test_readout_replay_keeps_local_skip_and_changes_only_evidence():
    from types import SimpleNamespace
    adapter = torch.nn.Linear(5, 2, bias=True)
    osram = SimpleNamespace(emotion_adapter=adapter, emotion_norm=torch.nn.Identity())
    model = SimpleNamespace(osram=osram, smax_fc=torch.nn.Linear(2, 1))
    x, skip = torch.ones(2, 3, 5), torch.full((2, 3, 2), 3.)
    full = model.smax_fc(skip + adapter(x))
    local, base, replay = module().replay_readouts(model, x, skip, 1, 1)
    torch.testing.assert_close(replay, full, atol=0, rtol=0)
    expected = x.clone(); expected[..., 1:] = 0
    torch.testing.assert_close(local, model.smax_fc(skip + adapter(expected)))
    expected = x.clone(); expected[..., 2:] = 0
    torch.testing.assert_close(base, model.smax_fc(skip + adapter(expected)))
    assert torch.equal(x, torch.ones_like(x))


def test_real_model_profile_keeps_original_predictions_and_one_scan():
    from gcnet_missing_m3.model import MissingM3GraphModel
    from tests.test_completion_memory_write import model_kwargs, inputs
    kwargs = model_kwargs(); kwargs.update(completion_path='none')
    model = MissingM3GraphModel(**kwargs).eval()
    x,a,q,u,lengths = inputs()
    with torch.no_grad():
        expected = model([x],a,q,u,lengths,predict_missing=False)[0]
        with module().Capture(model) as capture:
            actual = model([x],a,q,u,lengths,predict_missing=False)[0]
    assert torch.equal(expected, actual)
    assert capture.scans == 1
    assert len(capture.batches) == int(u.sum())


def test_head_ablation_only_requested_forward_slice():
    x = torch.arange(34.).reshape(1, 1, 34)
    for evidence in ('base', 'gap'):
        result = module().mask_head_input(x, 2, 8, 2, 1, evidence)
        expected = x.clone()
        slots = (0,) if evidence == 'base' else (1, 2, 3)
        for slot in slots:
            start = 2 + 8 * slot + 2
            expected[..., start:start+2] = 0
        assert torch.equal(result, expected)
        assert torch.equal(x, torch.arange(34.).reshape(1, 1, 34))


def test_real_head_replays_do_not_add_scan_or_change_full():
    from gcnet_missing_m3.model import MissingM3GraphModel
    from tests.test_completion_memory_write import model_kwargs, inputs
    kwargs = model_kwargs(); kwargs.update(completion_path='none')
    model = MissingM3GraphModel(**kwargs).eval()
    x,a,q,u,lengths = inputs()
    with torch.no_grad():
        expected = model([x],a,q,u,lengths,predict_missing=False)[0]
        with module().Capture(model, head_ablation=True) as capture:
            actual = model([x],a,q,u,lengths,predict_missing=False)[0]
    assert torch.equal(expected, actual)
    assert capture.scans == 1
    assert len(capture.head_batches) == 2 * model.osram.num_heads * int(u.sum())


def test_query_audit_uses_existing_perhead_diagnostics_and_nan_zero_reads():
    valid=torch.tensor([[True],[True]])
    availability=torch.tensor([[[1,0,1]],[[0,1,1]]])
    queries=torch.ones(2,1,4,2,2)
    queries[:,:,0,:,1]=0
    base=torch.zeros(2,1,4);base[1]=1
    gap=torch.zeros(2,1,3,4);gap[1,:,0]=1
    diagnostics={m:{key:list(range(4)) for key in ('rho','eta','cosine')}
                 for m in ('audio','text','visual')}
    rows=module().query_observables(queries,base,gap,availability,valid,diagnostics,2,2)
    assert len(rows)==4
    assert rows[0]['modality']=='T' and rows[2]['modality']=='A'
    assert np.isnan(rows[0]['cos_base_gap_read'])
    assert abs(rows[2]['cos_base_gap_read']-1)<1e-6
    assert rows[3]['cos_gap_residual_query']==3
    assert rows[3]['rho']==3 and rows[3]['eta']==3


def test_real_query_audit_keeps_full_prediction_and_single_scan():
    from gcnet_missing_m3.model import MissingM3GraphModel
    from tests.test_completion_memory_write import model_kwargs, inputs
    kwargs=model_kwargs();kwargs.update(completion_path='none')
    model=MissingM3GraphModel(**kwargs).eval()
    x,a,q,u,lengths=inputs()
    with torch.no_grad():
        expected=model([x],a,q,u,lengths,predict_missing=False)[0]
        with module().Capture(model,query_audit=True) as capture:
            actual=model([x],a,q,u,lengths,predict_missing=False)[0]
    assert torch.equal(expected,actual)
    assert capture.scans==1
    assert len(capture.query_batches)==int(((1-a)*u.T.unsqueeze(-1)).sum())*model.osram.num_heads
