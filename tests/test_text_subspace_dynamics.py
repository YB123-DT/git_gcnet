import importlib.util

import torch


def test_audit_available():
    assert importlib.util.find_spec('experiments.osram_text_subspace_20260915.audit_dynamics') is not None


def test_gradient_summary_uses_same_coordinates_and_zero_is_undefined():
    from experiments.osram_text_subspace_20260915.audit_dynamics import gradient_summary
    x = torch.nn.Parameter(torch.tensor([2., 3.]))
    unused = torch.nn.Parameter(torch.ones(1))
    losses = {'a': x[0], 'b': 2*x[1], 'zero': x.sum()*0}
    result = gradient_summary(losses, [('x', x), ('unused', unused)])
    assert result['norms'] == {'a': 1., 'b': 2., 'zero': 0.}
    assert result['cosines']['a__b'] == 0.
    assert result['cosines']['a__zero'] is None
    assert all(p.grad is None for p in (x, unused))


def test_common_parameter_support_excludes_private_heads():
    from experiments.osram_text_subspace_20260915.audit_dynamics import gradient_summary
    shared = torch.nn.Parameter(torch.ones(2))
    private = torch.nn.Parameter(torch.ones(1))
    result = gradient_summary({'emotion': shared.sum()+100*private.sum(),
                               'weighted_jepa': -.1*shared.sum()},
                              [('shared', shared), ('private', private)], common_support=True)
    assert result['parameter_names'] == ['shared']
    assert abs(result['cosines']['emotion__weighted_jepa'] + 1.) < 1e-8
    assert abs(result['norms']['weighted_jepa']/result['norms']['emotion']-.1) < 1e-7


def test_rank_baselines_share_samples_and_are_pure():
    from experiments.osram_text_subspace_20260915.audit_dynamics import rank_baselines
    r0, r = torch.nn.Linear(256,32), torch.nn.Linear(256,32)
    z = torch.randn(18,256)
    rng = torch.get_rng_state().clone()
    result = rank_baselines(z, r0, r)
    assert set(result) == {'teacher_text', 'initial_R0', 'trained_R'}
    assert all(v['count'] == 18 for v in result.values())
    assert result['teacher_text']['dimension'] == 256
    assert result['trained_R']['dimension'] == 32
    assert torch.equal(rng, torch.get_rng_state())
    assert all(p.grad is None for m in (r0,r) for p in m.parameters())
