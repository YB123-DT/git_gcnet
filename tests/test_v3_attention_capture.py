import numpy as np
import torch

from gcnet_missing_m3.source_role_v3 import HierarchicalSourceRoleV3
from test_source_role_v3 import inputs


def test_head_capture_preserves_exact_outputs_rng_and_role_masks():
    from experiments.osram_v3_source_attention_audit_20261022.capture import AttentionCapture
    module = HierarchicalSourceRoleV3(local_dim=5, forward_dim=7).eval()
    args = inputs()
    for layer in module.decoders:
        torch.nn.init.normal_(layer.weight, std=.01)
    state = torch.get_rng_state().clone()
    expected = module(*args)
    after = torch.get_rng_state().clone()
    torch.set_rng_state(state)
    with AttentionCapture(module) as capture:
        capture.conversation_ids = ['conv_a', 'conv_b']
        actual = module(*args)
        arrays = capture.arrays
    assert torch.equal(expected, actual)
    assert torch.equal(after, torch.get_rng_state())
    assert arrays['attention'].shape == (int(args[-1].sum()), 4, 4, 3)
    active = arrays['role_active']
    np.testing.assert_allclose(arrays['attention'].sum(-1)[active], 1., atol=1e-6)
    assert not arrays['attention'][~active].any()
    assert not arrays['attention'][~np.broadcast_to(arrays['seen'][:, None, None, :],
                                                  arrays['attention'].shape)].any()
    np.testing.assert_allclose(arrays['attention'].mean(2),
        module.last_diagnostics['cross_attention_weights'].transpose(0, 1)[args[-1].bool()].numpy())


def test_query_donors_never_cross_role_availability_source_set_or_same_conversation():
    from experiments.osram_v3_source_attention_audit_20261022.capture import matched_donors
    role = np.array([0, 0, 0, 1, 0])
    avail = np.array([1, 1, 2, 1, 1])
    seen = np.array([7, 7, 7, 7, 3])
    ids = np.array(['a', 'b', 'c', 'd', 'e'])
    donor = matched_donors(role, avail, seen, ids)
    assert donor.tolist() == [1, 0, -1, -1, -1]
