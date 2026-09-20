import torch

from gcnet_missing_m3.gap_query_swap import (
    GapQuerySwap,
    active_gap_swap_mask,
    swap_active_gap_queries,
)
from gcnet_missing_m3.osram import OSRAMBackbone


def test_swap_only_exchanges_the_two_missing_query_slots():
    queries = torch.arange(3 * 1 * 4 * 1 * 2, dtype=torch.float32).reshape(3, 1, 4, 1, 2)
    availability = torch.tensor(
        [[[1, 0, 0]], [[0, 1, 0]], [[1, 1, 1]]], dtype=torch.float32
    )
    swapped, mask = swap_active_gap_queries(queries, availability)
    assert torch.equal(mask, torch.tensor([[True], [True], [False]]))
    assert torch.equal(swapped[0, 0, 0], queries[0, 0, 0])
    assert torch.equal(swapped[0, 0, 1], queries[0, 0, 1])
    assert torch.equal(swapped[0, 0, 2], queries[0, 0, 3])
    assert torch.equal(swapped[0, 0, 3], queries[0, 0, 2])
    assert torch.equal(swapped[1, 0, 1], queries[1, 0, 3])
    assert torch.equal(swapped[1, 0, 3], queries[1, 0, 1])
    assert torch.equal(swapped[2], queries[2])


def test_swap_context_preserves_every_post_write_memory_tensor():
    torch.manual_seed(11)
    model = OSRAMBackbone(
        latent_dim=4,
        output_dim=4,
        num_heads=2,
        key_dim=2,
        value_dim=2,
        dropout=0.0,
        bidirectional=False,
    ).eval()
    # The default flat readout is zero-initialized on its context branch, so
    # initialize that final projection here to make the read-only query effect
    # observable in the returned hidden state.
    torch.nn.init.normal_(model.emotion_adapter[-1].weight)
    node = torch.randn(4, 1, 4)
    latents = {name: torch.randn(4, 1, 4) for name in ("audio", "text", "visual")}
    availability = torch.tensor(
        [[[1, 0, 0]], [[0, 1, 0]], [[1, 1, 1]], [[0, 0, 1]]], dtype=torch.float32
    )
    qmask = torch.zeros(1, 4, dtype=torch.long)
    umask = torch.ones(1, 4)

    with GapQuerySwap(model, swap_queries=False) as reference:
        out_reference = model(node, latents, availability, qmask, umask, [4])[0]
    with GapQuerySwap(model, swap_queries=True) as swapped:
        out_swapped = model(node, latents, availability, qmask, umask, [4])[0]

    assert len(reference.memory_snapshots) == len(swapped.memory_snapshots) == 4
    for left, right in zip(reference.memory_snapshots, swapped.memory_snapshots):
        assert left[0] == right[0]
        assert torch.equal(left[1], right[1])
        assert torch.equal(left[2], right[2])
    assert not torch.equal(out_reference, out_swapped)
    assert "_project_sequence" not in model.__dict__
    assert "_scan" not in model.__dict__
