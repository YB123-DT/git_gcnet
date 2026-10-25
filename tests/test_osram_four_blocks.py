from types import SimpleNamespace

import pytest
import torch

from osram_four_block_wiring.blocks import NeuMissBlock, SignatureBlock, RidgeResidualBlock, SupportBank, degree_two_signature


def frame(t=5):
    return SimpleNamespace(latents=torch.randn(t, 1, 3, 256), local=torch.randn(t, 1, 256),
                           context=torch.randn(t, 1, 2304), availability=torch.ones(t, 1, 3, dtype=torch.bool),
                           valid=torch.ones(t, 1, dtype=torch.bool))


def test_neumiss_masks_and_gradients():
    f = frame()
    f.availability[1, :, 1] = False
    f.availability[2] = False
    f.valid[3] = False
    block = NeuMissBlock()
    assert block.mu.count_nonzero() == 0
    assert torch.equal(block(f), torch.zeros(5, 1, 1600))
    torch.nn.init.normal_(block.out.weight)
    result = block(f)
    assert result[[0, 2, 3, 4]].count_nonzero() == 0
    changed = f.latents.clone()
    f.latents[1, :, 1] = 1e5
    torch.testing.assert_close(block(f), result)
    f.latents = changed
    block(f).sum().backward()
    assert block.steps[0].grad is not None


def test_signature_math():
    path = torch.tensor([[1., 0.], [1., 2.]])
    expected = torch.tensor([1., 2., .5, 2., 0., 2.])
    torch.testing.assert_close(degree_two_signature(path), expected)


def test_signature_causality_window_and_gradients():
    f = frame(12)
    block = SignatureBlock()
    baseline = block.descriptor(f)
    f.latents[10:] += 100
    torch.testing.assert_close(block.descriptor(f)[:11], baseline[:11])
    f.latents[0] += 100
    torch.testing.assert_close(block.descriptor(f)[10], baseline[10])
    f.availability[3:10, :, 0] = False
    desc = block.descriptor(f)
    assert desc[10, 0, -3] == 1
    torch.nn.init.normal_(block.out[-1].weight)
    out = block(f)
    assert out[0].count_nonzero() == 0
    out.sum().backward()
    assert block.slots[0][0].weight.grad is not None


def bank():
    return SupportBank(torch.randn(4, 1600), torch.tensor([[2.], [1e5], [1e5], [1e5]]),
                       torch.zeros(4, 1), ["other", "query", "other", "other"],
                       torch.tensor([1, 1, 0, 1]), torch.tensor([2, 2, 2, 0]), "parent")


def test_ridge_exclusions_empty_and_gradients():
    block = RidgeResidualBlock()
    q = torch.randn(1, 1600, requires_grad=True)
    support = bank()
    result = block(q, support, ["query"], 1, torch.tensor([2]), "parent")
    reduced = SupportBank(support.hidden[:1], support.labels[:1], support.base_prediction[:1],
                          ["other"], torch.tensor([1]), torch.tensor([2]), "parent")
    torch.testing.assert_close(result, block(q, reduced, ["query"], 1, torch.tensor([2]), "parent"))
    result.sum().backward()
    assert torch.isfinite(q.grad).all()
    assert block.raw_ridge.grad is not None
    assert block(q, support, ["query"], 4, torch.tensor([2]), "parent").count_nonzero() == 0
    support.split = "test"
    with pytest.raises(ValueError, match="training"):
        block(q, support, ["query"], 1, torch.tensor([2]), "parent")
    support.split = "train"
    with pytest.raises(ValueError, match="parent"):
        block(q, support, ["query"], 1, torch.tensor([2]), "wrong")


def test_empty_support_and_signature_padding():
    block = RidgeResidualBlock()
    empty = SupportBank(torch.empty(0, 1600), torch.empty(0, 1), torch.empty(0, 1), [],
                        torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), "p")
    out = block(torch.randn(2, 1600), empty, ["a", "b"], 0, torch.tensor([0, 0]), "p")
    torch.testing.assert_close(out, torch.zeros(2, 1))
    f = frame(4)
    f.valid[1] = False
    signature = SignatureBlock(window=2)
    baseline = signature.descriptor(f)
    f.latents[1] += 1e6
    torch.testing.assert_close(signature.descriptor(f)[2:], baseline[2:])
    torch.nn.init.normal_(signature.out[-1].weight)
    assert signature(f)[1].count_nonzero() == 0


def test_signature_descriptor_matches_loop_with_gaps():
    f = frame(13)
    f.valid[[1, 5, 9]] = False
    f.availability = torch.rand(13, 1, 3) > .4
    block = SignatureBlock(window=8)
    projected = [layer(f.latents[..., m, :]) for m, layer in enumerate(block.slots)]
    expected = []
    for t in range(13):
        indices = torch.where(f.valid[:t, 0])[0][-8:]
        times = (torch.arange(len(indices), dtype=f.local.dtype) + 8 - len(indices) + 1) / 8
        sig, counts = [], []
        for m in range(3):
            seen = f.availability[indices, 0, m]
            path = torch.cat([projected[m][indices, 0], times[:, None]], -1)[seen]
            sig.append(degree_two_signature(path))
            counts.append(seen.sum().to(f.local.dtype))
        expected.append(torch.cat([*sig, f.local[t, 0], f.availability[t, 0].to(f.local.dtype), torch.stack(counts)]))
    torch.testing.assert_close(block.descriptor(f)[:, 0], torch.stack(expected), atol=2e-6, rtol=2e-5)


def test_ridge_matches_independent_dual_solves():
    torch.manual_seed(18)
    block = RidgeResidualBlock(neighbors=3).double()
    support = SupportBank(torch.randn(7, 1600, dtype=torch.double), torch.randn(7, 1, dtype=torch.double),
                          torch.randn(7, 1, dtype=torch.double), ["a", "b", "a", "c", "b", "d", "b"],
                          torch.tensor([0, 0, 0, 0, 0, 1, 0]), torch.tensor([0, 0, 1, 0, 1, 0, 0]), "p")
    q = torch.randn(3, 1600, dtype=torch.double)
    patterns = torch.tensor([0, 1, 4])
    conversations = ["a", "b", "c"]
    qe, se = block.embedding(q), block.embedding(support.hidden)
    expected = []
    for n in range(3):
        ids = [i for i in range(7) if support.rate_ids[i] == 0 and support.pattern_ids[i] == patterns[n]
               and support.conversation_ids[i] != conversations[n]]
        if not ids:
            expected.append(q.new_zeros(1))
            continue
        ids = torch.tensor(ids)
        ids = ids[(se[ids] - qe[n]).square().sum(-1).argsort()[:3]]
        z = torch.nn.functional.pad(se[ids], (0, 1), value=1)
        qa = torch.nn.functional.pad(qe[n], (0, 1), value=1)
        lam = torch.nn.functional.softplus(block.raw_ridge) + 1e-4
        u = torch.linalg.solve(z @ z.T + lam * torch.eye(len(z), dtype=z.dtype),
                               support.labels[ids] - support.base_prediction[ids])
        expected.append(qa @ z.T @ u)
    torch.testing.assert_close(block(q, support, conversations, 0, patterns, "p"), torch.stack(expected), atol=1e-12, rtol=1e-10)
