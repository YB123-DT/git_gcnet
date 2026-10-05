import unittest
import torch

from gcnet_missing_m3.core20_storage import build, DNCStorage, HiPPOStorage


def inputs(length=14):
    torch.manual_seed(21)
    keys = {m: torch.randn(length, 2, 2, 3, requires_grad=True) for m in ('audio', 'text', 'visual')}
    values = {m: torch.randn(length, 2, 2, 4, requires_grad=True) for m in keys}
    queries = torch.randn(length, 2, 4, 2, 3, requires_grad=True)
    available = torch.ones(length, 2, 3, dtype=torch.bool)
    available[::2, :, 0] = False
    valid = torch.ones(length, 2, dtype=torch.bool)
    valid[3, 0] = False
    return keys, values, queries, available, valid


def check_history_padding_mask_and_gradients(method):
    model = build(method, num_heads=2, key_dim=3, value_dim=4, latent_dim=16)
    args = inputs()
    base, gap, diagnostics = model.scan(*args)
    assert base.shape == (14, 2, 8) and gap.shape == (14, 2, 3, 8)
    assert torch.count_nonzero(base[0]) == 0
    assert torch.count_nonzero(gap[0]) == 0
    assert torch.count_nonzero(base[3, 0]) == 0
    assert torch.count_nonzero(gap[3, 0]) == 0
    dirty = tuple({m: x.clone() for m, x in group.items()} for group in args[:2])
    for group in dirty:
        for i, m in enumerate(group):
            group[m][~(args[3][:, :, i] & args[4])] = (float('nan'), float('inf'), -float('inf'))[i]
    changed = model.scan(*dirty, *args[2:])
    torch.testing.assert_close(base, changed[0])
    torch.testing.assert_close(gap, changed[1])
    # Modifying current/future observations cannot alter current/past reads.
    future = tuple({m: x.clone() for m, x in group.items()} for group in args[:2])
    for group in future:
        for x in group.values():
            x[7:] = 100 * torch.randn_like(x[7:])
    altered = model.scan(*future, *args[2:])
    torch.testing.assert_close(base[:8], altered[0][:8])
    torch.testing.assert_close(gap[:8], altered[1][:8])
    # Padding is a true identity transition, including each backend's clocks.
    single = tuple({m: x[:, :1] for m, x in g.items()} for g in args[:2])
    select = torch.arange(14) != 3
    compact = tuple({m: x[select] for m, x in g.items()} for g in single)
    out = model.scan(*compact, args[2][select, :1], args[3][select, :1], args[4][select, :1])
    torch.testing.assert_close(base[select, :1], out[0])
    assert set(diagnostics) == {'audio', 'text', 'visual'}
    base.square().mean().add(gap.square().mean()).add(model.auxiliary_loss).backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for group in args[:2] for x in group.values())
    for group in args[:2]:
        for i, m in enumerate(group):
            assert torch.count_nonzero(group[m].grad[~(args[3][:, :, i] & args[4])]) == 0
    assert args[2].grad is not None and torch.isfinite(args[2].grad).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    before = [p.detach().clone() for p in model.parameters()]
    torch.optim.SGD(model.parameters(), lr=0.01).step()
    assert any(not torch.equal(old, p) for old, p in zip(before, model.parameters()))


def test_dnc_allocation_freeing_and_temporal_order():
    usage = torch.tensor([[0.8, 0.1, 0.5]])
    allocation = DNCStorage.allocation(usage)
    assert allocation.argmax(-1).item() == 1
    freed = DNCStorage.usage_update(usage, torch.zeros_like(usage), torch.ones(1, 4), torch.tensor([[[0., 1., 0.]]]*4).transpose(0, 1))
    assert freed[0, 1] == 0
    link = torch.zeros(1, 3, 3)
    precedence = torch.zeros(1, 3)
    link, precedence = DNCStorage.link_update(link, precedence, torch.tensor([[1., 0., 0.]]))
    link, precedence = DNCStorage.link_update(link, precedence, torch.tensor([[0., 1., 0.]]))
    assert link[0, 1, 0] == 1 and link.diagonal(dim1=-2, dim2=-1).sum() == 0


def test_hippo_legs_order_one_matches_bilinear_integrator():
    model = HiPPOStorage(1, 1, 1, 8)
    model.order = 1
    a, b = model.discretize(torch.tensor([1., 2., 3.]), torch.float64, torch.device('cpu'))
    torch.testing.assert_close(a[:, 0, 0], torch.tensor([1/3, 3/5, 5/7], dtype=torch.float64))
    torch.testing.assert_close(b[:, 0], torch.tensor([2/3, 2/5, 2/7], dtype=torch.float64))


def test_compression_has_actual_auxiliary_supervision():
    model = build('C02', num_heads=2, key_dim=3, value_dim=4, latent_dim=16)
    args = inputs()
    model.scan(*args)
    assert model.compression_events > 0 and model.auxiliary_loss.item() > 0
    model.auxiliary_loss.backward()
    assert model.compressor.weight.grad is not None
    assert model.compressor.weight.grad.abs().sum() > 0
    # A singleton compressed token makes attention query-independent and gives
    # its key channels no compression-supervision gradient.
    assert model.compressor.weight.grad[:model.key_dim].abs().sum() > 0
    assert model.compressor.weight.grad[model.key_dim:].abs().sum() > 0
    old = torch.randn(4, 2, 7)
    compressed = model.compressor(old.permute(1, 2, 0)).permute(2, 0, 1)
    query = torch.randn(4, 2, 3)
    assert not torch.allclose(model.attend(query, compressed, positional=False),
                              model.attend(-query, compressed, positional=False))
    assert args[2].grad is None
    assert all(x.grad is None for group in args[:2] for x in group.values())


class StorageTests(unittest.TestCase):
    def test_common_contract(self):
        for method in ('C01', 'C02', 'C04'):
            with self.subTest(method=method):
                check_history_padding_mask_and_gradients(method)

    def test_dnc(self):
        test_dnc_allocation_freeing_and_temporal_order()

    def test_hippo(self):
        test_hippo_legs_order_one_matches_bilinear_integrator()

    def test_compression(self):
        test_compression_has_actual_auxiliary_supervision()

    def test_dnc_strength(self):
        memory = torch.tensor([[[1., 0.], [0., 1.]]])
        query = torch.tensor([[[1., 0.]]])
        actual = DNCStorage.content(memory, query, torch.zeros(1, 1))
        expected = torch.tensor([[[1. + torch.log(torch.tensor(2.)).item(), 0.]]]).softmax(-1)
        torch.testing.assert_close(actual, expected)


if __name__ == '__main__':
    unittest.main()
