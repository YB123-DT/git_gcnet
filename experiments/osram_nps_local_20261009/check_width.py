"""Bounded CPU verification for the fixed NPS rule-width comparison."""
import torch
from gcnet_missing_m3.meaningful_input_new40 import build_new40

torch.set_num_threads(1)
torch.manual_seed(66)
old = build_new40('neural_production_local')
new = build_new40('neural_production_local_w256')
assert len(new.core.rule_mlps) == 4
assert all(rule[0].out_features == 256 for rule in new.core.rule_mlps)
assert sum(p.numel() for p in new.parameters()) == 315328
local = torch.randn(3, 256)
evidence = torch.randn(3, 4, 512)
active = torch.tensor([[1, 0, 1, 1], [1, 1, 0, 0], [0, 0, 0, 0]], dtype=torch.bool)
availability = torch.tensor([[1, 0, 0], [0, 1, 1], [0, 0, 0]])
for a, b in zip(old(local, evidence, active, availability), new(local, evidence, active, availability)):
    torch.testing.assert_close(a, b, rtol=0, atol=0)
optimizer = torch.optim.Adam(new.parameters(), lr=.001)
initial = [p.detach().clone() for p in new.core.rule_mlps.parameters()]
for _ in range(3):
    optimizer.zero_grad()
    a, b = new(local, evidence, active, availability)
    (a.square().mean() + b.square().mean()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in new.parameters())
    optimizer.step()
assert any(not torch.equal(a, b) for a, b in zip(initial, new.core.rule_mlps.parameters()))
clean = new(local, evidence, active, availability)
evidence[~active] = float('nan')
for a, b in zip(clean, new(local, evidence, active, availability)):
    torch.testing.assert_close(a, b)
assert torch.count_nonzero(clean[1][~active]) == 0
torch.testing.assert_close(clean[0][-1], local[-1], rtol=0, atol=0)
print('PASS: width=256, parameters=315328, zero-init parity, finite learning, safe masks')
