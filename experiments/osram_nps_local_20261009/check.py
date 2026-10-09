"""Only the added Local bridge differs; no extra rule execution or RNG draws."""
import torch
from gcnet_missing_m3.meaningful_input_new40 import build_new40

torch.set_num_threads(1)
torch.manual_seed(66)
old = build_new40('conditional_new_07_neural_production')
rng = torch.get_rng_state()
torch.manual_seed(66)
new = build_new40('neural_production_local')
assert torch.equal(rng, torch.get_rng_state())
assert all(torch.equal(v, new.state_dict()[k]) for k, v in old.state_dict().items())
assert set(new.state_dict()) - set(old.state_dict()) == {'local_bridge.weight', 'local_bridge.bias'}
local = torch.randn(2, 256)
evidence = torch.randn(2, 4, 512)
active = torch.tensor([[1, 0, 1, 1], [0, 0, 0, 0]], dtype=torch.bool)
availability = torch.tensor([[1, 0, 0], [0, 0, 0]])
for a, b in zip(old(local, evidence, active, availability), new(local, evidence, active, availability)):
    torch.testing.assert_close(a, b, rtol=0, atol=0)
optimizer = torch.optim.Adam(new.parameters(), lr=.001)
for _ in range(3):
    optimizer.zero_grad()
    a, b = new(local, evidence, active, availability)
    (a.square().mean()+b.square().mean()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in new.parameters())
    optimizer.step()
assert new.local_bridge.weight.abs().sum() > 0
clean = new(local, evidence, active, availability)
assert not torch.equal(clean[0][0], local[0])
torch.testing.assert_close(clean[0][1], local[1])
evidence[~active] = float('nan')
for a, b in zip(clean, new(local, evidence, active, availability)):
    torch.testing.assert_close(a, b)
assert torch.count_nonzero(clean[1][~active]) == 0
print('parameters', sum(p.numel() for p in old.parameters()), sum(p.numel() for p in new.parameters()))
print('PASS: exact initialization/RNG parity, Local bridge learns, finite gradients and safe masks')
