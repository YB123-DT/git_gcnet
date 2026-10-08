"""Check unchanged default and fixed-width graph with enlarged internal MLPs."""
import torch
from gcnet_missing_m3.meaningful_input_new40 import build_new40
from gcnet_missing_m3.meaningful_new40_structure import NestedGNN
from gcnet_missing_m3.nested_sweep import NestedSweepGNN

torch.set_num_threads(1)
torch.manual_seed(66)
old = NestedGNN()
rng = torch.get_rng_state()
torch.manual_seed(66)
default = NestedSweepGNN()
assert torch.equal(rng, torch.get_rng_state())
assert all(torch.equal(v, default.state_dict()[k]) for k, v in old.state_dict().items())
model = build_new40('nested_mlp256')
assert model.num_heads == 8 and model.value_dim == 64
for mlp in [*model.core.layers, model.core.pool, model.core.readout]:
    assert mlp[0].out_features == 256 and mlp[2].out_features == 64
local = torch.randn(2, 256)
evidence = torch.randn(2, 4, 512)
active = torch.tensor([[1, 0, 1, 1], [0, 0, 0, 0]], dtype=torch.bool)
availability = torch.tensor([[1, 0, 0], [0, 0, 0]])
a, b = model(local, evidence, active, availability)
torch.testing.assert_close(a, local)
torch.testing.assert_close(b, torch.where(active[..., None], evidence, 0.))
optimizer = torch.optim.Adam(model.parameters(), lr=.001)
before = model.core.layers[0][0].weight.detach().clone()
for _ in range(2):
    optimizer.zero_grad()
    a, b = model(local, evidence, active, availability)
    (a.square().mean()+b.square().mean()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
assert not torch.equal(before, model.core.layers[0][0].weight)
clean = model(local, evidence, active, availability)
evidence[~active] = float('nan')
for a, b in zip(clean, model(local, evidence, active, availability)):
    torch.testing.assert_close(a, b)
assert torch.count_nonzero(clean[1][~active]) == 0
print('module_parameters', sum(p.numel() for p in model.parameters()))
print('PASS: old default/RNG unchanged, fixed graph width/head count, zero-init identity, masks, gradient/update')
