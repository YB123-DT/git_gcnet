"""Check graph grouping, external dimensions, masks and actual updates."""
import torch
from gcnet_missing_m3.meaningful_input_new40 import build_new40

torch.set_num_threads(1)
torch.manual_seed(66)
model = build_new40('nested_groups1_dim512')
assert model.num_heads == 1 and model.value_dim == 512
assert model.tokenizer.local.in_features == 256
assert model.tokenizer.local.out_features == 512
assert model.tokenizer.memory[0].in_features == 512
assert model.tokenizer.memory[0].out_features == 512
local = torch.randn(2, 256)
evidence = torch.randn(2, 4, 512)
active = torch.tensor([[1, 0, 1, 1], [0, 0, 0, 0]], dtype=torch.bool)
availability = torch.tensor([[1, 0, 0], [0, 0, 0]])
tokens, mask = model.tokenizer(local, evidence, active)
assert tokens.shape == (2, 5, 512) and mask[0].sum() == 4
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
dirty = model(local, evidence, active, availability)
for a, b in zip(clean, dirty):
    torch.testing.assert_close(a, b)
assert torch.count_nonzero(dirty[1][~active]) == 0
for method in ('nested_gnn_rooted_evidence', 'nested_groups1', 'nested_groups1_dim512'):
    module = build_new40(method)
    print(method, sum(p.numel() for p in module.parameters()))
print('PASS: 1x512 graph grouping, 512d memory IO, zero-init identity, safe masks, finite gradients and core update')
