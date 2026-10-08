"""Bounded CPU checks for the single output-addition ablation."""
import torch
from gcnet_missing_m3.meaningful_input_new40 import build_new40

torch.set_num_threads(1)
torch.manual_seed(66)
residual = build_new40('nested_gnn_rooted_evidence')
rng = torch.get_rng_state()
torch.manual_seed(66)
direct = build_new40('nested_gnn_direct_evidence')
assert torch.equal(rng, torch.get_rng_state())
assert all(torch.equal(v, direct.state_dict()[k]) for k, v in residual.state_dict().items())
local = torch.randn(2, 256)
evidence = torch.randn(2, 4, 512)
active = torch.tensor([[1, 0, 1, 1], [0, 0, 0, 0]], dtype=torch.bool)
availability = torch.tensor([[1, 0, 0], [0, 0, 0]])
out = direct(local, evidence, active, availability)
assert all(torch.count_nonzero(x) == 0 for x in out)
with torch.no_grad():
    for layer in [direct.local_decoder, *direct.memory_decoders]:
        layer.weight.normal_(std=.01)
        layer.bias.fill_(.01)
residual.load_state_dict(direct.state_dict())
d = direct(local, evidence, active, availability)
r = residual(local, evidence, active, availability)
torch.testing.assert_close(r[0], local + d[0])
torch.testing.assert_close(r[1], torch.where(active[..., None], evidence + d[1], 0.))
dirty = evidence.clone()
dirty[~active] = float('nan')
clean = direct(local, dirty, active, availability)
for a, b in zip(d, clean):
    torch.testing.assert_close(a, b)
assert torch.count_nonzero(d[1][~active]) == 0
assert torch.count_nonzero(d[0][1]) == 0
optimizer = torch.optim.Adam(direct.parameters(), lr=.001)
before = direct.core.layers[0][0].weight.detach().clone()
sum(x.square().mean() for x in d).backward()
assert all(p.grad is None or torch.isfinite(p.grad).all() for p in direct.parameters())
optimizer.step()
assert not torch.equal(before, direct.core.layers[0][0].weight)
print('PASS: identical initialization/RNG/parameter count; exact addition ablation; masking; finite gradient and core update')
