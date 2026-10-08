"""Verify decoder-only random initialization, masking and first-step gradients."""
import torch
from gcnet_missing_m3.meaningful_input_new40 import build_new40

torch.set_num_threads(1)
torch.manual_seed(66)
zero = build_new40('nested_gnn_direct_evidence')
rng = torch.get_rng_state()
torch.manual_seed(66)
random = build_new40('nested_gnn_direct_random_evidence')
assert torch.equal(rng, torch.get_rng_state())
assert zero.state_dict().keys() == random.state_dict().keys()
for name, parameter in zero.state_dict().items():
    if name.startswith(('local_decoder.', 'memory_decoders.')):
        assert torch.count_nonzero(parameter) == 0
        assert torch.count_nonzero(random.state_dict()[name]) > 0
    else:
        assert torch.equal(parameter, random.state_dict()[name]), name
local = torch.randn(2, 256)
evidence = torch.randn(2, 4, 512)
active = torch.tensor([[1, 0, 1, 1], [0, 0, 0, 0]], dtype=torch.bool)
availability = torch.tensor([[1, 0, 0], [0, 0, 0]])
out = random(local, evidence, active, availability)
assert torch.count_nonzero(out[0][0]) > 0 and torch.count_nonzero(out[1][0, 0]) > 0
assert torch.count_nonzero(out[0][1]) == 0 and torch.count_nonzero(out[1][~active]) == 0
evidence[~active] = float('nan')
for a, b in zip(out, random(local, evidence, active, availability)):
    torch.testing.assert_close(a, b)
optimizer = torch.optim.Adam(random.parameters(), lr=.001)
before = random.core.layers[0][0].weight.detach().clone()
sum(x.square().mean() for x in out).backward()
assert all(p.grad is None or torch.isfinite(p.grad).all() for p in random.parameters())
assert random.core.layers[0][0].weight.grad.abs().sum() > 0
optimizer.step()
assert not torch.equal(before, random.core.layers[0][0].weight)
print('PASS: decoder-only initialization difference, unchanged RNG, nonzero outputs, safe masks, first-step core gradient/update')
