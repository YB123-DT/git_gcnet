"""Check the new composition; reuse existing tests of the unchanged graph core."""
import json
import sys
import torch
from experiments.osram_core20_20261005.run import candidate_config
from gcnet_missing_m3.osram import OSRAMBackbone

torch.set_num_threads(1)
reference = json.load(open(sys.argv[1]))
config, delta = candidate_config(reference, 'small_flat_nested_mlp256')
assert set(delta) == {'osram_adapter_hidden_dim', 'osram_meaningful_block'}
assert config.osram_adapter_hidden_dim == 256 and config.osram_output_dim == 1600
assert config.osram_meaningful_block == 'nested_mlp256'
net = OSRAMBackbone(latent_dim=256, output_dim=1600, num_heads=8,
    key_dim=64, value_dim=64, bidirectional=False, dropout=0., write_step=.6,
    osram_adapter_hidden_dim=256, osram_meaningful_block='nested_mlp256')
assert sum(p.numel() for p in net.emotion_adapter.parameters()) == 1534272
assert sum(p.numel() for p in net.meaningful_block.parameters()) == 332227
node = torch.randn(2, 2, 256)
latents = {m: torch.randn_like(node) for m in ('audio', 'text', 'visual')}
av = torch.tensor([[[1,1,1],[1,0,1]], [[1,0,0],[0,0,0]]])
qm = torch.zeros(2,2, dtype=torch.long)
um = torch.tensor([[1,1],[1,0]])
optimizer = torch.optim.Adam(net.parameters(), lr=.001)
before = net.meaningful_block.core.core.layers[0][0].weight.detach().clone()
target = torch.randn(2,2,1600)
for _ in range(3):
    optimizer.zero_grad()
    out = net(node, latents, av, qm, um)[0]
    assert out.shape == (2,2,1600) and torch.count_nonzero(out[1,1]) == 0
    (out-target).square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
    optimizer.step()
assert not torch.equal(before, net.meaningful_block.core.core.layers[0][0].weight)
print('PASS small adapter + expanded Nested: config, shapes, finite updates, padding')
