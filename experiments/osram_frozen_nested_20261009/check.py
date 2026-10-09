"""Bounded CPU check using the real trained OSRAM weights, not a fresh Flat."""
import sys
import torch
from torch import nn
from gcnet_missing_m3.osram import OSRAMBackbone
from experiments.osram_frozen_nested_20261009.run import load_parent, freeze_parent, frozen_hash

torch.set_num_threads(1)
kwargs = dict(latent_dim=256, output_dim=1600, num_heads=8, key_dim=64,
              value_dim=64, bidirectional=False, dropout=.5, write_step=.6)
flat = nn.Module()
flat.osram = OSRAMBackbone(**kwargs)
model = nn.Module()
model.osram = OSRAMBackbone(**kwargs, osram_meaningful_block='nested_gnn_rooted_evidence')
state = torch.load(sys.argv[1], map_location='cpu', weights_only=False)['model']
state = {k: v for k, v in state.items() if k.startswith('osram.')}
flat.load_state_dict(state, strict=True)
load_parent(model, state)
params = freeze_parent(model)
assert sum(p.numel() for p in params) == 159235
assert not model.osram.training and model.osram.meaningful_block.training
assert not model.osram.emotion_adapter.training
baseline_hash = frozen_hash(model)
node = torch.randn(3, 2, 256)
latents = {m: torch.randn_like(node) for m in ('audio', 'text', 'visual')}
availability = torch.tensor([[[1,1,1],[1,0,1]], [[1,0,0],[0,1,0]], [[0,0,1],[0,0,0]]])
umask = torch.tensor([[1,1,1],[1,1,0]])
qmask = torch.zeros(2,3,dtype=torch.long)
args = (node, latents, availability, qmask, umask)
flat.eval()
model.eval()
with torch.no_grad():
    assert torch.equal(flat.osram(*args)[0], model.osram(*args)[0])
before = {k: v.clone() for k, v in model.osram.meaningful_block.state_dict().items()}
optimizer = torch.optim.Adam(params, lr=.001, weight_decay=1e-5)
model.train()
for _ in range(3):
    optimizer.zero_grad()
    hidden = model.osram(*args)[0]
    assert torch.count_nonzero(hidden[2,1]) == 0
    hidden.square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in params)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in params)
    optimizer.step()
    assert frozen_hash(model) == baseline_hash
assert any(not torch.equal(v, before[k]) for k, v in model.osram.meaningful_block.state_dict().items())
print('PASS: real-parent exact parity; Nested-only gradients/updates; frozen state; eval modes; padding')
