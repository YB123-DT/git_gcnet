"""CPU checks: preserve legacy initialization, shrink only adapter, train safely."""
import importlib.util
import sys
import torch
from gcnet_missing_m3.osram import OSRAMBackbone
from gcnet_missing_m3.train_gcnet import TrainConfig

torch.set_num_threads(1)
assert 'osram_adapter_hidden_dim' in TrainConfig.__dataclass_fields__, 'missing width configuration'
spec = importlib.util.spec_from_file_location('gcnet_missing_m3._legacy_osram', sys.argv[1])
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
common = dict(latent_dim=256, output_dim=1600, num_heads=8, key_dim=64,
              value_dim=64, bidirectional=False, dropout=0., write_step=.6)
torch.manual_seed(66)
old = legacy.OSRAMBackbone(**common)
rng = torch.get_rng_state()
torch.manual_seed(66)
default = OSRAMBackbone(**common)
assert torch.equal(rng, torch.get_rng_state())
assert old.state_dict().keys() == default.state_dict().keys()
assert all(torch.equal(v, default.state_dict()[k]) for k, v in old.state_dict().items())
torch.manual_seed(66)
small = OSRAMBackbone(**common, osram_adapter_hidden_dim=256)
assert torch.equal(rng, torch.get_rng_state())
assert all(torch.equal(v, small.state_dict()[k]) for k, v in old.state_dict().items()
           if not k.startswith('emotion_adapter.'))
assert small.emotion_adapter[1].weight.shape == (256, 4352)
assert small.emotion_adapter[-1].weight.shape == (1600, 256)
assert sum(p.numel() for p in small.emotion_adapter.parameters()) == 1534272
node = torch.randn(3, 2, 256)
latents = {m: torch.randn_like(node) for m in ('audio', 'text', 'visual')}
av = torch.tensor([[[1,1,1],[1,0,1]], [[1,0,0],[0,1,0]], [[0,1,1],[0,0,0]]])
qm = torch.zeros(2, 3, dtype=torch.long)
um = torch.tensor([[1,1,1],[1,1,0]])
args = (node, latents, av, qm, um)
for module in (old, default, small): module.eval()
reference = old(*args)[0]
torch.testing.assert_close(reference, default(*args)[0], rtol=0, atol=0)
torch.testing.assert_close(reference, small(*args)[0], rtol=0, atol=0)
for method in ('none', 'conditional_new_07_neural_production', 'nested_gnn_rooted_evidence'):
    torch.manual_seed(66)
    net = OSRAMBackbone(**common, osram_adapter_hidden_dim=256, osram_meaningful_block=method)
    optimizer = torch.optim.Adam(net.parameters(), lr=.001)
    before = net.emotion_adapter[1].weight.detach().clone()
    branch_before = {k:p.detach().clone() for k,p in net.named_parameters() if k.startswith('meaningful_block.')}
    target = torch.randn(3,2,1600)
    for _ in range(3):
        optimizer.zero_grad()
        out = net(*args)[0]
        assert torch.count_nonzero(out[2,1]) == 0
        (out-target).square().mean().backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
        optimizer.step()
    assert not torch.equal(before, net.emotion_adapter[1].weight)
    if branch_before:
        assert any(not torch.equal(branch_before[k],p) for k,p in net.named_parameters() if k in branch_before)
    print('PASS training/padding:', method)
print('PASS legacy state/output/RNG, shared initialization and adapter count=1534272')
