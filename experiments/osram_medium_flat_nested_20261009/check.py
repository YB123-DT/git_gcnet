"""Only adapter width varies; original Nested and the outer initialization stay fixed."""
import json
import sys
import torch
from gcnet_missing_m3.osram import OSRAMBackbone
from experiments.osram_core20_20261005.run import candidate_config

torch.set_num_threads(1)
reference = json.load(open(sys.argv[1]))
common = dict(latent_dim=256, output_dim=1600, num_heads=8, key_dim=64,
              value_dim=64, bidirectional=False, dropout=0., write_step=.6,
              osram_meaningful_block='nested_gnn_rooted_evidence')
torch.manual_seed(66)
anchor = OSRAMBackbone(**common, osram_adapter_hidden_dim=256)
anchor_rng = torch.get_rng_state()
node = torch.randn(3,2,256)
latents = {m: torch.randn_like(node) for m in ('audio','text','visual')}
av = torch.tensor([[[1,1,1],[1,0,1]],[[1,0,0],[0,1,0]],[[0,0,1],[0,0,0]]])
args = (node,latents,av,torch.zeros(2,3,dtype=torch.long),torch.tensor([[1,1,1],[1,1,0]]))
for width in (384,512,768,1024,1280):
    plain, plain_delta = candidate_config(reference, f'flat{width}')
    assert set(plain_delta) == {'osram_adapter_hidden_dim'}
    assert plain.osram_adapter_hidden_dim == width and plain.osram_meaningful_block == 'none'
    config, delta = candidate_config(reference, f'flat{width}_nested')
    assert set(delta) == {'osram_adapter_hidden_dim','osram_meaningful_block'}
    assert config.osram_adapter_hidden_dim == width
    assert config.osram_meaningful_block == 'nested_gnn_rooted_evidence'
    torch.manual_seed(66)
    net = OSRAMBackbone(**common, osram_adapter_hidden_dim=width)
    assert torch.equal(torch.get_rng_state(),anchor_rng)
    assert all(torch.equal(v,net.state_dict()[k]) for k,v in anchor.state_dict().items()
               if not k.startswith('emotion_adapter.'))
    assert sum(p.numel() for p in net.meaningful_block.parameters()) == 159235
    assert sum(p.numel() for p in net.emotion_adapter.parameters()) == 5953*width+10304
    assert net.emotion_adapter[1].weight.shape == (width,4352)
    assert net.emotion_adapter[-1].weight.shape == (1600,width)
    target = torch.randn(3,2,1600)
    before = {k:p.detach().clone() for k,p in net.named_parameters()}
    optimizer = torch.optim.Adam(net.parameters(),lr=.001)
    for _ in range(3):
        optimizer.zero_grad()
        hidden = net(*args)[0]
        assert torch.count_nonzero(hidden[2,1]) == 0
        (hidden-target).square().mean().backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
        optimizer.step()
    for prefix in ('emotion_adapter.','meaningful_block.'):
        assert any(not torch.equal(before[k],p) for k,p in net.named_parameters() if k.startswith(prefix))
    print('PASS',width,'config/parameters/shared initialization/padding/finite updates',flush=True)
for method in ('small_flat','small_flat_nested','small_flat_nested_mlp256','small_flat_nested_mlp512'):
    assert candidate_config(reference,method)[0].osram_adapter_hidden_dim == 256
