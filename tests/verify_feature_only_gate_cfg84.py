"""Independent actual-dimension cfg84 identity/update check (not a benchmark)."""
import json
from dataclasses import fields, replace
from pathlib import Path

import torch

from gcnet_missing_m3.train_gcnet import TrainConfig
from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model


def main():
    source = Path('/data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full/seed_66/config.json')
    raw = json.loads(source.read_text())
    config = TrainConfig(**{k: v for k, v in raw.items() if k in {f.name for f in fields(TrainConfig)}})
    assert not config.osram_history_input_gate
    assert config.osram_readout_fusion == 'flat'
    assert config.training_objective == 'emotion-only'
    baseline = _build_model(config, (512, 1024, 1024)).cuda()
    gated = _build_model(replace(config, osram_hierarchical_evidence_gate=True, osram_hierarchical_feature_only=True), (512, 1024, 1024)).cuda()
    assert gated.osram.output_dim == 1600
    assert gated.osram.num_heads * gated.osram.value_dim == 512
    assert gated.osram.context_dim == 1024  # Existing forward/backward interface slots.
    assert gated.osram.hierarchical_evidence_gate.feature_output.out_features == gated.osram.context_dim
    for name, value in baseline.state_dict().items():
        assert torch.equal(value, gated.state_dict()[name]), name
    assert all(p.requires_grad for n, p in gated.osram.named_parameters() if not n.startswith('hierarchical_evidence_gate.'))
    torch.manual_seed(917)
    x = torch.randn(4, 2, 2560, device='cuda')
    availability = torch.tensor([[[1., 0., 1.], [0., 1., 0.]]] * 4, device='cuda')
    umask = torch.tensor([[1., 1., 1., 1.], [1., 1., 0., 0.]], device='cuda')
    availability[~umask.T.bool()] = 0
    qmask = torch.zeros(2, 4, dtype=torch.long, device='cuda')
    args = ([x], availability, qmask, umask, [4, 2])
    checks = {}
    for training in (False, True):
        baseline.train(training)
        gated.train(training)
        torch.manual_seed(881)
        with torch.no_grad():
            expected = baseline(*args, predict_missing=False)[0]
        rng = torch.cuda.get_rng_state()
        torch.manual_seed(881)
        with torch.no_grad():
            actual = gated(*args, predict_missing=False)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.equal(rng, torch.cuda.get_rng_state())
        checks['train' if training else 'eval'] = {'max_abs': (actual-expected).abs().max().item(), 'rng_equal': True}
    del baseline
    optimizer = torch.optim.Adam(gated.parameters(), lr=config.learning_rate)
    target = torch.randn_like(actual)
    changed = {k: p.detach().clone() for k, p in gated.osram.named_parameters()
               if k in ('local_skip.weight', 'emotion_adapter.4.weight', 'query_projector.weight',
                        'hierarchical_evidence_gate.feature_output.weight')}
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        prediction = gated(*args, predict_missing=False)[0]
        # No additional gate penalty: joint task gradients through both levels.
        (prediction-target).square().mean().backward()
        assert all(torch.isfinite(p.grad).all() for p in gated.parameters() if p.grad is not None)
        optimizer.step()
    for key, before in changed.items():
        assert not torch.equal(before, dict(gated.osram.named_parameters())[key]), key
    print(json.dumps({'source_config': str(source), 'dimensions': [512,1024,1024],
                      'output_dim': 1600, 'context_dim': gated.osram.context_dim,
                      'forward_slot_reuse': gated.osram.forward_slot_reuse,
                      'checks': checks, 'finite_update_steps': 3,
                      'original_parameters_updated': list(changed)}, indent=2))


if __name__ == '__main__':
    main()
