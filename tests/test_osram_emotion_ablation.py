"""Classification-only masking must not ablate the auxiliary predictor."""
from unittest.mock import patch
from dataclasses import asdict
import json

import pytest
import torch

from gcnet_missing_m3.osram import OSRAMBackbone
from gcnet_missing_m3.model import MissingM3GraphModel
from gcnet_missing_m3.train_gcnet import TrainConfig, build_parser


def inputs():
    torch.manual_seed(7)
    node = torch.randn(4, 2, 8)
    latents = {m: torch.randn_like(node) for m in ('audio', 'text', 'visual')}
    mask = torch.tensor([[[1,0,1],[0,1,1]], [[0,1,0],[1,0,0]],
                         [[1,1,1],[1,0,1]], [[0,0,0],[0,0,0]]]).float()
    return node, latents, mask, torch.zeros(2,4).long(), torch.tensor([[1,1,1,0]]*2).float(), [3,3]


def backbone(**kw):
    return OSRAMBackbone(latent_dim=8, output_dim=10, num_heads=2,
                         key_dim=3, value_dim=4, n_speakers=1,
                         dropout=0, bidirectional=False, write_step=.6, **kw)


@pytest.mark.parametrize('mode', ['full','local-only','local-base','local-gap'])
def test_emotion_masks_preserve_scan_contexts_and_initialization(mode):
    torch.manual_seed(12)
    ref = backbone().eval()
    rng = torch.get_rng_state().clone()
    torch.manual_seed(12)
    model = backbone(osram_emotion_ablation=mode).eval()
    assert torch.equal(rng, torch.get_rng_state())
    assert list(ref.state_dict()) == list(model.state_dict())
    for k, v in ref.state_dict().items():
        assert torch.equal(v, model.state_dict()[k])
    torch.nn.init.normal_(ref.emotion_adapter[-1].weight, std=.1)
    model.load_state_dict(ref.state_dict(), strict=True)
    args = inputs()

    def trace(m):
        states, fusion = [], []
        original = m.block_write
        def write(*a, **k):
            result = original(*a, **k)
            states.append(result.detach().clone())
            return result
        handle = m.emotion_adapter.register_forward_pre_hook(lambda _, a: fusion.append(a[0].detach().clone()))
        with patch.object(m, 'block_write', side_effect=write):
            out = m(*args)
        handle.remove()
        return out, states, fusion[0]

    (rh, rc), rs, rf = trace(ref)
    (h, c), states, fused = trace(model)
    assert len(rs) == len(states) == 3
    for a,b in zip(rs,states):
        assert torch.equal(a,b)
    for name in ('base','gap','local'):
        assert torch.equal(rc[name],c[name])
    expected = rf.clone()
    if mode in ('local-only','local-gap'):
        expected[...,8:24] = 0
    if mode in ('local-only','local-base'):
        expected[...,24:] = 0
    assert torch.equal(fused, expected)
    assert torch.equal(h, rh) == (mode == 'full')
    assert not h[3].count_nonzero()
    h.square().sum().backward()
    assert model.local_skip.weight.grad is not None
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_emotion_config_cli_and_exclusivity():
    assert TrainConfig().osram_emotion_ablation == 'full'
    assert build_parser().parse_args(['--osram-emotion-ablation','local-gap',
        '--audio-feature','a','--text-feature','t','--video-feature','v',
        '--output-dir','unused']).osram_emotion_ablation == 'local-gap'
    with pytest.raises(ValueError):
        backbone(osram_ablation='local-base', osram_emotion_ablation='local-gap')
    config = TrainConfig(osram_emotion_ablation='local-gap')
    assert TrainConfig(**json.loads(json.dumps(asdict(config)))).osram_emotion_ablation == 'local-gap'
    old = asdict(config)
    del old['osram_emotion_ablation']
    assert TrainConfig(**old).osram_emotion_ablation == 'full'


@pytest.mark.parametrize('mode', ['local-only','local-base','local-gap'])
def test_structured_prediction_unchanged_by_emotion_ablation(mode):
    model = MissingM3GraphModel('LSTM', 2,3,4, 8,4, n_speakers=1, n_classes=1,
        window_past=2, window_future=2,
        dropout=0, no_cuda=True, latent_dim=8, projector_dropout=0, predictor_dropout=0,
        backbone_type='osram', osram_output_dim=10, osram_num_heads=2,
        osram_key_dim=3, osram_value_dim=4, osram_bidirectional=False,
        osram_write_step=.6, osram_emotion_ablation=mode).eval()
    _,_,mask,q,u,lengths=inputs()
    x=torch.randn(4,2,9)
    torch.nn.init.normal_(model.osram.emotion_adapter[-1].weight, std=.1)
    changed = model([x],mask,q,u,lengths,predict_missing=True)
    model.osram.osram_emotion_ablation='full'
    full = model([x],mask,q,u,lengths,predict_missing=True)
    assert not torch.equal(changed[1], full[1])
    for name in ('reg_predictions','cl_predictions','target_mask','source_counts'):
        assert torch.equal(getattr(changed[3],name),getattr(full[3],name))
