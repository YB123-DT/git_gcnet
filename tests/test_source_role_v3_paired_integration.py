"""Check the three user-specified equations at their actual Flat interfaces."""
from dataclasses import replace

import pytest
import torch

from gcnet_missing_m3.osram import OSRAMBackbone
from gcnet_missing_m3.source_role_v3_paired import VARIANTS
from test_source_role_v3_connections_integration import _inputs


def make(variant='standard', enabled=True):
    return OSRAMBackbone(latent_dim=8, output_dim=10, num_heads=2,
        key_dim=3, value_dim=4, n_speakers=2, dropout=0., bidirectional=False,
        write_step=.6, osram_source_role_v3=enabled,
        osram_source_role_v3_variant=variant)


@pytest.mark.parametrize('a,b', [VARIANTS[:2], VARIANTS[2:4], VARIANTS[4:]])
def test_complete_pairs_have_identical_initial_parameters_rng_and_predictions(a,b):
    torch.manual_seed(66); left=make(a); rng=torch.get_rng_state()
    torch.manual_seed(66); right=make(b)
    assert torch.equal(rng,torch.get_rng_state())
    assert left.state_dict().keys()==right.state_dict().keys()
    assert all(torch.equal(v,right.state_dict()[k]) for k,v in left.state_dict().items())
    torch.manual_seed(66); flat=make(enabled=False)
    for k,v in flat.state_dict().items(): assert torch.equal(v,left.state_dict()[k]),k
    args=_inputs()
    x,xc=left(*args);y,yc=right(*args);z,zc=flat(*args)
    assert torch.equal(x,y) and torch.equal(y,z)
    for k in xc: assert torch.equal(xc[k],yc[k]) and torch.equal(xc[k],zc[k])


@pytest.mark.parametrize('variant', VARIANTS[2:])
def test_exact_local_and_task_equations_preserve_raw_reads_and_single_scan(variant):
    model=make(variant).eval()
    with torch.no_grad():
        model.source_role_v3.output_projection.weight.normal_(std=.02)
        model.source_role_v3.output_projection.bias.fill_(.1)
        model.emotion_adapter[-1].weight.normal_(std=.03)
    saved={};scans=[];scan=model._scan
    def counted(*a,**kw):scans.append(1);return scan(*a,**kw)
    model._scan=counted
    handle=model.source_role_v3.register_forward_hook(lambda _,a,v:saved.update(delta=v))
    args=_inputs();actual,ctx=model(*args);handle.remove()
    assert len(scans)==1
    local=ctx['local'];base=ctx['base'];gap=ctx['gap'];delta=saved['delta']
    current=local+delta if variant.startswith('local-') else local
    skip=current if variant=='local-shared' else local
    adapter=torch.cat((current,base,gap.flatten(2)),-1)
    anchor=model.local_skip(skip)+model.emotion_adapter(adapter)
    if variant=='task-pre-norm':expected=model.emotion_norm(anchor+delta)
    elif variant=='task-post-norm':expected=model.emotion_norm(anchor)+delta
    else:expected=model.emotion_norm(anchor)
    expected=torch.where(args[4].T.bool()[...,None],expected,torch.zeros_like(expected))
    assert torch.equal(actual,expected)
    assert not delta[0].any() and not delta[2,0].any() and not delta[3].any()
    # A single shared affine task head applies its bias exactly once.
    head=torch.nn.Linear(10,1)
    if variant=='task-post-norm':
        explicit=head(model.emotion_norm(anchor))+torch.nn.functional.linear(delta,head.weight)
        torch.testing.assert_close(head(actual)[args[4].T.bool()],explicit[args[4].T.bool()])


def test_first_step_local_projection_gradient_distinguishes_skip_routing():
    for variant,nonzero in [('local-adapter',False),('local-shared',True)]:
        torch.manual_seed(66);model=make(variant)
        hidden,_=model(*_inputs());hidden[...,0].sum().backward()
        grad=model.source_role_v3.output_projection.weight.grad
        assert grad is not None and torch.isfinite(grad).all()
        assert bool(grad.abs().sum()>0)==nonzero
        assert not model.source_role_v3.memory_projection.weight.grad.any()


def test_parser_and_factory_route_all_six_variants():
    from gcnet_missing_m3.train_gcnet import TrainConfig,build_parser
    from experiments.osram_meld_flat_source_combinations_20261013.evaluate import _build_model
    required=['--audio-feature','a','--text-feature','t','--video-feature','v','--output-dir','/tmp/unused-pairs']
    for variant in VARIANTS:
        args=build_parser().parse_args(required+['--osram-source-role-v3','--osram-source-role-v3-variant',variant])
        assert args.osram_source_role_v3_variant==variant
        cfg=replace(TrainConfig(),dataset='CMUMOSI',backbone_type='osram',training_objective='emotion-only',
            disable_unused_aux_modules=True,latent_dim=8,osram_output_dim=10,osram_num_heads=2,
            osram_key_dim=3,osram_value_dim=4,osram_bidirectional=False,
            osram_source_role_v3=True,osram_source_role_v3_variant=variant)
        model=_build_model(cfg,(4,5,6),torch.device('cpu'))
        assert model.osram.source_role_v3.variant==variant
