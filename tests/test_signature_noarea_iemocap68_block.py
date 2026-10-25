from types import SimpleNamespace

import torch


def frame():
    torch.manual_seed(42)
    return SimpleNamespace(latents=torch.randn(7,2,3,256),local=torch.randn(7,2,256),
        availability=torch.ones(7,2,3,dtype=torch.bool),valid=torch.ones(7,2,dtype=torch.bool))


def test_state_parameters_and_random_initialization_match_original():
    from osram_four_block_wiring import SignatureBlock
    from experiments.osram_signature_noarea_iemocap68_20261025.block import NoAreaSignatureBlock
    torch.manual_seed(68);original=SignatureBlock()
    torch.manual_seed(68);control=NoAreaSignatureBlock()
    assert sum(p.numel() for p in control.parameters())==281856
    assert original.state_dict().keys()==control.state_dict().keys()
    for key,value in original.state_dict().items():
        assert torch.equal(value,control.state_dict()[key])


def test_descriptor_keeps_endpoint_and_metadata_but_only_symmetric_second_level():
    from osram_four_block_wiring import SignatureBlock
    from experiments.osram_signature_noarea_iemocap68_20261025.block import NoAreaSignatureBlock
    x=frame();original=SignatureBlock();control=NoAreaSignatureBlock()
    control.load_state_dict(original.state_dict())
    full=original.descriptor(x);noarea=control.descriptor(x)
    first=full[...,:270].reshape(7,2,3,90)[...,:9]
    expected=.5*first[..., :,None]*first[...,None,:]
    got=noarea[...,:270].reshape(7,2,3,90)
    torch.testing.assert_close(got[...,:9],first,rtol=0,atol=0)
    torch.testing.assert_close(got[...,9:].reshape(7,2,3,9,9),expected,rtol=0,atol=0)
    torch.testing.assert_close(noarea[...,270:],full[...,270:],rtol=0,atol=0)
    assert not torch.allclose(full,noarea)


def test_forward_initial_zero_and_input_projectors_can_learn_after_outlet_opens():
    from experiments.osram_signature_noarea_iemocap68_20261025.block import NoAreaSignatureBlock
    # Torch1.13 CPU MKLDNN intermittently produces nonfinite backward gradients in
    # the ORIGINAL Signature as well; the plain CPU backend is stable. This
    # scoped test setting is restored and never applied to GPU training.
    with torch.backends.mkldnn.flags(enabled=False):
        torch.manual_seed(68)
        block=NoAreaSignatureBlock();x=frame()
        assert torch.count_nonzero(block(x))==0
        with torch.no_grad():block.out[-1].weight.normal_(0,.01)
        block(x).square().mean().backward()
        grads=[p.grad for p in block.slots.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert any(torch.count_nonzero(g)>0 for g in grads)
        assert torch.count_nonzero(block(x)[0])==0
