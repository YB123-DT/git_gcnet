"""Same Signature network and initialization, trained without oriented area."""
import torch

from osram_four_block_wiring import SignatureBlock


class NoAreaSignatureBlock(SignatureBlock):
    """Keep S1 and replace S2 by its symmetric part, half S1 outer S1.

    No parameters or RNG draws are added. The original network, masking,
    zero-initialized outlet and state-dict keys remain exactly the same.
    Input features lose directional degrees of freedom by design, even though
    the nominal trainable parameter count and optimization budget match.
    """

    def descriptor(self, frame):
        original = super().descriptor(frame)
        first = original[..., :270].reshape(*original.shape[:-1], 3, 90)[..., :9]
        symmetric = .5 * first[..., :, None] * first[..., None, :]
        signature = torch.cat((first, symmetric.flatten(-2)), dim=-1).flatten(-2)
        return torch.cat((signature, original[..., 270:]), dim=-1)
