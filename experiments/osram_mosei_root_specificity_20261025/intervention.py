"""Frozen readout-input intervention, without editing the trained GNN."""
from contextlib import contextmanager

import torch


def common_root_input(inputs):
    """Replace [x_i,r_i,r_bar] by [x_i,r_bar,r_bar], preserving other chunks."""
    if inputs.ndim != 3 or inputs.shape[-1] % 3 or inputs.shape[1] < 1:
        raise ValueError('Expected nonempty [batch,nodes,3*dim] readout input')
    dim = inputs.shape[-1] // 3
    x, _root, public = inputs.split(dim, dim=-1)
    return torch.cat((x, public, public), dim=-1)


@contextmanager
def suppress_root_specificity(core):
    """Keep extraction, GIN, pool and global mean; change only final MLP input."""
    stats = dict(calls=0, max_removed_root_difference=0.)

    def pre_hook(_module, args):
        if len(args) != 1:
            raise ValueError('Readout must have exactly one packed input')
        original = args[0]
        dim = original.shape[-1] // 3
        root = original[..., dim:2 * dim]
        public = original[..., 2 * dim:]
        if not torch.equal(root.mean(1, keepdim=True).expand_as(root), public):
            raise RuntimeError('Third chunk is not the original common rooted mean')
        replacement = common_root_input(original)
        if not torch.equal(replacement[..., :dim], original[..., :dim]):
            raise RuntimeError('Original node changed')
        if not torch.equal(replacement[..., 2 * dim:], public):
            raise RuntimeError('Public graph information changed')
        stats['calls'] += 1
        stats['max_removed_root_difference'] = max(
            stats['max_removed_root_difference'], float((root - public).abs().max()))
        return (replacement,)

    handle = core.readout.register_forward_pre_hook(pre_hook)
    try:
        yield stats
    finally:
        handle.remove()
