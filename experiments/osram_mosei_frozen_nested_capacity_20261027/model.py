"""One approved architecture factor changes inside the frozen Nested wrapper."""
from torch import nn

from gcnet_missing_m3.meaningful_input import MeaningfulInputAdapter
from experiments.osram_frozen_nested_gcnet_20261025.model import FrozenNestedModel
from .protocol import CONFIGS


class VariantFrozenNestedModel(FrozenNestedModel):
    def __init__(self, parent, config):
        # Deliberately do not call FrozenNestedModel.__init__: constructing and
        # replacing a default adapter would consume an extra set of RNG draws.
        nn.Module.__init__(self)
        if config not in CONFIGS:
            raise ValueError('Only the three approved Nested capacity configurations are allowed')
        if not isinstance(parent.smax_fc, nn.Linear):
            raise ValueError('requires original deterministic Flat linear task head')
        self.parent = parent.requires_grad_(False).eval()
        self.config = config
        self.block = MeaningfulInputAdapter(256, 1024, 1600, CONFIGS[config]['method'], 8, 64)
        expected = CONFIGS[config]
        adapter = self.block.core
        core = adapter.core
        actual = dict(dim=adapter.local_decoder.in_features, depth=len(core.layers),
            mlp_hidden=core.layers[0][0].out_features, groups=adapter.num_heads)
        if actual != {key: expected[key] for key in actual}:
            raise ValueError('Registered Nested implementation differs from approved architecture')
