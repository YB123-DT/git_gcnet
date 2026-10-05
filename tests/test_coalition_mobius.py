"""Exact observed-set coalitions, masking and shared stochastic fusion."""
import importlib.util

import pytest
import torch
from torch import nn


NAMES = ('audio', 'text', 'visual')
SETS = {'A': (0,), 'T': (1,), 'V': (2,), 'AT': (0, 1),
        'AV': (0, 2), 'TV': (1, 2), 'ATV': (0, 1, 2)}


class Encoder(nn.Module):
    """Faithful lightweight mean encoder, without the graph dependencies."""
    def __init__(self, dim=5, dropout=0.0, additive=False):
        super().__init__()
        self.dimensions = (2, 3, 4)
        self.latent_dim = dim
        self.fusion_type = 'mean'
        self.projectors = nn.ModuleDict({name: nn.Sequential(nn.Linear(width, dim),
            nn.Dropout(dropout)) for name, width in zip(NAMES, self.dimensions)})
        self.modality_embedding = nn.Embedding(3, dim)
        self.pattern_embedding = nn.Embedding(8, dim, padding_idx=0)
        self.fusion = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim),
                                    nn.GELU(), nn.Dropout(dropout))
        if additive:
            # Pattern encoding encodes an additive set function; feature evidence is zero.
            with torch.no_grad():
                self.modality_embedding.weight.zero_()
                for projector in self.projectors.values():
                    projector[0].weight.zero_()
                    projector[0].bias.zero_()
                singles = torch.randn(3, dim)
                for pattern in range(8):
                    self.pattern_embedding.weight[pattern] = sum(
                        (singles[i] for i, bit in enumerate((4, 2, 1)) if pattern & bit),
                        torch.zeros(dim))
            for i in range(3):
                self.fusion[i] = nn.Identity()

    def _validate(self, features, availability, umask):
        return umask.T.bool()

    def forward(self, features, availability, umask):
        valid = self._validate(features, availability, umask)
        evidence = features.new_zeros((*features.shape[:2], self.latent_dim))
        latents, start = {}, 0
        for i, (name, width) in enumerate(zip(NAMES, self.dimensions)):
            selected = valid & availability[..., i].bool()
            latent = torch.zeros_like(evidence)
            latent[selected] = self.projectors[name](features[..., start:start + width][selected])
            evidence[selected] += latent[selected] + self.modality_embedding.weight[i]
            latents[name] = latent
            start += width
        pattern = (availability.long() * availability.new_tensor([4, 2, 1], dtype=torch.long)).sum(-1)
        inputs = evidence / availability.sum(-1, keepdim=True).clamp_min(1)
        inputs = inputs + self.pattern_embedding(pattern)
        node = torch.zeros_like(evidence)
        node[valid] = self.fusion(inputs[valid])
        return node, latents


def inputs():
    availability = torch.tensor([[[1, 0, 0], [0, 1, 0]],
        [[0, 0, 1], [1, 1, 0]], [[1, 0, 1], [0, 1, 1]],
        [[1, 1, 1], [0, 0, 0]]], dtype=torch.float32)
    return torch.randn(4, 2, 9), availability, torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])


def test_feature_available():
    assert importlib.util.find_spec('gcnet_missing_m3.coalition_mobius') is not None


def test_known_additive_and_third_order_algebra():
    from gcnet_missing_m3.coalition_mobius import mobius_transform
    singles = torch.randn(3, 2, 5)
    third = torch.randn(2, 5)
    values = {key: sum((singles[i] for i in indices), torch.zeros_like(third))
              for key, indices in SETS.items()}
    for key in ('AT', 'AV', 'TV', 'ATV'):
        torch.testing.assert_close(mobius_transform(values)[key], torch.zeros_like(third), atol=1e-6, rtol=0)
    values['ATV'] = values['ATV'] + third
    torch.testing.assert_close(mobius_transform(values)['ATV'], third)


def test_coalitions_match_encoder_and_reconstruct_only_observed_set():
    from gcnet_missing_m3.coalition_mobius import CoalitionMobiusBlock
    torch.manual_seed(10)
    encoder, block = Encoder().eval(), CoalitionMobiusBlock(5).eval()
    features, availability, umask = inputs()
    calls = {name: [] for name in NAMES}
    handles = [encoder.projectors[name].register_forward_hook(
        lambda module, args, result, name=name: calls[name].append(len(args[0]))) for name in NAMES]
    result = block.decompose(encoder, features, availability, umask)
    for handle in handles:
        handle.remove()
    assert all(len(counts) == 1 for counts in calls.values())
    for key, indices in SETS.items():
        eligible = result['eligible'][key]
        subset = torch.zeros_like(availability)
        subset[..., list(indices)] = 1
        subset[~eligible] = 0
        expected, _ = encoder(features, subset, eligible.T)
        torch.testing.assert_close(result['coalitions'][key], expected)
    expected, latents = encoder(features, availability, umask)
    torch.testing.assert_close(sum(result['dividends'].values()), expected)
    for name in NAMES:
        torch.testing.assert_close(result['latents'][name], latents[name])
    assert not any('projector' in name or 'embedding' in name for name, _ in block.named_parameters())


def test_missing_nan_bias_padding_and_finite_gradients():
    from gcnet_missing_m3.coalition_mobius import CoalitionMobiusBlock
    encoder, block = Encoder(), CoalitionMobiusBlock(5)
    features, availability, umask = inputs()
    start = 0
    for i, width in enumerate(encoder.dimensions):
        features[..., start:start + width][~availability[..., i].bool()] = float('nan')
        start += width
    for projector in (block.p1, block.p2, block.p3):
        nn.init.constant_(projector[1].bias, 3)
    node, latents = block(encoder, features, availability, umask)
    assert torch.isfinite(node).all()
    assert torch.count_nonzero(node[~umask.T.bool()]) == 0
    slots = block.last_order_slots
    for order in (1, 2, 3):
        inactive = availability.sum(-1) < order
        assert torch.count_nonzero(slots[..., order - 1, :][inactive]) == 0
    assert all(torch.isfinite(value).all() for value in latents.values())
    node.square().sum().backward()
    for parameter in list(encoder.parameters()) + list(block.parameters()):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert block.diagnostics['coalition_count_mean'] == pytest.approx(19 / 7)
    assert block.diagnostics['reconstruction_max_error'] < 1e-5


def test_shared_dropout_preserves_additivity_and_rng_state():
    from gcnet_missing_m3.coalition_mobius import CoalitionMobiusBlock
    encoder, block = Encoder(dropout=.35, additive=True).train(), CoalitionMobiusBlock(5)
    features, availability, umask = inputs()
    state = torch.get_rng_state()
    expected, _ = encoder(features, availability, umask)
    expected_rng = torch.get_rng_state()
    torch.set_rng_state(state)
    result = block.decompose(encoder, features, availability, umask)
    torch.testing.assert_close(result['full_node'], expected)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for key in ('AT', 'AV', 'TV', 'ATV'):
        torch.testing.assert_close(result['dividends'][key], torch.zeros_like(expected), atol=2e-6, rtol=0)


def test_latent256_order_means_and_direct_output_without_anchor():
    from gcnet_missing_m3.coalition_mobius import CoalitionMobiusBlock
    torch.manual_seed(23)
    encoder, block = Encoder(dim=256).eval(), CoalitionMobiusBlock(256).eval()
    features, availability, umask = inputs()
    parts = block.decompose(encoder, features, availability, umask)
    expected_slots = []
    for order, projector in enumerate((block.p1, block.p2, block.p3), 1):
        slot = torch.zeros(4, 2, 256)
        for row in range(4):
            for batch in range(2):
                terms = [projector(parts['dividends'][key][row, batch])
                         for key, indices in SETS.items() if len(indices) == order
                         and bool(parts['eligible'][key][row, batch])]
                if terms:
                    slot[row, batch] = torch.stack(terms).mean(0)
        expected_slots.append(slot)
    node, _ = block(encoder, features, availability, umask)
    torch.testing.assert_close(block.last_order_slots, torch.stack(expected_slots, -2))
    expected = torch.zeros_like(node)
    valid = umask.T.bool()
    expected[valid] = block.pout(torch.cat([*expected_slots, availability], -1)[valid])
    torch.testing.assert_close(node, expected)
    assert node.shape == (4, 2, 256)
    assert not torch.allclose(node[valid], parts['full_node'][valid])
    assert block.last_diagnostics['order_active_counts'] == [7, 4, 1]
