"""Factories only: independent algorithms share the unchanged Flat interface."""
import importlib

from .meaningful_v3_registry import V3_FAMILIES


def build_v3(method, latent_dim=256, num_heads=8, value_dim=64):
    for family, methods in V3_FAMILIES.items():
        if method in methods:
            module = importlib.import_module('.meaningful_v3_' + family, __package__)
            return module.build(method, latent_dim, num_heads, value_dim)
    raise ValueError('Unknown source-grounded incremental method: ' + method)
