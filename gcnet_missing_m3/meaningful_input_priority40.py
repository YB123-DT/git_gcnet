"""Only M11/M13 modify Flat adapter inputs; the Local skip is unchanged."""
def build_priority40(method, latent_dim=256, num_heads=8, value_dim=64):
    from .priority40_conditioning import build_input
    return build_input(method, latent_dim=latent_dim, forward_dim=num_heads * value_dim)
