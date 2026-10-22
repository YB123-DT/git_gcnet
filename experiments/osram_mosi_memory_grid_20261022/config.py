"""Two one-factor MOSI Flat sweeps, sharing one freshly trained control."""
from dataclasses import replace

RATES = tuple(f'{i/10:.1f}' for i in range(8))
MODES = {
    'baseline': (.6,64),
    'step_0p2': (.2,64), 'step_0p4': (.4,64),
    'step_0p8': (.8,64), 'step_1p0': (1.,64),
    'head_32': (.6,32), 'head_48': (.6,48),
    'head_96': (.6,96), 'head_128': (.6,128),
}
FAMILIES = {
    'write_step': ('step_0p2','step_0p4','baseline','step_0p8','step_1p0'),
    'head_dim': ('head_32','head_48','baseline','head_96','head_128'),
}


def candidate_config(cfg,mode):
    if mode not in MODES:
        raise ValueError('Only the nine predeclared MOSI configurations are approved')
    # Reuse the existing complete original-Flat guard, not its small-Flat change.
    from experiments.osram_mosi_v3_capacity_20261022.worker import candidate_config as guard
    guard(cfg,'f512_plain')
    step,dim = MODES[mode]
    return replace(cfg,osram_write_step=step,osram_key_dim=dim,osram_value_dim=dim)
