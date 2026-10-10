"""MOSI original Nested with a scalar Gate only on its decoded Gap-T residual."""
from experiments.osram_core20_20261005.run import main


if __name__ == '__main__':
    main(fixed_method='nested_gnn_gap_t_residual_gate')
