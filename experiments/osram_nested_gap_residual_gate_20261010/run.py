"""Source-pinned MOSI original Nested with only decoded Gap residual gates."""
from experiments.osram_core20_20261005.run import main


if __name__ == '__main__':
    main(fixed_method='nested_gnn_gap_residual_gate')
