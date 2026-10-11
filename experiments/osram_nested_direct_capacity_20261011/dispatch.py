"""Reuse the verified direct-input dispatcher, two concurrent healthy GPU7 runs."""
from experiments.osram_direct_input4_20261010 import dispatch

dispatch.METHODS = ('nested_direct_random_mlp256', 'nested_direct_random_mlp512',
                    'nested_direct_random_dim128', 'nested_direct_random_dim256')

if __name__ == '__main__':
    dispatch.main()
