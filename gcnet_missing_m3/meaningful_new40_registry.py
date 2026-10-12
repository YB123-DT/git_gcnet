"""Code catalog only: registering a method does not authorize a training run."""
NEW40_FAMILIES = {
    'geometry': (
        'optimization_dpp_subset_readout',
        'next_hyperbolic_gyrovector_readout',
        'next_bernstein_spectral_readout',
        'next_diffusion_scattering_readout',
        'next_sandwich_lipschitz_readout',
    ),
    'functions': (
        'repr_kan_function_composition',
        'repr_deep_lattice_composition',
        'repr_differentiable_logic_circuit',
        'conditional_03_neural_ode_finite_flow',
    ),
    'structure': (
        'matrix_tree_nonprojective_evidence',
        'sparsemap_role_partition',
        'janossy_full_role_symmetrization',
        'diffpool_hierarchical_evidence_graph',
        'cwn_cellular_evidence',
        'simplicial_hodge_evidence',
        'nested_gnn_rooted_evidence',
        'graph_unet_evidence_encoder_decoder',
        'crfrnn_latent_evidence_states',
        'graph_matching_base_gap_pairs',
    ),
    'conditional': (
        'conditional_new_01_full_dynamic_hypernetwork',
        'conditional_new_02_ltc_conductance',
        'conditional_new_03_hamiltonian_flow',
        'conditional_new_04_lagrangian_flow',
        'conditional_new_05_contracting_ren',
        'conditional_new_06_rim',
        'conditional_new_07_neural_production',
        'conditional_new_08_neural_interpreter',
        'conditional_new_09_sympnet',
        'conditional_new_10_cornn',
    ),
    'representation': (
        'repr_dst_corrected_evidence_combination',
        'repr_bcos_alignment_network',
        'repr_mfn_gabor_filter_chain',
        'repr_linf_distance_network',
        'repr_lista_cpss_sparse_pursuit',
    ),
    'representation_extra': (
        'repr_sos_signed_probability_circuit',
        'repr_grassmann_projection_pooling',
        'repr_neural_kernel_composition',
        'repr_convex_potential_gradient_flow',
    ),
    'retrieval': ('n3_recursive_neighbor_volumes', 'pointcnn_x_transformed_evidence'),
}
NEW40_METHODS = tuple(method for methods in NEW40_FAMILIES.values() for method in methods)
# MELD uses a balanced 64-run orthogonal grid instead of the complete 4^4
# factorial.  For every (dimension, depth, grouping) triple, the MLP level is
# assigned cyclically.  This balances every level and every pair of factors
# while keeping the user-authorized screen to 64 Nested configurations.
_MELD_DIMS = (32, 64, 128, 256)
_MELD_DEPTHS = (1, 2, 3, 4)
_MELD_GROUPS = (1, 2, 4, 8)
_MELD_MLPS = (64, 128, 256, 512)
MELD_NESTED_GRID = {
    f"meld_nested_d{dim}_l{depth}_g{groups}_m{mlp}": {
        "dim": dim,
        "depth": depth,
        "groups": groups,
        "mlp_hidden": mlp,
    }
    for dim_index, dim in enumerate(_MELD_DIMS)
    for depth_index, depth in enumerate(_MELD_DEPTHS)
    for group_index, groups in enumerate(_MELD_GROUPS)
    for mlp in (_MELD_MLPS[(dim_index + depth_index + group_index) % 4],)
}

# A single 14-run local refinement around the compact region favored by the
# broad MELD screen.  Ten entries fill the previously untested cells of the
# 32/64-d, depth-3/4, group-2/4, MLP-64/128 cube; four 48-d entries probe the
# midpoint without increasing the MLP width.
_MELD_LOCAL_CUBE = (
    (dim, depth, groups, mlp)
    for dim in (32, 64)
    for depth in (3, 4)
    for groups in (2, 4)
    for mlp in (64, 128)
)
_MELD_EXISTING_CONFIGS = {
    tuple(config[key] for key in ("dim", "depth", "groups", "mlp_hidden"))
    for config in MELD_NESTED_GRID.values()
}
_MELD_LOCAL_NEW = [
    config for config in _MELD_LOCAL_CUBE if config not in _MELD_EXISTING_CONFIGS
] + [
    (48, depth, groups, 64)
    for depth in (3, 4)
    for groups in (2, 4)
]
MELD_NESTED_LOCAL14 = {
    f"meld_local_d{dim}_l{depth}_g{groups}_m{mlp}": {
        "dim": dim,
        "depth": depth,
        "groups": groups,
        "mlp_hidden": mlp,
    }
    for dim, depth, groups, mlp in _MELD_LOCAL_NEW
}

# Pooling variants are comparisons, not additional independent catalog methods.
NESTED_SWEEP = {
    # Locked CMU-MOSEI seed66 capacity screen.  These names are dataset-scoped
    # to keep the six-run authorization distinct from the earlier MOSI sweep.
    'mosei_nested_dim32': {'dim': 32},
    'mosei_nested_depth2': {'depth': 2},
    'mosei_nested_dim32_depth2': {'dim': 32, 'depth': 2},
    'mosei_nested_mlp128': {'mlp_hidden': 128},
    'mosei_nested_groups4': {'groups': 4},
    'mosei_nested_dim32_depth2_mlp64': {'dim': 32, 'depth': 2, 'mlp_hidden': 64},
    'nested_direct_random_mlp128': {'mlp_hidden': 128, 'residual': False, 'zero_decoder': False},
    'nested_direct_random_dim128_mlp256': {'dim': 128, 'mlp_hidden': 256, 'residual': False, 'zero_decoder': False},
    'nested_direct_random_dim128_mlp512': {'dim': 128, 'mlp_hidden': 512, 'residual': False, 'zero_decoder': False},
    'nested_direct_random_dim256_mlp512': {'dim': 256, 'mlp_hidden': 512, 'residual': False, 'zero_decoder': False},
    'nested_direct_random_mlp256': {'mlp_hidden': 256, 'residual': False, 'zero_decoder': False},
    'nested_direct_random_mlp512': {'mlp_hidden': 512, 'residual': False, 'zero_decoder': False},
    'nested_direct_random_dim128': {'dim': 128, 'residual': False, 'zero_decoder': False},
    'nested_direct_random_dim256': {'dim': 256, 'residual': False, 'zero_decoder': False},
    'nested_ab_plain_gin': {'plain_gin': True},
    'nested_ab_no_markers': {'markers': False},
    'nested_ab_no_head_edges': {'head_edges': False},
    'nested_ab_last_layer': {'last_layer': True},
    'nested_dim32': {'dim': 32},
    'nested_dim128': {'dim': 128},
    'nested_dim704': {'dim': 704},
    'nested_mlp256': {'mlp_hidden': 256},
    'nested_mlp512': {'mlp_hidden': 512},
    'nested_depth1': {'depth': 1},
    'nested_depth2': {'depth': 2},
    'nested_groups1': {'groups': 1},
    'nested_groups1_dim512': {'groups': 1, 'dim': 512},
    'nested_groups4': {'groups': 4},
    **MELD_NESTED_GRID,
    **MELD_NESTED_LOCAL14,
}
NEW40_VARIANTS = ('cwn_cellular_direct', 'perceiver_io_direct', 'neural_production_direct', 'nested_gnn_rootaware_evidence', 'nested_gnn_direct_evidence',
                  'nested_gnn_gap_residual_gate',
                  'nested_gnn_gap_t_residual_gate',
                  'nested_gnn_direct_random_evidence', 'neural_production_local', 'neural_production_local_w256') + tuple(NESTED_SWEEP) + ('nested_local8_evidence',)
