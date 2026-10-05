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
# Pooling variants are comparisons, not additional independent catalog methods.
NESTED_SWEEP = {
    'nested_ab_plain_gin': {'plain_gin': True},
    'nested_ab_no_markers': {'markers': False},
    'nested_ab_no_head_edges': {'head_edges': False},
    'nested_ab_last_layer': {'last_layer': True},
    'nested_dim32': {'dim': 32},
    'nested_dim128': {'dim': 128},
    'nested_depth1': {'depth': 1},
    'nested_depth2': {'depth': 2},
    'nested_groups1': {'groups': 1},
    'nested_groups4': {'groups': 4},
}
NEW40_VARIANTS = ('nested_gnn_rootaware_evidence',) + tuple(NESTED_SWEEP)
