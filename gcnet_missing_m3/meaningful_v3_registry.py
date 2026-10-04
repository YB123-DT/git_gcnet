"""Incremental implementations; the 120 target is not a completion count."""

V3_FAMILIES = {
    'routing': (
        'routing_predinet_bound_predicates',
        'routing_soft_moe_dispatch_expert_combine',
        'routing_scl_shared_compositional_maps',
        'routing_esbn_ephemeral_symbol_binding',
    ),
    'representation': (
        'repr80_grande_nonoblivious_leaf_ensemble',
        'repr80_mps_feature_contraction',
        'repr80_neural_power_units',
        'repr80_mera_density_coarsegraining',
    ),
    'probabilistic': (
        'survey40_adf_gaussian_moment_propagation',
        'survey80_recurrent_kalman_network',
        'survey80_robust_gnc_tls_consensus',
        'survey80_particle_filter_rnn',
    ),
    'structured': (
        'survey80_struct_nbfnet',
        'survey80_struct_neural_lp',
        'survey80_struct_qre_game',
    ),
    'dynamics': (
        'dynamics_srwm_self_modifying_program',
        'dynamics_differentiable_tree_machine',
        'dynamics_neural_shuffle_exchange',
    ),
    'geometry': (
        'geometry_po_canonical_sequence',
        'geometry_repset_exact_template_matching',
    ),
    'logic': (
        'survey80_struct_neural_theorem_prover',
        'survey80_struct_deepproblog',
    ),
}
V3_METHODS = tuple(name for names in V3_FAMILIES.values() for name in names)
