# Further storage inventory, not deletion authorization

Read-only check on biggpu2026-10-11. No additional files deleted.
Partial screening criterion: completed seed66 runs with mean8 W-F1 below80.3%
and existing config,history,metrics and8prediction files. Weight totals sum
regular .pt file sizes, not measured filesystem reclamation. Do not interpret
the storage threshold as a significance test or remove negative result records.

All roots below are relative to /data2/yb/remote_experiments.

|Batch|Eligible runs|Weight GiB|
|---|---:|---:|
|osram_meaningful20_20261003_round1|12|23.380|
|osram_meaningful20_round2_20261004|11|23.128|
|osram_new40_gpu0123_20261004|5|9.470|
|osram_new40_gpu6_eleven_20261004|6|12.122|
|Total|34|68.100|

Candidate IDs by batch:

- Round1: dgcnn_dynamic_edgeconv, tabnet, rrn_evidence, ed_hnn,
  graph_multiset_transformer, otke, node, allset_transformer,
  equilibrium_aggregation, capsule_variational_bayes, capsule_dynamic_routing,
  slot_attention.
- Round2: dense_coattention, mcan_encoder_decoder, fspool_fsunpool_evidence,
  mac_control_read_write, conditional_rational_quadratic_spline_coupling,
  learned_robust_pca_evidence_decomposition, dual_attention_symbolic_relations,
  compositional_search_retrieval, rat_spn_evidence_circuit,
  spdnet_bimap_reeig_logeig, tokenlearner_fuser_v11.
- New40 GPU0123: sparsemap_role_partition, repr_lista_cpss_sparse_pursuit,
  next_hyperbolic_gyrovector_readout, conditional_new_01_full_dynamic_hypernetwork,
  graph_matching_base_gap_pairs.
- New40 GPU6: conditional_new_05_contracting_ren,
  conditional_new_03_hamiltonian_flow, conditional_new_02_ltc_conductance,
  conditional_new_04_lagrangian_flow, repr_sos_signed_probability_circuit,
  janossy_full_role_symmetrization.

Nested, Neural Production, Perceiver, CWN and previously protected high-score
candidates are excluded. No three-seed or cross-dataset roots inspected for
deletion. Before deleting a future approved subset, recheck live dependencies,
resolve exact paths, archive hashes and verify retained evidence as previously.
