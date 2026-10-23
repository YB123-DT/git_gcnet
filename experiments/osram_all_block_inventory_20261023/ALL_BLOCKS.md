# 全部机制/配置入口逐项清单

INTERNAL DIAGNOSTIC ONLY

注册不等于训练；训练证据数量含本地/远程/归档副本，不是独立训练次数。
精确配置、分数和选点见 RUNS.csv、METRICS.csv，原文表格见 SOURCE_TABLE_ROWS.csv。

| ID | 名称 | 所属批次 | 配置变体 | 插入位置 | 实现状态 | 同ID运行证据 |
|---|---|---|---|---|---|---:|
| conditional_hold_11_nbeats | N-BEATS: Neural basis expansion analysis for interpretable time series forecasting | catalog_only_not_current_selector | False | Not established by current implemented selectors | catalog_only_no_selector_mapping | 0 |
| C01 | Differentiable Neural Computer | core20 | False | Memory backend replacement | source_present | 1 |
| C02 | Compressive Transformer | core20 | False | Memory backend replacement | source_present | 1 |
| C03 | VQ-VAE | core20 | False | Value write representation | source_present | 1 |
| C04 | HiPPO | core20 | False | Memory backend replacement | source_present | 1 |
| C05 | Hierarchical Multiscale RNN | core20 | False | Memory backend replacement | source_present | 1 |
| C06 | Neural CDE | core20 | False | Memory backend replacement | source_present | 1 |
| C07 | Recurrent Independent Mechanisms | core20 | False | Memory backend replacement | source_present | 3 |
| C08 | TTT-MLP | core20 | False | Memory backend replacement | source_present | 1 |
| C09 | Recurrent Kalman Network | core20 | False | Memory backend replacement | source_present | 1 |
| C10 | Particle Filter RNN | core20 | False | Memory backend replacement | source_present | 1 |
| C11 | MMVAE | core20 | False | Observed representation and auxiliary objective | source_present | 1 |
| C12 | Natural Posterior Network | core20 | False | Task head/distribution replacement | source_present | 1 |
| C13 | Deep Equilibrium Model | core20 | False | Readout replacement | source_present | 1 |
| C14 | Universal Transformer + ACT | core20 | False | Readout replacement | source_present | 1 |
| C15 | FactorCL | core20 | False | Observed representation and auxiliary objective | source_present | 1 |
| C16 | Neural Relational Inference | core20 | False | Readout replacement | source_present | 1 |
| C17 | ProtoPNet | core20 | False | Task head/distribution replacement | source_present | 1 |
| C18 | Group DRO | core20 | False | Training objective/optimizer | source_present | 1 |
| C19 | Fishr | core20 | False | Training objective/optimizer | source_present | 1 |
| C20 | CAGrad | core20 | False | Training objective/optimizer | source_present | 2 |
| C17-binary-control | Matched binary task control for ProtoPNet | core20_controls | True | Task/optimizer control | source_present | 1 |
| C20-mean-control | Mean-gradient control for CAGrad | core20_controls | True | Task/optimizer control | source_present | 1 |
| allset_transformer | You are AllSet: A Multiset Learning Framework for Hypergraph Neural Networks | meaningful20_round1 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 2 |
| capsule_dynamic_routing | Dynamic Routing Between Capsules | meaningful20_round1 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 2 |
| capsule_variational_bayes | Capsule Routing via Variational Bayes | meaningful20_round1 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 2 |
| crate_mssa_ista_full | White-Box Transformers via Sparse Rate Reduction | meaningful20_round1 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 2 |
| dgcnn_dynamic_edgeconv | Dynamic Graph CNN for Learning on Point Clouds | meaningful20_round1 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 2 |
| ed_hnn | Equivariant Hypergraph Diffusion Neural Operators | meaningful20_round1 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 2 |
| egt_evidence | Global Self-Attention as a Replacement for Graph Convolution | meaningful20_round1 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 2 |
| equilibrium_aggregation | Equilibrium Aggregation: Encoding Sets via Optimization | meaningful20_round1 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 2 |
| graph_multiset_transformer | Accurate Learning of Graph Representations with Graph Multiset Pooling | meaningful20_round1 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 2 |
| hamburger_nmf_full | Is Attention Better Than Matrix Decomposition? | meaningful20_round1 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 2 |
| hyper_sagnn | Hyper-SAGNN: a self-attention based graph neural network for hypergraphs | meaningful20_round1 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 2 |
| node | Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data | meaningful20_round1 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 2 |
| otke | A Trainable Optimal Transport Embedding for Feature Aggregation and its Relationship to Attention | meaningful20_round1 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 2 |
| perceiver_io | Perceiver IO: A General Architecture for Structured Inputs & Outputs | meaningful20_round1 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 2 |
| pna_evidence | Principal Neighbourhood Aggregation for Graph Nets | meaningful20_round1 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 2 |
| residual_gated_graph_evidence | Residual Gated Graph ConvNets | meaningful20_round1 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 2 |
| rrn_evidence | Recurrent Relational Networks | meaningful20_round1 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 2 |
| sheaf_hypergnn_diag | Sheaf Hypergraph Networks | meaningful20_round1 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 2 |
| slot_attention | Object-Centric Learning with Slot Attention | meaningful20_round1 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 2 |
| tabnet | TabNet: Attentive Interpretable Tabular Learning | meaningful20_round1 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 2 |
| compositional_search_retrieval | compositional search retrieval | meaningful20_round2 | False | Flat Adapter input Local/Base/Gap correction; original Local Skip unchanged | source_present | 1 |
| conditional_rational_quadratic_spline_coupling | conditional rational quadratic spline coupling | meaningful20_round2 | False | Flat Adapter input Local/Base/Gap correction; original Local Skip unchanged | source_present | 1 |
| contractive_deep_equilibrium_evidence | contractive deep equilibrium evidence | meaningful20_round2 | False | Flat Adapter input Local/Base/Gap correction; original Local Skip unchanged | source_present | 1 |
| dcformer_dynamic_head_composition | dcformer dynamic head composition | meaningful20_round2 | False | Flat Adapter input Local/Base/Gap correction; original Local Skip unchanged | source_present | 1 |
| dense_coattention | dense coattention | meaningful20_round2 | False | Flat Adapter input Local/Base/Gap correction; original Local Skip unchanged | source_present | 1 |
| differential_attention_v1 | differential attention v1 | meaningful20_round2 | False | Flat Adapter input Local/Base/Gap correction; original Local Skip unchanged | source_present | 1 |
| dual_attention_symbolic_relations | dual attention symbolic relations | meaningful20_round2 | False | Flat Adapter input Local/Base/Gap correction; original Local Skip unchanged | source_present | 109 |
| flowformer_conservation | flowformer conservation | meaningful20_round2 | False | Flat Adapter input Local/Base/Gap correction; original Local Skip unchanged | source_present | 1 |
| fspool_fsunpool_evidence | fspool fsunpool evidence | meaningful20_round2 | False | Flat Adapter input Local/Base/Gap correction; original Local Skip unchanged | source_present | 1 |
| learned_robust_pca_evidence_decomposition | learned robust pca evidence decomposition | meaningful20_round2 | False | Flat Adapter input Local/Base/Gap correction; original Local Skip unchanged | source_present | 418 |
| mac_control_read_write | mac control read write | meaningful20_round2 | False | Flat Adapter input Local/Base/Gap correction; original Local Skip unchanged | source_present | 1 |
| mbt_role_bottleneck | mbt role bottleneck | meaningful20_round2 | False | Flat Adapter input Local/Base/Gap correction; original Local Skip unchanged | source_present | 145 |
| mcan_encoder_decoder | mcan encoder decoder | meaningful20_round2 | False | Flat Adapter input Local/Base/Gap correction; original Local Skip unchanged | source_present | 1 |
| netvlad_residual_encoding | netvlad residual encoding | meaningful20_round2 | False | Flat Adapter input Local/Base/Gap correction; original Local Skip unchanged | source_present | 0 |
| neural_logic_machine_evidence | neural logic machine evidence | meaningful20_round2 | False | Flat Adapter input Local/Base/Gap correction; original Local Skip unchanged | source_present | 1 |
| ppgn_pair_composition | ppgn pair composition | meaningful20_round2 | False | Flat Adapter input Local/Base/Gap correction; original Local Skip unchanged | source_present | 0 |
| rat_spn_evidence_circuit | rat spn evidence circuit | meaningful20_round2 | False | Flat Adapter input Local/Base/Gap correction; original Local Skip unchanged | source_present | 1 |
| spdnet_bimap_reeig_logeig | spdnet bimap reeig logeig | meaningful20_round2 | False | Flat Adapter input Local/Base/Gap correction; original Local Skip unchanged | source_present | 1 |
| tokenlearner_fuser_v11 | tokenlearner fuser v11 | meaningful20_round2 | False | Flat Adapter input Local/Base/Gap correction; original Local Skip unchanged | source_present | 8 |
| tome_merge_reconstruct | tome merge reconstruct | meaningful20_round2 | False | Flat Adapter input Local/Base/Gap correction; original Local Skip unchanged | source_present | 0 |
| dynamics_differentiable_tree_machine | dynamics differentiable tree machine | meaningful_v3_incremental | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 0 |
| dynamics_neural_shuffle_exchange | dynamics neural shuffle exchange | meaningful_v3_incremental | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 0 |
| dynamics_srwm_self_modifying_program | dynamics srwm self modifying program | meaningful_v3_incremental | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 0 |
| geometry_po_canonical_sequence | geometry po canonical sequence | meaningful_v3_incremental | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 0 |
| geometry_repset_exact_template_matching | geometry repset exact template matching | meaningful_v3_incremental | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 0 |
| repr80_grande_nonoblivious_leaf_ensemble | repr80 grande nonoblivious leaf ensemble | meaningful_v3_incremental | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 0 |
| repr80_mera_density_coarsegraining | repr80 mera density coarsegraining | meaningful_v3_incremental | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 0 |
| repr80_mps_feature_contraction | repr80 mps feature contraction | meaningful_v3_incremental | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 0 |
| repr80_neural_power_units | repr80 neural power units | meaningful_v3_incremental | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 0 |
| routing_esbn_ephemeral_symbol_binding | routing esbn ephemeral symbol binding | meaningful_v3_incremental | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 0 |
| routing_predinet_bound_predicates | routing predinet bound predicates | meaningful_v3_incremental | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 0 |
| routing_scl_shared_compositional_maps | routing scl shared compositional maps | meaningful_v3_incremental | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 0 |
| routing_soft_moe_dispatch_expert_combine | routing soft moe dispatch expert combine | meaningful_v3_incremental | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 0 |
| survey40_adf_gaussian_moment_propagation | survey40 adf gaussian moment propagation | meaningful_v3_incremental | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 0 |
| survey80_particle_filter_rnn | survey80 particle filter rnn | meaningful_v3_incremental | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 0 |
| survey80_recurrent_kalman_network | survey80 recurrent kalman network | meaningful_v3_incremental | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 0 |
| survey80_robust_gnc_tls_consensus | survey80 robust gnc tls consensus | meaningful_v3_incremental | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 0 |
| survey80_struct_deepproblog | survey80 struct deepproblog | meaningful_v3_incremental | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 0 |
| survey80_struct_nbfnet | survey80 struct nbfnet | meaningful_v3_incremental | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 0 |
| survey80_struct_neural_lp | survey80 struct neural lp | meaningful_v3_incremental | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 0 |
| survey80_struct_neural_theorem_prover | survey80 struct neural theorem prover | meaningful_v3_incremental | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 0 |
| survey80_struct_qre_game | survey80 struct qre game | meaningful_v3_incremental | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 0 |
| nested_concat_capacity_control | nested concat capacity control | nested_concat_comparison | True | No graph; enlarge Flat Adapter hidden width to match concatenation-model parameter budget | source_present | 0 |
| nested_rooted_concat | nested rooted concat | nested_concat_comparison | True | Concatenate 320-d rooted graph features to original Flat Adapter INPUT (4352->4672); retain Local Skip | source_present | 0 |
| meld_local_d32_l3_g2_m128 | meld local d32 l3 g2 m128 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_local_d32_l3_g2_m64 | meld local d32 l3 g2 m64 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_local_d32_l3_g4_m128 | meld local d32 l3 g4 m128 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_local_d32_l4_g2_m128 | meld local d32 l4 g2 m128 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_local_d32_l4_g4_m64 | meld local d32 l4 g4 m64 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_local_d48_l3_g2_m64 | meld local d48 l3 g2 m64 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_local_d48_l3_g4_m64 | meld local d48 l3 g4 m64 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_local_d48_l4_g2_m64 | meld local d48 l4 g2 m64 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_local_d48_l4_g4_m64 | meld local d48 l4 g4 m64 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_local_d64_l3_g2_m128 | meld local d64 l3 g2 m128 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_local_d64_l3_g4_m64 | meld local d64 l3 g4 m64 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_local_d64_l4_g2_m64 | meld local d64 l4 g2 m64 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_local_d64_l4_g4_m128 | meld local d64 l4 g4 m128 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_local_d64_l4_g4_m64 | meld local d64 l4 g4 m64 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d128_l1_g1_m256 | meld nested d128 l1 g1 m256 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d128_l1_g2_m512 | meld nested d128 l1 g2 m512 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d128_l1_g4_m64 | meld nested d128 l1 g4 m64 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d128_l1_g8_m128 | meld nested d128 l1 g8 m128 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d128_l2_g1_m512 | meld nested d128 l2 g1 m512 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d128_l2_g2_m64 | meld nested d128 l2 g2 m64 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d128_l2_g4_m128 | meld nested d128 l2 g4 m128 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d128_l2_g8_m256 | meld nested d128 l2 g8 m256 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d128_l3_g1_m64 | meld nested d128 l3 g1 m64 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d128_l3_g2_m128 | meld nested d128 l3 g2 m128 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d128_l3_g4_m256 | meld nested d128 l3 g4 m256 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d128_l3_g8_m512 | meld nested d128 l3 g8 m512 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d128_l4_g1_m128 | meld nested d128 l4 g1 m128 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d128_l4_g2_m256 | meld nested d128 l4 g2 m256 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d128_l4_g4_m512 | meld nested d128 l4 g4 m512 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d128_l4_g8_m64 | meld nested d128 l4 g8 m64 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 3 |
| meld_nested_d256_l1_g1_m512 | meld nested d256 l1 g1 m512 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d256_l1_g2_m64 | meld nested d256 l1 g2 m64 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d256_l1_g4_m128 | meld nested d256 l1 g4 m128 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d256_l1_g8_m256 | meld nested d256 l1 g8 m256 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d256_l2_g1_m64 | meld nested d256 l2 g1 m64 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d256_l2_g2_m128 | meld nested d256 l2 g2 m128 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d256_l2_g4_m256 | meld nested d256 l2 g4 m256 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d256_l2_g8_m512 | meld nested d256 l2 g8 m512 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d256_l3_g1_m128 | meld nested d256 l3 g1 m128 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 3 |
| meld_nested_d256_l3_g2_m256 | meld nested d256 l3 g2 m256 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d256_l3_g4_m512 | meld nested d256 l3 g4 m512 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d256_l3_g8_m64 | meld nested d256 l3 g8 m64 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d256_l4_g1_m256 | meld nested d256 l4 g1 m256 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d256_l4_g2_m512 | meld nested d256 l4 g2 m512 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d256_l4_g4_m64 | meld nested d256 l4 g4 m64 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d256_l4_g8_m128 | meld nested d256 l4 g8 m128 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d32_l1_g1_m64 | meld nested d32 l1 g1 m64 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d32_l1_g2_m128 | meld nested d32 l1 g2 m128 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d32_l1_g4_m256 | meld nested d32 l1 g4 m256 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d32_l1_g8_m512 | meld nested d32 l1 g8 m512 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 3 |
| meld_nested_d32_l2_g1_m128 | meld nested d32 l2 g1 m128 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d32_l2_g2_m256 | meld nested d32 l2 g2 m256 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d32_l2_g4_m512 | meld nested d32 l2 g4 m512 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d32_l2_g8_m64 | meld nested d32 l2 g8 m64 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d32_l3_g1_m256 | meld nested d32 l3 g1 m256 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d32_l3_g2_m512 | meld nested d32 l3 g2 m512 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d32_l3_g4_m64 | meld nested d32 l3 g4 m64 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d32_l3_g8_m128 | meld nested d32 l3 g8 m128 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 3 |
| meld_nested_d32_l4_g1_m512 | meld nested d32 l4 g1 m512 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d32_l4_g2_m64 | meld nested d32 l4 g2 m64 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 6 |
| meld_nested_d32_l4_g4_m128 | meld nested d32 l4 g4 m128 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 3 |
| meld_nested_d32_l4_g8_m256 | meld nested d32 l4 g8 m256 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d64_l1_g1_m128 | meld nested d64 l1 g1 m128 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d64_l1_g2_m256 | meld nested d64 l1 g2 m256 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d64_l1_g4_m512 | meld nested d64 l1 g4 m512 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d64_l1_g8_m64 | meld nested d64 l1 g8 m64 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d64_l2_g1_m256 | meld nested d64 l2 g1 m256 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d64_l2_g2_m512 | meld nested d64 l2 g2 m512 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d64_l2_g4_m64 | meld nested d64 l2 g4 m64 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d64_l2_g8_m128 | meld nested d64 l2 g8 m128 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d64_l3_g1_m512 | meld nested d64 l3 g1 m512 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d64_l3_g2_m64 | meld nested d64 l3 g2 m64 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d64_l3_g4_m128 | meld nested d64 l3 g4 m128 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d64_l3_g8_m256 | meld nested d64 l3 g8 m256 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d64_l4_g1_m64 | meld nested d64 l4 g1 m64 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d64_l4_g2_m128 | meld nested d64 l4 g2 m128 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| meld_nested_d64_l4_g4_m256 | meld nested d64 l4 g4 m256 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 3 |
| meld_nested_d64_l4_g8_m512 | meld nested d64 l4 g8 m512 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| mosei_nested_depth2 | mosei nested depth2 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 2 |
| mosei_nested_dim32 | mosei nested dim32 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| mosei_nested_dim32_depth2 | mosei nested dim32 depth2 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 4 |
| mosei_nested_dim32_depth2_mlp64 | mosei nested dim32 depth2 mlp64 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| mosei_nested_groups4 | mosei nested groups4 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| mosei_nested_mlp128 | mosei nested mlp128 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 2 |
| nested_ab_last_layer | nested ab last layer | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| nested_ab_no_head_edges | nested ab no head edges | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| nested_ab_no_markers | nested ab no markers | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| nested_ab_plain_gin | nested ab plain gin | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| nested_depth1 | nested depth1 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| nested_depth2 | nested depth2 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| nested_dim128 | nested dim128 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| nested_dim32 | nested dim32 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| nested_dim704 | nested dim704 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| nested_direct_random_dim128 | nested direct random dim128 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| nested_direct_random_dim128_mlp256 | nested direct random dim128 mlp256 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| nested_direct_random_dim128_mlp512 | nested direct random dim128 mlp512 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| nested_direct_random_dim256 | nested direct random dim256 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| nested_direct_random_dim256_mlp512 | nested direct random dim256 mlp512 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| nested_direct_random_mlp128 | nested direct random mlp128 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| nested_direct_random_mlp256 | nested direct random mlp256 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| nested_direct_random_mlp512 | nested direct random mlp512 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| nested_groups1 | nested groups1 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| nested_groups1_dim512 | nested groups1 dim512 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| nested_groups4 | nested groups4 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| nested_mlp256 | nested mlp256 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 2 |
| nested_mlp512 | nested mlp512 | nested_configuration_sweeps | True | Flat Adapter input correction/replacement, per variant | source_present | 1 |
| conditional_03_neural_ode_finite_flow | Neural Ordinary Differential Equations | new40 | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 0 |
| conditional_new_01_full_dynamic_hypernetwork | HyperNetworks | new40 | False | Flat Adapter input Base/Gap correction; original adapter Local and Local Skip retained | source_present | 1 |
| conditional_new_02_ltc_conductance | Liquid Time-constant Networks | new40 | False | Flat Adapter input Base/Gap correction; original adapter Local and Local Skip retained | source_present | 1 |
| conditional_new_03_hamiltonian_flow | Hamiltonian Neural Networks | new40 | False | Flat Adapter input Base/Gap correction; original adapter Local and Local Skip retained | source_present | 1 |
| conditional_new_04_lagrangian_flow | Lagrangian Neural Networks | new40 | False | Flat Adapter input Base/Gap correction; original adapter Local and Local Skip retained | source_present | 1 |
| conditional_new_05_contracting_ren | Recurrent Equilibrium Networks: Flexible Dynamic Models with Guaranteed Stability and Robustness | new40 | False | Flat Adapter input Base/Gap correction; original adapter Local and Local Skip retained | source_present | 1 |
| conditional_new_06_rim | Recurrent Independent Mechanisms | new40 | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 1 |
| conditional_new_07_neural_production | Neural Production Systems: Learning Rule-Governed Visual Dynamics | new40 | False | Flat Adapter input Base/Gap correction; original adapter Local and Local Skip retained | source_present | 11 |
| conditional_new_08_neural_interpreter | Dynamic Inference with Neural Interpreters | new40 | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 1 |
| conditional_new_09_sympnet | SympNets: Intrinsic structure-preserving symplectic networks for identifying Hamiltonian systems | new40 | False | Flat Adapter input Base/Gap correction; original adapter Local and Local Skip retained | source_present | 1 |
| conditional_new_10_cornn | Coupled Oscillatory Recurrent Neural Network (coRNN): An accurate and (gradient) stable architecture for learning long time dependencies | new40 | False | Flat Adapter input Base/Gap correction; original adapter Local and Local Skip retained | source_present | 1 |
| crfrnn_latent_evidence_states | Conditional Random Fields as Recurrent Neural Networks | new40 | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 1 |
| cwn_cellular_evidence | Weisfeiler and Lehman Go Cellular: CW Networks | new40 | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 1 |
| diffpool_hierarchical_evidence_graph | Hierarchical Graph Representation Learning with Differentiable Pooling | new40 | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 1 |
| graph_matching_base_gap_pairs | Graph Matching Networks for Learning the Similarity of Graph Structured Objects | new40 | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 8 |
| graph_unet_evidence_encoder_decoder | Graph U-Nets | new40 | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 1 |
| janossy_full_role_symmetrization | Janossy Pooling: Learning Deep Permutation-Invariant Functions for Variable-Size Inputs | new40 | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 1 |
| matrix_tree_nonprojective_evidence | Learning Structured Text Representations | new40 | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 1 |
| n3_recursive_neighbor_volumes | Neural Nearest Neighbors Networks | new40 | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 1 |
| nested_gnn_rooted_evidence | Nested Graph Neural Networks | new40 | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 129 |
| next_bernstein_spectral_readout | BernNet: Learning Arbitrary Graph Spectral Filters via Bernstein Approximation | new40 | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 0 |
| next_diffusion_scattering_readout | Diffusion Scattering Transforms on Graphs | new40 | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 0 |
| next_hyperbolic_gyrovector_readout | Hyperbolic Neural Networks | new40 | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 1 |
| next_sandwich_lipschitz_readout | Direct Parameterization of Lipschitz-Bounded Deep Networks | new40 | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 0 |
| optimization_dpp_subset_readout | Determinantal Point Processes for Machine Learning | new40 | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 1 |
| pointcnn_x_transformed_evidence | PointCNN: Convolution On X-Transformed Points | new40 | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 0 |
| repr_bcos_alignment_network | B-cos Networks: Alignment is All We Need for Interpretability | new40 | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 0 |
| repr_convex_potential_gradient_flow | Convex Potential Flows: Universal Probability Distributions with Optimal Transport and Convex Optimization | new40 | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 1 |
| repr_deep_lattice_composition | Deep Lattice Networks and Partial Monotonic Functions | new40 | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 0 |
| repr_differentiable_logic_circuit | Deep Differentiable Logic Gate Networks | new40 | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 0 |
| repr_dst_corrected_evidence_combination | An evidential classifier based on Dempster-Shafer theory and deep learning | new40 | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 1 |
| repr_grassmann_projection_pooling | Building Deep Networks on Grassmann Manifolds | new40 | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 1 |
| repr_kan_function_composition | KAN: Kolmogorov-Arnold Networks | new40 | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 0 |
| repr_linf_distance_network | Towards Certifying L-infinity Robustness using Neural Networks with L-inf-dist Neurons | new40 | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 0 |
| repr_lista_cpss_sparse_pursuit | Theoretical Linear Convergence of Unfolded ISTA and Its Practical Weights and Thresholds | new40 | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 1 |
| repr_mfn_gabor_filter_chain | Multiplicative Filter Networks | new40 | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 0 |
| repr_neural_kernel_composition | Differentiable Compositional Kernel Learning for Gaussian Processes | new40 | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 1 |
| repr_sos_signed_probability_circuit | Sum of Squares Circuits | new40 | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 1 |
| simplicial_hodge_evidence | Simplicial Neural Networks | new40 | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 1 |
| sparsemap_role_partition | SparseMAP: Differentiable Sparse Structured Inference | new40 | False | Flat Adapter input correction; original Local Skip unchanged | source_present | 1 |
| cwn_cellular_direct | cwn cellular direct | new40_architecture_and_placement_variants | True | Flat Adapter input correction/replacement; inspect explicit variant | source_present | 2 |
| nested_gnn_direct_evidence | nested gnn direct evidence | new40_architecture_and_placement_variants | True | Flat Adapter input correction/replacement; inspect explicit variant | source_present | 1 |
| nested_gnn_direct_random_evidence | nested gnn direct random evidence | new40_architecture_and_placement_variants | True | Flat Adapter input correction/replacement; inspect explicit variant | source_present | 1 |
| nested_gnn_gap_residual_gate | nested gnn gap residual gate | new40_architecture_and_placement_variants | True | Flat Adapter input correction/replacement; inspect explicit variant | source_present | 3 |
| nested_gnn_gap_t_residual_gate | nested gnn gap t residual gate | new40_architecture_and_placement_variants | True | Flat Adapter input correction/replacement; inspect explicit variant | source_present | 3 |
| nested_gnn_rootaware_evidence | nested gnn rootaware evidence | new40_architecture_and_placement_variants | True | Flat Adapter input correction/replacement; inspect explicit variant | source_present | 3 |
| nested_local8_evidence | nested local8 evidence | new40_architecture_and_placement_variants | True | Flat Adapter input correction/replacement; inspect explicit variant | source_present | 1 |
| neural_production_direct | neural production direct | new40_architecture_and_placement_variants | True | Flat Adapter input correction/replacement; inspect explicit variant | source_present | 2 |
| neural_production_local | neural production local | new40_architecture_and_placement_variants | True | Flat Adapter input correction/replacement; inspect explicit variant | source_present | 1 |
| neural_production_local_w256 | neural production local w256 | new40_architecture_and_placement_variants | True | Flat Adapter input correction/replacement; inspect explicit variant | source_present | 1 |
| perceiver_io_direct | perceiver io direct | new40_architecture_and_placement_variants | True | Flat Adapter input correction/replacement; inspect explicit variant | source_present | 2 |
| m01_sab | Set Transformer SAB | priority40 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 1 |
| m02_dat | DAT relational/symbolic attention | priority40 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 1 |
| m03_gatv2 | GATv2 | priority40 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 1 |
| m04_edgeconv | EdgeConv | priority40 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 1 |
| m05_pna | PNA | priority40 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 1 |
| m06_tfn | Tensor Fusion Network | priority40 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 1 |
| m07_lmf | Low-rank Multimodal Fusion | priority40 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 1 |
| m08_mfb | MFB | priority40 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 1 |
| m09_tucker | Tucker fusion | priority40 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 1 |
| m10_block | BLOCK fusion | priority40 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 1 |
| m11_film | FiLM | priority40 | False | Flat Adapter input modulation; original Local Skip unchanged | source_present | 1 |
| m12_cross_stitch | Cross-stitch | priority40 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 1 |
| m13_mmtm | MMTM | priority40 | False | Flat Adapter input modulation; original Local Skip unchanged | source_present | 1 |
| m14_gct | GCT | priority40 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 1 |
| m15_dynamic_relu | Dynamic ReLU | priority40 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 1 |
| m16_deep_sets | Deep Sets | priority40 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 1 |
| m17_janossy | Janossy Pooling | priority40 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 1 |
| m18_fspool | FSPool | priority40 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 1 |
| m19_netvlad | NetVLAD | priority40 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 1 |
| m20_attentive_statistics | Attentive Statistics Pooling | priority40 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 1 |
| m21_nfm | NFM bi-interaction | priority40 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 1 |
| m22_dcnv2 | DCN-V2 | priority40 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 1 |
| m23_cin | xDeepFM CIN | priority40 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 1 |
| m24_afm | Attentional Factorization Machine | priority40 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 1 |
| m25_kan | KAN | priority40 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 1 |
| m26_dolg | DOLG orthogonal fusion | priority40 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 1 |
| m27_isqrt_cov | iSQRT-COV | priority40 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 1 |
| m28_xcit_xca | XCiT XCA | priority40 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 3 |
| m29_set_norm | Set Norm | priority40 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 1 |
| m30_dyt | Dynamic Tanh DyT | priority40 | False | Replace first Flat Adapter LayerNorm only; final emotion_norm retained | source_present | 1 |
| m31_mlp_mixer | MLP-Mixer | priority40 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 1 |
| m32_gmlp_sgu | gMLP Spatial Gating Unit | priority40 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 1 |
| m33_fnet | FNet Fourier mixing | priority40 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 1 |
| m34_convnext_grn | ConvNeXt V2 Global Response Normalization | priority40 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 1 |
| m35_dynamixer | DynaMixer | priority40 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 1 |
| m36_hopfield | Modern Hopfield | priority40 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 1 |
| m37_entmax_sab | Entmax-1.5 SAB control | priority40 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 1 |
| m38_soft_moe | Soft MoE | priority40 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 1 |
| m39_node | NODE differentiable oblivious trees | priority40 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 1 |
| m40_capsule | Capsule Dynamic Routing | priority40 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 1 |
| m03_gatv2_direct | m03 gatv2 direct | priority40_placement_controls | True | Direct Flat Adapter input replacement; not a new source algorithm | source_present | 2 |
| m05_pna_direct | m05 pna direct | priority40_placement_controls | True | Direct Flat Adapter input replacement; not a new source algorithm | source_present | 2 |
| m28_xcit_xca_direct | m28 xcit xca direct | priority40_placement_controls | True | Direct Flat Adapter input replacement; not a new source algorithm | source_present | 2 |
| R02 | DeltaProduct | research_r4 | False | Memory update replacement | source_present | 1 |
| R03 | MesaNet | research_r4 | False | Sufficient-statistic Memory and exact linear solve | source_present | 1 |
| R12 | Rank-N-Contrast | research_r4 | False | Task hidden representation ranking objective | source_present | 3 |
| R18 | LUPI Heteroscedastic Dropout | research_r4 | False | Training-only privileged task-hidden noise; inference identity | source_present | 1 |
| aff | aff | withdrawn_readout20 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 0 |
| autoint | autoint | withdrawn_readout20 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 0 |
| ban | ban | withdrawn_readout20 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 0 |
| block | block | withdrawn_readout20 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 0 |
| cbam | cbam | withdrawn_readout20 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 0 |
| cin | cin | withdrawn_readout20 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 0 |
| dcnv2 | dcnv2 | withdrawn_readout20 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 0 |
| din | din | withdrawn_readout20 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 0 |
| dlrm | dlrm | withdrawn_readout20 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 0 |
| eca | eca | withdrawn_readout20 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 0 |
| film | film | withdrawn_readout20 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 0 |
| gct | gct | withdrawn_readout20 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 0 |
| mcb | mcb | withdrawn_readout20 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 0 |
| mfb | mfb | withdrawn_readout20 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 0 |
| mlb | mlb | withdrawn_readout20 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 0 |
| mutan | mutan | withdrawn_readout20 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 0 |
| nonlocal | nonlocal | withdrawn_readout20 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 0 |
| se | se | withdrawn_readout20 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 0 |
| simam | simam | withdrawn_readout20 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 0 |
| sk | sk | withdrawn_readout20 | False | Flat pre-norm supplementary residual; original Flat retained | source_present | 0 |
