# Forty additional runnable readout adaptations

INTERNAL DIAGNOSTIC ONLY — CODE IMPLEMENTATION, NOT PERFORMANCE RESULTS

Implemented forty distinct registered cores on baseline commit `789c6dd`, branch `feature/osram-uniform-forced-text`. No training, evaluation, or new W-F1 was produced. This is a code catalog, not a forty-job queue.

The original Memory/query, Local skip, Flat/task head, task loss and random-missing protocol remain unchanged. Each module transforms only the Flat adapter inputs, with a zero-initialized correction bridge. No new dependencies. Shared hypotheses are allowed; aliases, width/depth variants and primitive wrappers are not counted as new mechanisms. Meaningful hypotheses do not establish efficacy.

## Verification

Executed on CPU:

```bash
/home/yangbin/miniconda3/envs/multimodalerc310/bin/python -m unittest tests.test_meaningful_new40
/home/yangbin/miniconda3/envs/multimodalerc310/bin/python -m unittest tests.test_meaningful_input tests.test_meaningful_block_integration
```

Two catalog/cap tests and three shared input/full-model integration tests passed. They cover the new forty and existing paths: identity initialization, inactive NaNs, padding/first-utterance masking, finite backward and optimizer updates, complete-model train/eval initial prediction and backbone/RNG equality. Full-model checks use synthetic features, not a MOSI experiment.

Real batch32 CUDA profiling on biggpu, full training, speed/memory feasibility and performance remain untested. HNN/LNN, adaptive ODE and cellular graphs may be expensive. Reuse the existing CUDA template to generate source-bound readiness before training; CPU tests are not GPU feasibility evidence.

## Explicit adaptations and limitations

NPS uses deterministic straight-through argmax instead of Gumbel noise. LNN retains actual Hessian/mixed derivatives with a position-dependent SPD mass Lagrangian and nonsingular solve. CPFlow computes the full ICNN gradient analytically, retaining mixed second derivatives during training. Grassmann uses across-branch projector pooling with regularized tiny eigengaps; its source includes AFEW facial-emotion video, though not MSA/MERC. SOS integrates paired leaves after squaring; DST corrects the author's Omega formula according to the paper. MFN's direct PDF was blocked: cached method text, official slides and full author core were used, not a claimed full-PDF download. Difflogic intentionally differs between soft training and hard evaluation. No claim of physical emotion dynamics, reliable uncertainty or recovered missing modalities is made.

These are equation-based readout transfers, not reproductions of forty complete source models or training recipes. Sources, nearest old methods and differences: [CATALOG.json](CATALOG.json), with detailed linked design files.

## Invocation

Keep an existing cfg84 command unchanged except `--osram-meaningful-block n3_recursive_neighbor_volumes` (or another ID below). The existing trainer supports every registered ID.

Preview a controlled-run selection without writes (executed):

```bash
python -m experiments.osram_new40_20261004.prepare --candidate n3_recursive_neighbor_volumes --dry-run
```

When training is authorized, reserve a selection on the model's biggpu environment:

```bash
python -m experiments.osram_new40_20261004.prepare --candidate n3_recursive_neighbor_volumes --output experiments/osram_new40_20261004/SELECTED.json
```

This updates the reservation ledger but starts no process and refuses to overwrite a manifest. Commit the selection and ledger, then use the existing round-two `manifest snapshot`, shared `cuda_check`, and `run` commands with this selected manifest. Keep all original required source/data/baseline hashes, readiness, GPU UUID and output-isolation arguments. Never train directly from the mutable worktree.

The global sixty-trained-method cap is unchanged: forty existing slots leave at most twenty additional reservations, not forty. Training remains seed66/100epochs/eight rates; existing per-rate Test-oracle scores are internal only, not validation-selected paper results. Physical biggpu GPU4 is prohibited. No existing remote queue was touched.

## Measured added parameters

Includes projections and decoders; these are not source-paper model sizes. [PARAMETERS.json](PARAMETERS.json) records file hashes.

| # | Method ID | Added parameters |
|---:|---|---:|
| 1 | `optimization_dpp_subset_readout` | 173,489 |
| 2 | `next_hyperbolic_gyrovector_readout` | 449,056 |
| 3 | `next_bernstein_spectral_readout` | 21,287 |
| 4 | `next_diffusion_scattering_readout` | 187,168 |
| 5 | `next_sandwich_lipschitz_readout` | 318,208 |
| 6 | `repr_kan_function_composition` | 334,336 |
| 7 | `repr_deep_lattice_composition` | 195,680 |
| 8 | `repr_differentiable_logic_circuit` | 240,128 |
| 9 | `conditional_03_neural_ode_finite_flow` | 299,648 |
| 10 | `matrix_tree_nonprojective_evidence` | 129,665 |
| 11 | `sparsemap_role_partition` | 322,819 |
| 12 | `janossy_full_role_symmetrization` | 347,648 |
| 13 | `diffpool_hierarchical_evidence_graph` | 130,386 |
| 14 | `cwn_cellular_evidence` | 362,112 |
| 15 | `simplicial_hodge_evidence` | 166,400 |
| 16 | `nested_gnn_rooted_evidence` | 159,235 |
| 17 | `graph_unet_evidence_encoder_decoder` | 122,176 |
| 18 | `crfrnn_latent_evidence_states` | 112,090 |
| 19 | `graph_matching_base_gap_pairs` | 91,712 |
| 20 | `conditional_new_01_full_dynamic_hypernetwork` | 825,792 |
| 21 | `conditional_new_02_ltc_conductance` | 449,088 |
| 22 | `conditional_new_03_hamiltonian_flow` | 305,793 |
| 23 | `conditional_new_04_lagrangian_flow` | 238,193 |
| 24 | `conditional_new_05_contracting_ren` | 298,504 |
| 25 | `conditional_new_06_rim` | 274,752 |
| 26 | `conditional_new_07_neural_production` | 175,168 |
| 27 | `conditional_new_08_neural_interpreter` | 141,296 |
| 28 | `conditional_new_09_sympnet` | 293,888 |
| 29 | `conditional_new_10_cornn` | 559,552 |
| 30 | `repr_dst_corrected_evidence_combination` | 99,958 |
| 31 | `repr_bcos_alignment_network` | 286,528 |
| 32 | `repr_mfn_gabor_filter_chain` | 258,496 |
| 33 | `repr_linf_distance_network` | 255,392 |
| 34 | `repr_lista_cpss_sparse_pursuit` | 239,365 |
| 35 | `repr_sos_signed_probability_circuit` | 13,961 |
| 36 | `repr_grassmann_projection_pooling` | 186,992 |
| 37 | `repr_neural_kernel_composition` | 117,131 |
| 38 | `repr_convex_potential_gradient_flow` | 78,578 |
| 39 | `n3_recursive_neighbor_volumes` | 119,201 |
| 40 | `pointcnn_x_transformed_evidence` | 114,624 |

No performance ranking or automatic multi-seed promotion follows from this code delivery.
